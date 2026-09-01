"""Cut the black bars off the endoscope picture.

The console writes a 1340x1072 picture into a 1920x1080 file and pads the rest with
black. Those bars are not picture, and keeping them costs more than screen space --
everything downstream measures pixels. The segmentation networks spend a fifth of their
input resolution on black; max_overall_size scales to a short edge that is partly
padding; and, most concretely, an instrument that runs off the side of the picture no
longer touches the image border, which is exactly the anchor gui/robot_arm.py pins its
centerline to.

So the bars are cut once, when the frames are extracted, and never seen again.

The box is a CONSTANT, not something detected per clip. Measured over 75 recordings the
picture lands in the same place every time; per-clip detection only ever reproduced it,
give or take the two px of codec ringing that blends bar into picture at the boundary --
wobble, not information. A source that is not the size we know is left alone entirely.

The box is written to <workspace>/source_crop.json so the mapping back to the original
video's pixels is never lost: an annotation at (x, y) in the workspace was at
(x * crop_w / out_w + crop_x, y * crop_h / out_h + crop_y) in the source.

Run the self-check with:  python -m gui.border_crop
"""
import json
from os import path

# --- CONFIG ---------------------------------------------------------------
# ponytail: one known source geometry, not a table. Add a second entry (and make
# crop_for a dict lookup) if a different console ever shows up.
SOURCE_SIZE = (1920, 1080)          # w, h of the recordings this box belongs to
BOX = (289, 4, 1340, 1072)          # x, y, w, h -- the picture inside those recordings.
                                    # Measured clean picture is cols 288-1629, rows
                                    # 4-1075; this sits 1 px inside that on the sides, so
                                    # no ringing or half-blended edge line survives.
CROP_FILE = 'source_crop.json'      # written inside the workspace
# --------------------------------------------------------------------------


def crop_for(source_size):
    """(x, y, w, h) picture box for a source of this (w, h), or None to crop nothing."""
    return BOX if tuple(source_size) == SOURCE_SIZE else None


def apply_crop(frame, box):
    """frame[y:y+h, x:x+w]. A None box means the frame is already the picture."""
    if box is None:
        return frame
    x, y, w, h = box
    return frame[y:y + h, x:x + w]


def describe(box, size):
    """One line saying what was cut, for the console."""
    w, h = size
    if box is None:
        return f'no crop for {w}x{h} (kept as is)'
    x, y, cw, ch = box
    sides = ', '.join(f'{n} {name}' for name, n in
                      (('left', x), ('right', w - x - cw), ('top', y), ('bottom', h - y - ch))
                      if n)
    return f'cut {sides} px of black border: {w}x{h} -> {cw}x{ch}'


def save(workspace, box, source_size, output_size):
    """Record the crop so workspace pixels can be mapped back to the source video."""
    x, y, w, h = box if box is not None else (0, 0, *source_size)
    with open(path.join(workspace, CROP_FILE), 'w') as f:
        json.dump({'version': 1,
                   'source_size': list(source_size),      # [w, h] of the video/images
                   'crop': {'x': x, 'y': y, 'w': w, 'h': h},
                   'output_size': list(output_size)}, f, indent=2)   # [w, h] written


def load(workspace):
    """The recorded crop dict, or None if this workspace predates the cropping (its
    images are whatever was extracted at the time)."""
    file = path.join(workspace, CROP_FILE)
    if not path.exists(file):
        return None
    try:
        with open(file) as f:
            return json.load(f)
    except (ValueError, OSError):
        return None


def to_source(x, y, info):
    """Map a workspace pixel back to the original video's pixels, using load()'s dict."""
    if not info:
        return float(x), float(y)
    c, out = info['crop'], info['output_size']
    return (x * c['w'] / out[0] + c['x'], y * c['h'] / out[1] + c['y'])


if __name__ == '__main__':
    import numpy as np

    # the box has to fit inside the frame it claims to describe
    x, y, w, h = BOX
    assert 0 <= x and x + w <= SOURCE_SIZE[0] and 0 <= y and y + h <= SOURCE_SIZE[1], BOX

    assert crop_for(SOURCE_SIZE) == BOX
    assert crop_for((1280, 720)) is None, 'an unknown size must be left alone'
    assert crop_for([1920, 1080]) == BOX, 'lists are accepted alongside tuples'

    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[y:y + h, x:x + w] = 200
    out = apply_crop(frame, crop_for((1920, 1080)))
    assert out.shape == (1072, 1340, 3)
    assert out.min() == 200, 'a bar or blended edge line survived the crop'
    assert apply_crop(frame, None) is frame

    assert describe(BOX, SOURCE_SIZE) == ('cut 289 left, 291 right, 4 top, 4 bottom px '
                                          'of black border: 1920x1080 -> 1340x1072')
    assert describe(None, (1280, 720)) == 'no crop for 1280x720 (kept as is)'

    # save/load and the mapping back to source pixels, including the resize
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        assert load(d) is None
        assert to_source(10, 20, None) == (10.0, 20.0)      # no record: identity
        save(d, BOX, SOURCE_SIZE, (1340, 1072))
        info = load(d)
        assert info['crop'] == {'x': 289, 'y': 4, 'w': 1340, 'h': 1072}
        assert to_source(0, 0, info) == (289.0, 4.0)        # workspace origin
        assert to_source(1340, 1072, info) == (1629.0, 1076.0)
        save(d, BOX, SOURCE_SIZE, (670, 536))               # cropped then halved
        assert to_source(0, 0, load(d)) == (289.0, 4.0)
        assert to_source(670, 536, load(d)) == (1629.0, 1076.0)
        save(d, None, (1280, 720), (1280, 720))             # nothing cut, still recorded
        assert to_source(5, 5, load(d)) == (5.0, 5.0)

    print('[border_crop] ok | %s | workspace(0,0) maps back to source %s'
          % (describe(BOX, SOURCE_SIZE),
             to_source(0, 0, {'crop': dict(zip('xywh', BOX)), 'output_size': [1340, 1072]})))
