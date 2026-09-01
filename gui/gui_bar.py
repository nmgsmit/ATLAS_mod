"""Boxes over on-screen GUI junk the frame extraction left behind (a status bar, an
overlay strip) -- drawn by hand, carried through the clip, then painted out.

Mode -> "Scale annotation", arm "COVER GUI", drag a box over the junk. TRACK (the same
button the references use) carries the boxes through the video, and BLANK FRAMES paints
them black into <workspace>/images -- which is what everything downstream reads, so the
bar is gone for the networks and the export both.

Tracking watches the box's INSIDE and nothing else. A GUI bar is painted at a fixed
place on the screen by the console, so the box does not need to be followed -- it needs
to be CHECKED: the same rectangle is copied onto frame after frame, and its interior is
compared against the frame you drew it on. While the bar is there that interior is the
same pixels every frame. As soon as it changes -- the bar disappears, the console redraws
it somewhere else, the picture underneath comes through -- the run STOPS at that frame
and says so, rather than blanking tissue for the rest of the clip.

CHANGE_TOL is the knob: mean absolute difference in gray levels (0-255) between the
box's interior now and on the seed frame. It is set low on purpose -- a static overlay
against a moving picture is a big difference, so stopping early costs you one TRACK from
the next frame, while not stopping costs you blanked tissue you have to spot by eye.

Run the self-check with:  python -m gui.gui_bar
"""
import json
from os import path

import cv2
import numpy as np

# --- CONFIG ---------------------------------------------------------------
BARS_FILE = 'gui_bars.json'     # saved inside the workspace
COLOR = (255, 0, 255)           # RGB; magenta, used by nothing else on the overlay
THICKNESS = 2
MIN_SIDE = 4                    # px; a smaller drag is a stray click, not a box
INSET = 0.15                    # fraction of the box trimmed off each side before the
                                # interior is read, so the border (which blends into
                                # whatever is behind it) never enters the comparison
PATCH = 32                      # the interior is compared at this size, in gray levels
CHANGE_TOL = 6.0                # mean abs gray-level change that ends the tracking run
# --------------------------------------------------------------------------


class Bar:
    """One axis-aligned box, in image pixels."""

    def __init__(self, x0, y0, x1, y1, source='manual'):
        self.x0, self.x1 = sorted((float(x0), float(x1)))
        self.y0, self.y1 = sorted((float(y0), float(y1)))
        self.source = source        # 'manual' (a TRACK keyframe) | 'tracked'

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def h(self):
        return self.y1 - self.y0

    def big_enough(self):
        return self.w >= MIN_SIDE and self.h >= MIN_SIDE

    def rect(self):
        """(x0, y0, x1, y1) as ints, for cv2."""
        return (int(round(self.x0)), int(round(self.y0)),
                int(round(self.x1)), int(round(self.y1)))

    def contains(self, x, y):
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def interior(self, image):
        """The box's inside as a fixed-size gray patch -- what TRACK compares. None if
        the box has no interior left once INSET is trimmed off."""
        x0 = int(round(self.x0 + INSET * self.w))
        x1 = int(round(self.x1 - INSET * self.w))
        y0 = int(round(self.y0 + INSET * self.h))
        y1 = int(round(self.y1 - INSET * self.h))
        h, w = image.shape[:2]
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)
        if x1 - x0 < 1 or y1 - y0 < 1:
            return None
        patch = image[y0:y1, x0:x1]
        if patch.ndim == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        return cv2.resize(patch, (PATCH, PATCH),
                          interpolation=cv2.INTER_AREA).astype(np.float32)

    def copy(self, source='tracked'):
        """The same rectangle again, marked as carried by TRACK rather than drawn."""
        return Bar(self.x0, self.y0, self.x1, self.y1, source)

    def as_list(self):
        return [self.x0, self.y0, self.x1, self.y1, self.source]


def changed(patch, seed_patch):
    """How far the box's interior has drifted from the frame it was drawn on, in mean
    gray levels. inf when either patch is missing, so a box that fell off the image ends
    the run instead of being carried blind."""
    if patch is None or seed_patch is None:
        return float('inf')
    return float(np.abs(patch - seed_patch).mean())


def draw(vis, bars, pending=None):
    """Outline the boxes on the overlay (filled preview while one is being dragged)."""
    for bar in bars:
        x0, y0, x1, y1 = bar.rect()
        cv2.rectangle(vis, (x0, y0), (x1, y1), COLOR,
                      THICKNESS if bar.source == 'manual' else 1)
    if pending is not None:
        x0, y0, x1, y1 = pending.rect()
        cv2.rectangle(vis, (x0, y0), (x1, y1), COLOR, 1)


def blank(image, bars):
    """Paint the boxes black into an image, in place. Returns it."""
    for bar in bars:
        x0, y0, x1, y1 = bar.rect()
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 0), -1)
    return image


def save(workspace, bars_by_frame):
    with open(path.join(workspace, BARS_FILE), 'w') as f:
        json.dump({'version': 1,
                   'bars': {str(ti): [b.as_list() for b in bars]
                            for ti, bars in sorted(bars_by_frame.items()) if bars}},
                  f, indent=2)


def load(workspace):
    """{frame_index: [Bar, ...]}, empty if this workspace has none."""
    file = path.join(workspace, BARS_FILE)
    if not path.exists(file):
        return {}
    try:
        with open(file) as f:
            data = json.load(f)
    except (ValueError, OSError):
        return {}
    return {int(ti): [Bar(*b[:4], source=b[4] if len(b) > 4 else 'tracked') for b in bars]
            for ti, bars in data.get('bars', {}).items() if bars}


if __name__ == '__main__':
    import tempfile

    bar = Bar(30, 8, 10, 2)                  # corners in any order
    assert bar.rect() == (10, 2, 30, 8)
    assert bar.big_enough() and not Bar(0, 0, 2, 100).big_enough()
    # the interior comparison: same picture under the box reads 0, a changed one reads
    # well past the tolerance, and a box with no interior left ends the run
    im = np.full((20, 40, 3), 200, np.uint8)
    im[2:9, 10:31] = 60                      # the "bar" under the box
    seed = bar.interior(im)
    assert seed is not None and seed.shape == (PATCH, PATCH)
    assert changed(bar.interior(im), seed) == 0.0
    moved = im.copy()
    moved[2:9, 10:31] = 90                   # the bar redrawn / gone: 30 gray levels
    assert changed(bar.interior(moved), seed) > CHANGE_TOL
    assert changed(None, seed) == float('inf')
    assert Bar(5, 5, 9, 9).interior(np.zeros((3, 3, 3), np.uint8)) is None, 'off-image'

    c = bar.copy()
    assert c.rect() == bar.rect() and c.source == 'tracked'

    im = np.full((20, 40, 3), 200, np.uint8)
    blank(im, [bar])
    assert im[2:9, 10:31].max() == 0 and im[0, 0].min() == 200, 'wrong pixels blanked'

    with tempfile.TemporaryDirectory() as d:
        assert load(d) == {}
        save(d, {3: [bar], 5: []})
        got = load(d)
        assert list(got) == [3] and got[3][0].rect() == bar.rect()
        assert got[3][0].source == 'manual'

    print('[gui_bar] ok | box %s | interior %dx%d, stops at %.1f gray levels'
          % (bar.rect(), PATCH, PATCH, CHANGE_TOL))
