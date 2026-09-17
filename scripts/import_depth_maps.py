"""Import precomputed EndoDAC depth maps into a clip's GUI workspace.

The estimator writes one frame_NNNNNN.npz per sampled frame, holding metric depth
(mm) at the network's own resolution. The GUI reads depth the way it writes it --
<workspace>/depth/<frame name>.png, 8-bit, at frame resolution -- so each npz is
resized to the frame and scaled into 1..255 using the CLIP's depth range, not the
frame's, so the colour of a given depth does not drift between frames. The npz is
copied along too, since the millimetres are what any later measurement wants.

    python scripts/import_depth_maps.py <depth_src_dir> <workspace_clip_dir>
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


def main(src_dir, ws_dir):
    src, ws = Path(src_dir), Path(ws_dir)
    npzs = sorted(src.glob('frame_*.npz'))
    if not npzs:
        sys.exit(f'no frame_*.npz in {src}')

    images = ws / 'images'
    out = ws / 'depth'
    out.mkdir(parents=True, exist_ok=True)

    depths = {p: np.load(p)['depth'].astype(np.float32) for p in npzs}
    lo = min(float(d.min()) for d in depths.values())
    hi = max(float(d.max()) for d in depths.values())
    span = max(hi - lo, 1e-6)

    written = 0
    for p, depth in depths.items():
        name = f'{int(p.stem.split("_")[1]):07d}'      # frame_000763 -> 0000763
        frame = images / f'{name}.jpg'
        if not frame.exists():
            print(f'skip {p.name}: no {frame.name} in the workspace')
            continue
        h, w = cv2.imread(str(frame)).shape[:2]
        norm = (depth - lo) / span
        # 1..255: 0 is the GUI's "no depth here" value, so no valid pixel may land on it
        eight = (1 + norm * 254).round().astype(np.uint8)
        cv2.imwrite(str(out / f'{name}.png'),
                    cv2.resize(eight, (w, h), interpolation=cv2.INTER_LINEAR))
        shutil.copy2(p, out / f'{name}.npz')
        written += 1

    (out / 'depth_range.json').write_text(json.dumps(
        {'min_mm': lo, 'max_mm': hi, 'note': 'the 1..255 png scale spans this range'},
        indent=2))
    print(f'{written} depth maps -> {out}  ({lo:.1f}-{hi:.1f} mm)')


if __name__ == '__main__':
    main(*sys.argv[1:3])
