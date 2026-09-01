"""Scale-reference objects for the ATLAS GUI (Mode -> "Scale annotation").

Three objects of KNOWN physical length, annotated per frame so a depth-estimation loss
can compare a predicted metric length against the truth:

    1 Ruler        -- a span picked off the surgical ruler, 10 mm (1 cm) by default
    2 Catheter tip -- always 16/3 mm
    3 Robot arm    -- always 8 mm, and NOT drawn by hand: it is the arm's diameter,
                     read square to the centerline between four points (where the arm
                     enters the picture, and where its straight shaft ends). That tool
                     lives in gui/robot_arm.py; this module only stores what it produces.

Geometry (deliberately simpler than the Retzius arch in gui/retzius_arch.py): each
object is a STRAIGHT segment -- two endpoints, no tip and no sharpness. Along it sit
N_TRACK probe points, stored as parameters t in [0,1] rather than as free 2D points, so
a probe can never leave the line and the object stays straight by construction. The
probes are what TRACK follows; by default they are spread evenly (both ends included)
and the user never sees them as handles. Turning "Edit tracking points" on (OFF by
default) makes the interior probes draggable, so one that sits on a bad patch --
specular highlight, an instrument edge -- can be slid to a better spot along the line.

Objects are stored PER FRAME, PER CLASS (one of each class per frame). TRACK carries
them through the video in a pass of its own, exactly like the arch: probes are followed
with segmenter.PointTracker (CSRT/LK + occlusion guard), then the segment is re-fitted
to the surviving probes with a weighted similarity (per-step scale clamp) under an IRLS
re-weighting, so a probe that jumps somewhere the others contradict is dropped instead
of dragging the segment. The robust machinery is imported from retzius_arch rather than
copied -- only the shape being fitted differs.

The robot arm does NOT take part in any of that: it has no probes, and its four points
are tracked by gui/robot_arm.py's own pass instead. It reaches this module only at save
time, as the chord its diameter spans.

The GUI wiring (clicks, dragging, class picking, the TRACK loop) lives in
main_controller.py; objects persist to <workspace>/scale_objects.json.

Run the self-check with:  python -m gui.scale_objects
"""
import json
from os import path

import cv2
import numpy as np

from gui import robot_arm
# The robust-fitting machinery is shared with the arch: same probes, same trust model,
# only the fitted shape differs (a segment instead of a parabola).
from gui.retzius_arch import (BLOCK_MARGIN, HANDLE_R, GRAB_PX, HOLD_COLOR, IRLS_ITERS,
                              LOW_CONF, MAX_HOLD_FRAMES, MIN_EFF_PROBES, SCALE_CLIP,
                              SEED_MIN_VIS, SHAKY_COLOR, SIGMA_CEIL_FRAC, SIGMA_FLOOR,
                              THICKNESS, _robust_sigma, _tukey, fit_similarity,
                              probe_visibility)

# --- CONFIG ---------------------------------------------------------------
# The annotatable references. 'mm' is the physical length the segment spans; only the
# ruler's is editable (you choose how much of the ruler to span -- 1 cm by default),
# the other two are fixed properties of the instrument. Colours are RGB and are picked
# to stay legible on tissue and apart from each other; catheter green echoes the
# 'Catheter' segmentation class in gui/cutie/utils/palette.py.
CLASSES = {
    1: {'name': 'Ruler',        'mm': 10.0,      'fixed_mm': False, 'color': (0, 200, 255)},
    2: {'name': 'Catheter tip', 'mm': 16.0 / 3,  'fixed_mm': True,  'color': (0, 255, 0)},
    3: {'name': 'Robot arm',    'mm': robot_arm.ARM_MM,
                                                 'fixed_mm': True,  'color': robot_arm.ARM_COLOR},
}
ARM_CLASS = 3                  # the one class this module does not draw or track itself

N_TRACK = 5                    # probe points per object (ends included)
DEFAULT_TS = tuple(np.linspace(0.0, 1.0, N_TRACK))
T_MARGIN = 0.05                # interior probes stay this far from either end, so a probe
                               # handle can never hide under an endpoint handle
PROBE_R = 3                    # px; probe dot radius
PROBE_MARGIN = 8               # px; probes stay this far inside the image border
SCALE_FILE = 'scale_objects.json'   # saved inside the workspace

# Camera focal length in PIXELS, in the coordinate frame the annotations live in (the
# cropped frame -- source_crop.json is a pure crop, so f is the source video's f; only
# the principal point shifts). THIS IS THE CALIBRATION KNOB: mm_per_px alone is scale up
# to an unknown constant, and only f turns it into millimetres:
#     Z_mm = f_px * mm_per_px          (segment roughly perpendicular to the optical axis)
# Measure it once per scope (checkerboard, or the endoscope's spec sheet) and set it here.
# None means "not calibrated": anchors() then reports depth_mm=None instead of guessing.
FOCAL_PX = None
# Classes anchors() refuses to hand a loss function. 2 (catheter tip) is excluded because
# the annotated span disagrees with the ruler by a STABLE 5.1-5.8x across every frame the
# two share -- not tracker noise, the drawn span simply is not 16/3 mm. Drop the entry
# once the class has been re-annotated against a span that is.
EXCLUDE_CLASSES = {2}
# --------------------------------------------------------------------------


def class_name(cls_id):
    return CLASSES[cls_id]['name'] if cls_id in CLASSES else str(cls_id)


def default_mm(cls_id):
    return float(CLASSES[cls_id]['mm']) if cls_id in CLASSES else 1.0


class ScaleLine:
    """A straight segment of known physical length, with N_TRACK probes along it."""

    def __init__(self, cls_id, a, b, ts=None, mm=None, source='manual', conf=1.0):
        self.cls_id = int(cls_id)
        self.a = (float(a[0]), float(a[1]))
        self.b = (float(b[0]), float(b[1]))
        self.mm = default_mm(self.cls_id) if mm is None else float(mm)
        ts = DEFAULT_TS if ts is None else ts
        self.ts = [float(np.clip(t, 0.0, 1.0)) for t in ts]
        self.source = source     # 'manual' (user keyframe) | 'tracked' | 'hold' (lost) |
                                 # 'measured' (the arm's own chord)
        self.conf = float(conf)  # [0,1] mean probe trust behind this fit (1 if manual)

    def copy(self, **kw):
        """Same segment with a few fields replaced (source/conf on a hold, mostly)."""
        fields = dict(cls_id=self.cls_id, a=self.a, b=self.b, ts=list(self.ts),
                      mm=self.mm, source=self.source, conf=self.conf)
        fields.update(kw)
        return ScaleLine(**fields)

    # --- geometry ---------------------------------------------------------
    @property
    def length_px(self):
        return float(np.hypot(self.b[0] - self.a[0], self.b[1] - self.a[1]))

    @property
    def mm_per_px(self):
        """The whole point of the object: millimetres one pixel spans, here and now."""
        d = self.length_px
        return self.mm / d if d > 1e-6 else 0.0

    def at(self, ts):
        """(n,2) points on the segment at parameters ts in [0,1] (0 = a, 1 = b)."""
        a = np.asarray(self.a, np.float64)
        b = np.asarray(self.b, np.float64)
        t = np.asarray(ts, np.float64).reshape(-1, 1)
        return a + t * (b - a)

    def probes(self):
        """The N_TRACK probe positions, in order."""
        return [(float(p[0]), float(p[1])) for p in self.at(self.ts)]

    def project_t(self, x, y):
        """Parameter of the point on the segment's line nearest (x, y), clamped to [0,1]."""
        a = np.asarray(self.a, np.float64)
        ab = np.asarray(self.b, np.float64) - a
        den = float(ab @ ab)
        if den < 1e-9:
            return 0.0
        return float(np.clip((np.asarray([x, y], np.float64) - a) @ ab / den, 0.0, 1.0))

    def handles(self, edit_points=False):
        """(name, point) pairs the mouse can grab. Endpoints always; the interior probes
        only while "Edit tracking points" is on -- the ones at t=0/t=1 ARE the endpoints,
        so they are never offered separately."""
        out = [('a', self.a), ('b', self.b)]
        if edit_points:
            pts = self.probes()
            out += [(f'p{i}', pts[i]) for i, t in enumerate(self.ts) if 0.0 < t < 1.0]
        return out

    # --- persistence ------------------------------------------------------
    def to_dict(self):
        """One JSON record. length_px / mm_per_px are derived, and written out anyway so
        a loss function can read the scale straight off without redoing the geometry."""
        return {
            'class_id': self.cls_id,
            'class_name': class_name(self.cls_id),
            'mm': self.mm,
            'a': list(self.a),
            'b': list(self.b),
            'ts': list(self.ts),
            'points': [list(p) for p in self.probes()],
            'length_px': round(self.length_px, 4),
            'mm_per_px': round(self.mm_per_px, 8),
            'source': self.source,
            'conf': round(self.conf, 4),
        }

    @staticmethod
    def from_dict(d):
        return ScaleLine(d['class_id'], d['a'], d['b'], d.get('ts'), d.get('mm'),
                         d.get('source', 'manual'), d.get('conf', 1.0))


def hit_test(lines, x, y, radius=GRAB_PX, edit_points=False):
    """Nearest (line, handle_name) within radius of (x, y), else None."""
    best, best_d = None, float(radius)
    for line in lines:
        for name, (px, py) in line.handles(edit_points):
            dist = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
            if dist <= best_d:
                best, best_d = (line, name), dist
    return best


def move_handle(line, name, x, y):
    """Drag a handle. An endpoint takes the cursor; a probe is PROJECTED onto the line,
    so it slides along the segment and the object stays straight."""
    if name in ('a', 'b'):
        setattr(line, name, (float(x), float(y)))
    else:
        i = int(name[1:])
        line.ts[i] = float(np.clip(line.project_t(x, y), T_MARGIN, 1.0 - T_MARGIN))


def sample_probes(line, w, h, margin=PROBE_MARGIN, blocked=None):
    """(t_list, point_list) of the probes that are actually trackable on this frame:
    inside the image and not sitting on an instrument. Probes off-image or behind an
    occluder have nothing to follow, so they are left out and the survivors carry the
    segment (the same rule the arch uses for its probes)."""
    ts = np.asarray(line.ts, np.float64)
    pts = line.at(ts)
    ok = ((pts[:, 0] >= margin) & (pts[:, 0] <= w - 1 - margin) &
          (pts[:, 1] >= margin) & (pts[:, 1] <= h - 1 - margin))
    if blocked is not None:
        ok &= probe_visibility(blocked, pts) >= SEED_MIN_VIS
    ts, pts = ts[ok], pts[ok]
    return ts.tolist(), [(float(p[0]), float(p[1])) for p in pts]


def _fit_once(line, old, P, w, scale_clip):
    """One weighted similarity fit of the segment onto probe positions P: the transform
    from the probes' old positions to their new ones, applied to both endpoints. The
    per-step scale is clamped so the segment cannot 'breathe' when the evidence bunches
    up -- but it is NOT pinned, because the pixel length IS the measurement (it grows as
    the reference comes toward the camera)."""
    s, R, T = fit_similarity(old, P, w)
    s = float(np.clip(s, *scale_clip))
    wn = w / w.sum()                       # re-derive T so centroids match at clipped scale
    T = (wn[:, None] * P).sum(0) - s * (R @ (wn[:, None] * old).sum(0))
    a = s * (R @ np.asarray(line.a, np.float64)) + T
    b = s * (R @ np.asarray(line.b, np.float64)) + T
    return line.copy(a=a, b=b, source='tracked', conf=1.0)


def refit(line, ts, new_pts, weights, scale_clip=SCALE_CLIP, iters=IRLS_ITERS):
    """New ScaleLine fitted to tracked probes, or None if they disagree too much to fit
    at all (the caller should then hold the previous pose). ts are the probes' segment
    parameters (from sample_probes), new_pts their tracked positions, weights their
    appearance x visibility trust.

    Robust in exactly the way refit() in retzius_arch is: fit, measure how far each probe
    sits from the fitted segment, downweight the ones that disagree with the consensus,
    refit. A probe that still looks right but jumped off the line ends at weight 0."""
    ts = np.asarray(ts, np.float64)
    P = np.asarray(new_pts, np.float64)
    w0 = np.clip(np.asarray(weights, np.float64), 0.0, 1.0)
    old = line.at(ts)
    # tolerance adapts to the frame but is bounded: never tighter than tracker jitter,
    # never looser than a fraction of the segment's own half-length
    sigma_ceil = max(SIGMA_CEIL_FRAC * line.length_px / 2, SIGMA_FLOOR)

    w = w0.copy()
    out = None
    for _ in range(max(1, iters)):
        if w.sum() < 1e-6:
            return None                    # nothing left to fit to
        out = _fit_once(line, old, P, w, scale_clip)
        r = np.linalg.norm(out.at(ts) - P, axis=1)
        sigma = float(np.clip(_robust_sigma(r), SIGMA_FLOOR, sigma_ceil))
        w = w0 * _tukey(r, sigma)

    if out is None or w.sum() < 1e-6:
        return None
    # Kish effective sample size: how many probes really vote once the disagreers are
    # discounted. Two is the minimum a similarity fit needs.
    if float(w.sum() ** 2 / (w ** 2).sum()) < MIN_EFF_PROBES:
        return None
    out.conf = float(np.clip(w.sum() / len(w), 0.0, 1.0))
    return out


def _pt(p):
    return (int(round(p[0])), int(round(p[1])))


def _line_color(line):
    """Class colour = trusted, amber = tracked but the probes disagreed (nudge it),
    gray = tracking lost its grip and this frame is holding the last pose."""
    if line.source == 'hold':
        return HOLD_COLOR
    if line.source == 'tracked' and line.conf < LOW_CONF:
        return SHAKY_COLOR
    return CLASSES.get(line.cls_id, {}).get('color', (255, 255, 255))


def draw(img, lines, editing=False, pending=(), pending_color=(255, 255, 255),
         edit_points=False):
    """Draw the segments onto img in place. editing adds the endpoint handles, the probe
    dots and a mm label; edit_points additionally rings the draggable interior probes.
    pending is the first endpoint click of an object being placed."""
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3:
        return  # e.g. torch tensor from the fast 'image' mode -- nothing to draw on
    for line in lines:
        color = _line_color(line)
        cv2.line(img, _pt(line.a), _pt(line.b), color, THICKNESS, cv2.LINE_AA)
        if not editing:
            continue
        for i, p in enumerate(line.probes()):
            cv2.circle(img, _pt(p), PROBE_R, color, -1, cv2.LINE_AA)
            if edit_points and 0.0 < line.ts[i] < 1.0:
                cv2.circle(img, _pt(p), PROBE_R + 2, (255, 255, 255), 1, cv2.LINE_AA)
        for _, p in (('a', line.a), ('b', line.b)):
            cv2.circle(img, _pt(p), HANDLE_R, color, -1, cv2.LINE_AA)
            cv2.circle(img, _pt(p), HANDLE_R, (255, 255, 255), 1, cv2.LINE_AA)
        label = f'{class_name(line.cls_id)} {line.mm:g} mm'
        cv2.putText(img, label, (_pt(line.a)[0] + 8, _pt(line.a)[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    for p in pending:
        cv2.circle(img, _pt(p), HANDLE_R - 1, pending_color, -1, cv2.LINE_AA)


def save(workspace, by_frame, class_mm=None, width=None, height=None,
         arm_by_frame=None, arm_seed=None, arm_calib=1.0):
    """Persist the annotations to <workspace>/scale_objects.json.

    by_frame is {frame_index: {class_id: ScaleLine}} for the hand-drawn references;
    arm_by_frame is {frame_index: robot_arm.ArmMeasure} for the measured robot arm.

    'frames' is the layout meant to be consumed directly by a depth-estimation loss:
    frame -> list of objects, each carrying its real length in mm, its endpoints, its
    probe points, and the mm/px the pair implies there. The arm joins that list as the
    chord its diameter spans -- ONLY on the frames you marked trusted, so every record
    under 'frames' is a measurement someone stands behind.

    'robot_arm' is the tool's own state (the four points and the verdict) for every frame
    it has looked at, trusted or not, so an untrusted frame keeps the points you placed on
    it -- and so a saved annotation can be re-read later without the segmentation, since
    the four points ARE the state. A loss function reads 'frames' and can ignore this
    section entirely.

    class_mm overrides the per-class default in the header (the ruler span is editable);
    width/height record the frame size the pixel coordinates belong to."""
    arm_by_frame = arm_by_frame or {}
    frames = {}
    for ti in sorted(set(by_frame) | set(arm_by_frame)):
        objs = {c: l for c, l in by_frame.get(ti, {}).items() if c != ARM_CLASS}
        ann = arm_by_frame.get(ti)
        line = None if ann is None else ann.to_scale_line()
        if line is not None:
            objs[ARM_CLASS] = line
        if objs:
            frames[str(ti)] = [objs[c].to_dict() for c in sorted(objs)]
    classes = {str(cid): {'name': spec['name'],
                          'mm': float((class_mm or {}).get(cid, spec['mm'])),
                          'fixed_mm': spec['fixed_mm']}
               for cid, spec in CLASSES.items()}
    # 'calibration' is the one multiplier for the whole clip, for a segmentation that is
    # systematically fat or thin -- the error no per-frame geometry can see, because a
    # dilated mask moves both sides of the arm together and the reading stays consistent
    # and wrong.
    arm = {'mm': robot_arm.ARM_MM, 'object_id': robot_arm.ARM_OBJECT_ID,
           'seed': None if arm_seed is None else list(arm_seed),
           'calibration': float(arm_calib),
           'smooth_window': robot_arm.SMOOTH_WIN,
           'frames': {str(ti): arm_by_frame[ti].to_dict() for ti in sorted(arm_by_frame)}}
    with open(path.join(workspace, SCALE_FILE), 'w') as f:
        json.dump({'version': 2, 'n_track_points': N_TRACK, 'focal_px': FOCAL_PX,
                   'frame_size': None if width is None or height is None
                   else [int(width), int(height)],
                   'classes': classes, 'frames': frames, 'robot_arm': arm}, f, indent=2)


def load(workspace):
    """(by_frame, class_mm, arm_by_frame, arm_seed, arm_calib) from the workspace. Missing
    or unreadable file -> empty annotations and the built-in class defaults.

    Class 3 is never returned in by_frame: the arm is owned by arm_by_frame, and the
    class-3 records under 'frames' are its export, regenerated on every save. An arm
    record from before the four-point tool loads as nothing (robot_arm.ArmMeasure.
    from_dict) -- press Measure clip to redo them; the rest of the references are
    unaffected."""
    defaults = {cid: float(spec['mm']) for cid, spec in CLASSES.items()}
    file = path.join(workspace, SCALE_FILE)
    if not path.exists(file):
        return {}, defaults, {}, None, 1.0
    try:
        with open(file) as f:
            data = json.load(f)
        by_frame = {}
        for ti, objs in data.get('frames', {}).items():
            frame = {}
            for d in objs:
                line = ScaleLine.from_dict(d)
                if line.cls_id in CLASSES and line.cls_id != ARM_CLASS:
                    frame[line.cls_id] = line
            if frame:
                by_frame[int(ti)] = frame
        for cid, spec in data.get('classes', {}).items():
            if int(cid) in defaults and 'mm' in spec:
                defaults[int(cid)] = float(spec['mm'])
        arm = data.get('robot_arm') or {}
        arm_by_frame = {}
        for ti, d in (arm.get('frames') or {}).items():
            ann = robot_arm.ArmMeasure.from_dict(d)
            if ann is not None:
                arm_by_frame[int(ti)] = ann
        seed = arm.get('seed')
        return (by_frame, defaults, arm_by_frame,
                None if seed is None else tuple(seed),
                float(arm.get('calibration', 1.0) or 1.0))
    except (ValueError, KeyError, TypeError, IndexError) as e:
        print(f'[scale_objects] could not read {file}: {e}')
        return {}, defaults, {}, None, 1.0


def anchors(workspace, focal_px=None):
    """The scale anchors of a workspace, filtered for a depth loss:
    [{frame, class_id, class_name, mm, points, length_px, mm_per_px, source, conf,
      depth_mm}], one entry per usable object per frame.

    Reads the 'frames' section ONLY. That matters for the arm: 'robot_arm.frames' is the
    tool's state for every frame it looked at, rejected ones included (mm_per_px there
    runs from 0.0 upwards), while 'frames' holds only the reconciled, trusted export.

    Dropped: EXCLUDE_CLASSES, source 'hold' (the tracker lost the object and the previous
    pose is being held -- a pose, not a measurement), and non-positive mm_per_px.

    depth_mm = focal_px * mm_per_px, or None when no focal length is known (argument
    first, else FOCAL_PX). It assumes the segment is roughly perpendicular to the optical
    axis -- a tilted span is foreshortened and reads too near -- and it is the depth AT
    THAT SEGMENT, not of the frame."""
    f = FOCAL_PX if focal_px is None else float(focal_px)
    file = path.join(workspace, SCALE_FILE)
    if not path.exists(file):
        return []
    try:
        with open(file) as fh:
            data = json.load(fh)
    except ValueError as e:
        print(f'[scale_objects] could not read {file}: {e}')
        return []
    out = []
    for ti, objs in sorted((data.get('frames') or {}).items(), key=lambda kv: int(kv[0])):
        for d in objs:
            mmpp = float(d.get('mm_per_px') or 0.0)
            if d.get('class_id') in EXCLUDE_CLASSES or d.get('source') == 'hold':
                continue
            if mmpp <= 0:
                continue
            rec = dict(d, frame=int(ti),
                       depth_mm=None if f is None else f * mmpp)
            out.append(rec)
    return out



if __name__ == '__main__':
    # geometry: endpoints exact, probes evenly spread and on the line, mm/px consistent
    ln = ScaleLine(2, (100, 100), (200, 100))
    assert ln.mm == 16 / 3 and abs(ln.length_px - 100) < 1e-9
    assert abs(ln.mm_per_px - (16 / 3) / 100) < 1e-12
    pts = ln.probes()
    assert len(pts) == N_TRACK and pts[0] == ln.a and pts[-1] == ln.b
    assert all(abs(p[1] - 100) < 1e-9 for p in pts)                 # all on the segment
    assert abs(pts[2][0] - 150) < 1e-9                              # midpoint
    assert ScaleLine(1, (0, 0), (0, 0)).mm_per_px == 0.0            # degenerate: no crash
    assert ScaleLine(1, (0, 0), (10, 0)).mm == 10.0                 # ruler default 1 cm
    assert ScaleLine(3, (0, 0), (10, 0)).mm == 8.0                  # robot arm always 8

    # a straight line stays straight: an endpoint drag moves only that end, and a probe
    # drag is projected back onto the line no matter how far off-line the cursor is
    move_handle(ln, 'b', 200, 160)
    assert ln.b == (200.0, 160.0) and ln.a == (100.0, 100.0)
    move_handle(ln, 'p1', 150, -400)                                # cursor way off-line
    p1 = ln.probes()[1]
    (vx, vy), (wx, wy) = np.subtract(ln.b, ln.a), np.subtract(p1, ln.a)
    assert abs(vx * wy - vy * wx) < 1e-6, p1                        # exactly on the line
    assert T_MARGIN <= ln.ts[1] <= 1 - T_MARGIN
    move_handle(ln, 'p1', *ln.a)                                    # dragged onto the end
    assert ln.ts[1] == T_MARGIN                                     # clamped off the handle

    # handles: endpoints always, interior probes only while point editing is on
    assert [n for n, _ in ln.handles()] == ['a', 'b']
    assert [n for n, _ in ln.handles(edit_points=True)] == ['a', 'b', 'p1', 'p2', 'p3']
    assert hit_test([ln], ln.a[0] + 3, ln.a[1] - 3, radius=8)[1] == 'a'
    assert hit_test([ln], -500, -500, radius=8) is None
    ln2 = ScaleLine(1, (10, 10), (210, 10))
    assert hit_test([ln2], *ln2.probes()[2], radius=8, edit_points=True)[1] == 'p2'
    assert hit_test([ln2], *ln2.probes()[2], radius=8) is None      # off by default

    # drawing clips off-image parts without complaint; label + handles + pending render
    img = np.zeros((240, 320, 3), np.uint8)
    draw(img, [ln2], editing=True, pending=[(10, 10)], edit_points=True)
    assert img.any()
    draw(np.zeros((2, 2, 3), np.uint8), [ScaleLine(1, (0, 0), (0, 0))])   # degenerate

    # colour reflects trust, per class
    assert _line_color(ScaleLine(2, (0, 0), (9, 0))) == CLASSES[2]['color']
    assert _line_color(ScaleLine(2, (0, 0), (9, 0), source='tracked', conf=0.1)) == SHAKY_COLOR
    assert _line_color(ScaleLine(2, (0, 0), (9, 0), source='hold', conf=0.0)) == HOLD_COLOR

    # probe sampling: only trackable probes, and an occluder band is avoided entirely
    ts_v, pts_v = sample_probes(ln2, 320, 240)
    assert len(ts_v) == N_TRACK
    blk = np.zeros((240, 320), bool)
    blk[:, 100:160] = True
    ts_b, pts_b = sample_probes(ln2, 320, 240, blocked=blk)
    assert 0 < len(ts_b) < N_TRACK
    for p in pts_b:
        assert not blk[int(round(p[1])), int(round(p[0]))], p
    assert sample_probes(ScaleLine(1, (-500, -500), (-400, -500)), 320, 240) == ([], [])

    # refit: probes moved by a rigid shift -> the whole segment follows, length intact
    ts_r, pts_r = sample_probes(ln2, 320, 240)
    moved = [(p[0] + 9, p[1] - 4) for p in pts_r]
    r1 = refit(ln2, ts_r, moved, np.ones(len(ts_r)))
    assert np.allclose(r1.a, (19, 6), atol=1e-6) and np.allclose(r1.b, (219, 6), atol=1e-6)
    assert r1.source == 'tracked' and r1.conf > 0.99 and r1.mm == ln2.mm
    assert abs(r1.length_px - ln2.length_px) < 1e-6

    # the pixel length is the measurement, so a genuine shrink IS followed (within the
    # per-step clamp) rather than pinned -- that is how mm/px tracks depth
    mid = np.array([110.0, 10.0])
    shrunk = [tuple(mid + 0.95 * (np.asarray(p) - mid)) for p in pts_r]
    r2 = refit(ln2, ts_r, shrunk, np.ones(len(ts_r)))
    assert abs(r2.length_px - 0.95 * ln2.length_px) < 1e-6
    assert r2.mm_per_px > ln2.mm_per_px                     # same mm over fewer px

    # ROBUSTNESS: one probe jumps far off with a PERFECT appearance score. The consensus
    # must ignore it entirely (this is the failure NCC alone cannot see).
    rogue = list(moved)
    rogue[2] = (rogue[2][0] + 40, rogue[2][1] - 50)
    r3 = refit(ln2, ts_r, rogue, np.ones(len(ts_r)))
    assert np.allclose(r3.a, r1.a, atol=0.5) and np.allclose(r3.b, r1.b, atol=0.5)
    assert r3.conf < r1.conf, 'a disagreeing probe must lower the frame confidence'
    # ... and a set of probes NO straight line can explain (a zig-zag) is refused
    # outright, so the caller holds the previous pose instead of inventing a fit
    zigzag = [(p[0], p[1] + (60 if i % 2 else -60)) for i, p in enumerate(pts_r)]
    assert refit(ln2, ts_r, zigzag, np.ones(len(ts_r))) is None
    assert refit(ln2, ts_r, moved, np.zeros(len(ts_r))) is None   # no appearance trust
    # a low-appearance probe is discounted even when it agrees perfectly
    r4 = refit(ln2, ts_r, moved, [0.2] * len(ts_r))
    assert r4 is not None and r4.conf < 0.25

    # save/load roundtrip: per-frame per-class, edited ruler mm, derived fields present
    import tempfile
    empty = ({}, {1: 10.0, 2: 16 / 3, 3: 8.0}, {}, None, 1.0)
    with tempfile.TemporaryDirectory() as d:
        assert load(d) == empty                                    # nothing saved yet
        ruler = ScaleLine(1, (5, 5), (105, 5), mm=20.0, source='tracked', conf=0.42)
        cath = ScaleLine(2, (10, 40), (60, 90))
        save(d, {7: {1: ruler, 2: cath}, 9: {}}, class_mm={1: 20.0},
             width=320, height=240)
        back, mms, arms, seed, calib = load(d)
        assert list(back) == [7] and sorted(back[7]) == [1, 2]     # empty frame dropped
        assert mms[1] == 20.0 and mms[2] == 16 / 3
        assert arms == {} and seed is None
        assert calib == 1.0                           # defaults for a fresh file
        b1 = back[7][1]
        assert b1.mm == 20.0 and b1.source == 'tracked' and abs(b1.conf - 0.42) < 1e-4
        assert b1.a == ruler.a and b1.b == ruler.b and b1.ts == ruler.ts
        with open(path.join(d, SCALE_FILE)) as f:
            raw = json.load(f)
        rec = raw['frames']['7'][0]
        assert rec['class_name'] == 'Ruler' and len(rec['points']) == N_TRACK
        assert abs(rec['mm_per_px'] - 20.0 / 100) < 1e-6           # ready for the loss fn
        assert raw['frame_size'] == [320, 240]

        # the robot arm: its own section always, but a class-3 measurement under
        # 'frames' only where it is trusted -- an untrusted frame still keeps its four
        # points, it just is not a measurement anyone can use
        good = robot_arm.ArmMeasure([(0, 90), (0, 110), (80, 90), (80, 110)])
        off = robot_arm.ArmMeasure([(0, 92), (0, 108), (60, 92), (60, 108)], manual=True)
        off.toggle(False)
        save(d, {7: {1: ruler}}, class_mm={1: 20.0}, width=320, height=240,
             arm_by_frame={7: good, 8: off}, arm_seed=(12, 34), arm_calib=1.05)
        back, _, arms, seed, calib = load(d)
        assert seed == (12.0, 34.0) and sorted(arms) == [7, 8]
        assert abs(calib - 1.05) < 1e-9                             # the clip-level knob
        assert arms[7].trusted and not arms[8].trusted and arms[8].manual
        assert abs(arms[8].diameter - 16.0) < 1e-6                 # kept, just not exported
        assert 3 not in back.get(7, {}), 'the arm is owned by arm_by_frame, not by_frame'
        with open(path.join(d, SCALE_FILE)) as f:
            raw = json.load(f)
        assert [r['class_id'] for r in raw['frames']['7']] == [1, 3]
        assert abs(raw['frames']['7'][1]['mm_per_px'] - robot_arm.ARM_MM / 20.0) < 1e-9
        assert '8' not in raw['frames'], 'an untrusted frame exports no measurement'
        assert raw['robot_arm']['frames']['8']['trusted_by'] == 'user'

        with open(path.join(d, SCALE_FILE), 'w') as f:
            f.write('{ not json')
        assert load(d) == empty                                    # corrupt file survives

    print('[scale_objects] ok | classes=%s | probes=%d'
          % ({c: f"{s['name']} {s['mm']:.3f}mm" for c, s in CLASSES.items()}, N_TRACK))
