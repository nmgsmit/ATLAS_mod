"""Mask-driven robot-arm diameter for the scale tool (Mode -> "Scale annotation", key 3).

The robot arm is a cylinder of known width (ARM_MM), so its apparent DIAMETER is a scale
reference exactly like the ruler's span -- but only if it is measured square to the shaft.
A cylinder read at any other angle comes out too wide, and there is no way to tell that
from the number alone.

FOUR POINTS say everything, and nothing else is fitted:

    entry pair    where the arm crosses the image border, one point per side
    (e0, e1)

    tip pair      the end of the STRAIGHT SHAFT, one point per side. The wrist -- NOT the
    (t0, t1)      far end of the tool, which flares and bends and is not 8 mm across.

They are guessed off the instrument mask and then dragged wherever the guess was wrong.
The guess walks the mask's perpendicular chord outward from the border, and the shaft ends
at the first station whose width leaves the shaft's own (the wrist flaring, or the jaws
opening) -- that chord is the tip pair.

The ENTRY pair is not read off the mask at the border, though, and this is the one place
the mask must not be believed: a segmentation does not draw the arm meeting the border at
an angle, it draws a curve into it, so the ends of its border run sit well inside the real
corner (40 px in, for a 40 px rounding). The arm's silhouette does not curve -- a
cylinder's sides are straight lines -- so each side is FITTED over the clean stretch of
shaft and followed back until it leaves the picture. Where it leaves is the corner. That
needs no assumption about which border it leaves through, which matters for an arm
crossing an image corner: its two sides go out through two different borders, and that is
exactly the case reading a single border run gets most wrong.

Everything drawn and exported follows from those four points:

    centerline    entry midpoint -> tip midpoint
    diameter      the tip pair's separation MEASURED ALONG THE PERPENDICULAR, so a tip
                  point dropped a little short or long ALONG the arm does not fatten the
                  reading -- only its distance ACROSS the arm counts
    scale line    a chord of that length across the tip midpoint, square to the
                  centerline. The one thing exported, and it sits at the deep end of the
                  shaft: the furthest point in the picture where 8 mm is something the
                  camera actually saw.

The entry pair is ONE-DIMENSIONAL: each of the two lives on the image edge it entered
across and only slides along it, whether you drag it or the tracker moves it. Across the
edge is not a choice anybody gets to make, so nothing is allowed to make it.

TRACK carries the four points with segmenter.PointTracker -- the same tracker the ruler
and the catheter tip use -- and puts the entry pair back on its edges after each step. A
frame you correct by hand is a keyframe: the tracker re-seeds there, and TRACK never
overwrites it. The run STOPS the moment an entry point moves further in one frame than
the arm possibly could (ENTRY_DRIFT_PX): the entry pair is the far end of a long rigid
arm and slides a few pixels a frame at most, so a bigger step is the tracker having let
go, and every frame after it would be measured from the wrong place.

What the tracker does NOT decide is the width. A cylinder's two edges are smooth along
their length, so no tracker can tell how far apart they have got -- handed a pair of edge
points it carries them along at the width it was given, and an arm coming visibly closer
reports the same diameter on every frame. So on each tracked frame the tip pair is put
back onto the MASK's own chord at the station the tracker left it at (snap_width): the
tracker chooses where along the arm to measure, the segmentation says how wide the arm is
there, and the millimetres per pixel move with the arm the way they should.

Every frame exports its OWN reading. The neighbouring frames decide WHETHER a frame is
exported, never WHAT it exports: the arm is a per-frame scale paired with a per-frame
depth map, and a neighbourhood median would hand a depth estimator the scale of a
different moment -- on a moving camera, smoothing away the very change the reference is
there to record.

Trust is yours, but it does not have to be checked by hand. clip_scale() drops a frame
that disagrees with its NEIGHBOURS (noise) and, separately, one whose reading has JUMPED
away from the last frame anybody believed (the points coming off the shaft). The second
test is the one that matters on a tracked clip: when the points slide onto something else
they stay there, so the neighbours agree and only the step from before the jump shows it.
Everything after an unrecovered jump is dropped too -- go to the frame the readout names,
put the four points back, and the chain re-anchors there. Every frame still carries a
verdict of its own, the automatic one only sets the initial state, and yours is final.

The GUI wiring (clicks, dragging, TRACK) lives in main_controller.py; annotations persist
inside <workspace>/scale_objects.json, in a 'robot_arm' section beside the ordinary
references (see scale_objects.save).

Run the self-check with:  python -m gui.robot_arm
"""
import cv2
import numpy as np

from gui.retzius_arch import GRAB_PX, HANDLE_R, HOLD_COLOR, OCCLUDER_OBJECT_ID, THICKNESS

# --- CONFIG ---------------------------------------------------------------
ARM_OBJECT_ID = OCCLUDER_OBJECT_ID   # 5, 'Non-anatomical' -- the instrument masks you
                                     # already annotate and propagate. The arm is picked
                                     # out of them as one connected component, so a second
                                     # instrument in frame does not confuse the guess.
ARM_NAME = 'Non-anatomical'          # what that object is called in the GUI's class list
ARM_MM = 8.0                         # the arm's real diameter; the whole point of measuring
ARM_COLOR = (255, 128, 0)            # RGB; matches scale_objects.CLASSES[3]
OFF_COLOR = (150, 150, 150)          # RGB; a frame whose diameter is not exported
CHORD_COLOR = (255, 255, 255)        # RGB; the scale line itself, so it reads against the
                                     # arm colour it is drawn across

# Guessing the tip pair: the mask's perpendicular chord, walked outward from the entry.
SHAFT_STEP = 12.0              # px along the axis between stations
MAX_STATIONS = 40              # ... but no more than this many over the whole arm: each
                               # one is a chord walk, and a long arm is not better
                               # understood for being sampled twice as often
REF_STATIONS = 5               # the leading stations whose median width IS the shaft's,
                               # so one bad chord near the border cannot set the reference
FLARE_TOL = 0.30               # relative departure from that width which ends the shaft.
                               # Loose on purpose: an arm in perspective genuinely tapers
                               # along its length, and only the wrist flares. On a steeply
                               # foreshortened arm the taper alone can spend this budget
                               # before the wrist does, so the guess stops short (measured:
                               # ~2/3 of the way down a 2:1 cone) -- conservative, since a
                               # short shaft still reads the right width where it stops,
                               # and the tip pair is draggable. Compare the STEP between
                               # neighbouring stations instead of the absolute departure if
                               # the guess ever needs to reach the wrist by itself.
MAX_MISSES = 3                 # consecutive stations off the mask before the walk gives
                               # up. Not 1: a notch in the boundary must not stop it
                               # short. Not unlimited: a second blob further down the axis
                               # is not this arm and must not be walked to.
# Placing the two corners, where the arm crosses the border. Read off the arm's SIDES
# rather than off the mask there: a segmentation rounds that corner into a curve, and the
# arm's silhouette does not curve -- it is a straight edge running into a straight edge.
CORNER_SKIP_FRAC = 0.5         # of the shaft's WIDTH: how deep the rounded stretch runs,
                               # and so how much of the shaft nearest the border is left
                               # out of the side fit. About the arm's own radius, which is
                               # what a mask's corner rounding tends to measure.
SIDE_MIN_PTS = 3               # stations a side needs before a line through it means
                               # anything. Below that the mask's own corner is used --
                               # rounded, but not invented.
CORNER_MAX_FRAC = 1.5          # of the width: how far the fitted corner may sit from where
                               # the mask meets the border before the fit is disbelieved.
                               # A sanity bound, not a precision one -- the correction is
                               # normally a few px, and anything way out is a broken fit
                               # rather than a better answer.

BACK_OFF = 1                   # stations to step back from the last good one. The chord
                               # before a flare is already half into the joint, and the
                               # last chord of an arm that simply ends clips its end cap
                               # and reads narrow -- either way the tip pair belongs one
                               # station short of where the walk stopped.

# Walking one chord out to the mask boundary.
CHORD_SUBPX = 0.25             # px; sampling step
CHORD_GAP_PX = 1.5             # px; a chord walks THROUGH gaps this short. A mask boundary
                               # is a staircase and a diagonal chord crosses its notches,
                               # so without this every oblique measurement reads ~1 px
                               # short on each side -- a systematic bias, not noise.
MAX_CHORD_PX = 400             # px; give up rather than walk forever off a broken mask
MIN_DIAM_PX = 4.0              # below this the mask is too thin to measure honestly

# Reconciling the clip. The arm is rigid and 8 mm, so its apparent width changes only as
# fast as the camera-arm distance does: a frame that disagrees with its neighbours is
# telling you about its own mask, not about the arm.
#
# Note what the neighbours are used FOR: deciding whether to believe a frame, and nothing
# else. Every frame exports its OWN reading. The arm is a per-frame scale reference paired
# with a per-frame depth map, so handing out a neighbourhood median would be answering a
# question nobody asked -- and on a moving camera it answers it wrongly, by smoothing away
# the very thing the reference is there to measure. Whoever consumes the export can smooth
# it if they want to; they cannot un-smooth it.
SMOOTH_WIN = 15                # frames of neighbouring MEASUREMENTS the local median is
                               # taken over, at MOST -- see _window, which shrinks it on a
                               # short annotation. Counted in measured frames, not frame
                               # numbers.
WINDOW_FRAC = 3                # ... and never wider than this fraction of the annotation.
                               # A window that spans the whole clip is not a rolling median
                               # at all, it is the global one: every frame then gets tested
                               # against the same number, a real trend reads as everybody
                               # being an outlier at the ends, and with the old smoothed
                               # export it made a 9-frame clip report one constant scale.
OUTLIER_MAD = 3.0              # a frame this many MADs off its local median is not exported
OUTLIER_FLOOR_PX = 0.5         # ... but never reject over less than this: half a pixel of
                               # disagreement on a rasterised boundary is not a fault.
OUTLIER_FLOOR_FRAC = 0.10      # ... nor is this fraction of the local median, whatever the
                               # MAD says. A rolling median LAGS a trend, so during a zoom
                               # -- the one time the arm's width legitimately marches in
                               # one direction -- every frame sits a little off its own
                               # median and the ends of the clip sit a lot off, and a
                               # tolerance chasing a small MAD throws them all away. The
                               # frames it would drop there are not wrong, they are early.
                               # Anything bigger than this band is the chain's job anyway.

# The entry pair's speed limit, which stops a TRACK run rather than dropping a frame.
# Where the arm crosses the border moves SLOWLY -- it is the far end of a long rigid arm
# pivoting somewhere off-screen, so even a hard camera move slides it a few pixels a
# frame. A bigger step than that is not the arm moving, it is the tracker having let go
# and landed somewhere else, and every frame after it would be measured from the wrong
# place. Better to stop and say so at the frame it happened than to carry on and leave a
# hundred frames to be found later.
ENTRY_DRIFT_PX = 8.0           # px one entry point may move between frames. Raise it for
                               # a clip with genuinely violent camera motion -- the number
                               # the run reports when it stops says how much it wanted.

# The continuity chain -- the test the neighbourhood one above cannot make. A rolling
# median compares a frame with its neighbours, so it catches a frame that is noisy and
# misses a frame that is WRONG: when the points come off the shaft and stay off, the
# neighbours come with them, the local median follows, and every frame from there on
# agrees beautifully about the wrong thing. What that jump does show up in is the STEP
# from the last frame anybody believed -- a rigid arm cannot change width abruptly, so a
# reading that has, is not a reading of the arm.
STEP_TOL = 0.15                # relative change from the last accepted frame that is
                               # still the same arm. Generous next to the ~1-2% a real
                               # zoom moves it between frames, because it has to pass the
                               # per-frame noise the rolling median is there to handle --
                               # this test is for the 30-100% jumps (the points landing on
                               # a second instrument, the tip walking into the jaws), not
                               # for the wobble.
STEP_GAP_MAX = 5               # the tolerance grows with the gap between measured frames,
                               # since the arm has had longer to move -- but only up to
                               # this many frames, or a long unmeasured stretch would let
                               # anything through on the far side of it.
# --------------------------------------------------------------------------


def _unit(v):
    n = float(np.hypot(v[0], v[1]))
    return np.asarray(v, np.float64) / n if n > 1e-9 else np.array([1.0, 0.0])


def _perp(u):
    return np.array([-u[1], u[0]])


def _pt(p):
    return (int(round(p[0])), int(round(p[1])))


# --- mask -> the arm ------------------------------------------------------

def arm_pixels(mask):
    """HxW bool of the instrument class in a GUI object-id mask (None -> None)."""
    if mask is None:
        return None
    m = np.asarray(mask) == ARM_OBJECT_ID
    return m if m.any() else None


def _border_labels(labels):
    """Label ids present on any image border (0 = background, excluded)."""
    edges = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    return set(int(v) for v in np.unique(edges) if v)


def pick_component(m, seeds=()):
    """The connected component that IS the arm: HxW bool, or None.

    Only components touching the image border are eligible -- the arm enters the frame,
    a free-floating blob of instrument mask is something else. Among those, the first
    seed point that lands on one wins (the click that identified the arm, or the previous
    frame's centerline, so the choice carries across the video); with no usable seed, the
    largest border component is taken."""
    if m is None or not m.any():
        return None
    n, labels = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    eligible = _border_labels(labels)
    if not eligible:
        return None
    h, w = labels.shape
    for s in seeds:
        if s is None:
            continue
        x, y = int(round(s[0])), int(round(s[1]))
        if 0 <= x < w and 0 <= y < h and int(labels[y, x]) in eligible:
            return labels == labels[y, x]
    best = max(eligible, key=lambda k: int((labels == k).sum()))
    return labels == best


def border_pair(comp):
    """The two ends of the arm's longest border run: ((x, y), (x, y)), or None.

    The arm is a tube crossing the image border, so on that border its mask is an interval
    -- and that interval's two ends are the two points where the arm enters the picture.
    The LONGEST run is used so a few pixels of the same component clipping a second edge
    cannot put the pair on the wrong border."""
    if comp is None or not comp.any():
        return None
    h, w = comp.shape
    # (edge pixels along the border, function turning an index on that edge into (x, y))
    edges = ((comp[0], lambda i: (float(i), 0.0)),
             (comp[-1], lambda i: (float(i), float(h - 1))),
             (comp[:, 0], lambda i: (0.0, float(i))),
             (comp[:, -1], lambda i: (float(w - 1), float(i))))
    best = None
    for line, to_xy in edges:
        i = 0
        while i < len(line):
            if not line[i]:
                i += 1
                continue
            j = i
            while j < len(line) and line[j]:
                j += 1
            if best is None or j - i > best[0]:
                best = (j - i, (to_xy(i), to_xy(j - 1)))
            i = j
    return None if best is None else best[1]


# --- the perpendicular chord ----------------------------------------------

def _inside(comp, p):
    h, w = comp.shape
    x, y = int(round(p[0])), int(round(p[1]))
    return 0 <= x < w and 0 <= y < h and bool(comp[y, x])


def _walk(comp, center, direction):
    """(boundary point, distance to it) walking from center along direction until the mask
    ends. Gaps shorter than CHORD_GAP_PX are walked through, so the staircase of a
    rasterised boundary does not cut an oblique chord short. The whole ray is sampled at
    once -- stepping it in Python costs ~800 array lookups per half-chord."""
    c = np.asarray(center, np.float64)
    h, w = comp.shape
    n = int(MAX_CHORD_PX / CHORD_SUBPX)
    d = np.arange(1, n + 1) * CHORD_SUBPX
    pts = c + d[:, None] * direction
    x = np.rint(pts[:, 0]).astype(np.int64)
    y = np.rint(pts[:, 1]).astype(np.int64)
    on = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    inside = np.zeros(n, bool)
    inside[on] = comp[y[on], x[on]]
    hits = np.flatnonzero(inside)
    if hits.size == 0:
        return c, 0.0
    # the run ends at the first gap too long to be a rasterisation notch
    breaks = np.flatnonzero(np.diff(hits) > int(round(CHORD_GAP_PX / CHORD_SUBPX)))
    last = int(hits[breaks[0]] if breaks.size else hits[-1])
    edge = (last + 1) * CHORD_SUBPX
    if last == n - 1:
        return c, float(MAX_CHORD_PX)       # never left the mask: caller treats as runaway
    edge -= 0.5 * CHORD_SUBPX               # the boundary sits between the samples
    return c + edge * direction, edge


def chord_at(comp, center, perp):
    """(p0, p1, length) of the mask's chord through center along +-perp, or None if center
    is off the mask or the chord runs away (a hole, or a broken component)."""
    if not _inside(comp, center):
        return None
    p0, d0 = _walk(comp, center, -perp)
    p1, d1 = _walk(comp, center, perp)
    if d0 >= MAX_CHORD_PX or d1 >= MAX_CHORD_PX:
        return None
    return (float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])), float(d0 + d1)


# --- the guess ------------------------------------------------------------

def _shaft(comp, mid, u):
    """The shaft, as the perpendicular chords along it: (chords, step, width) or None.

    Chords are collected outward from mid until the mask runs out, the shaft's own width
    is the median of the first few, and the list is cut at the first station that departs
    from that width by FLARE_TOL -- the wrist swelling, or the jaws opening into two
    half-width pieces that a single chord reads as one wide one. The last chord left is
    the tip; all of them together are the two sides."""
    n = _perp(u)
    reach = float(np.hypot(*comp.shape))
    step = max(SHAFT_STEP, reach / MAX_STATIONS)
    chords, misses, t = [], 0, step
    while t <= reach:
        c = chord_at(comp, np.asarray(mid, np.float64) + t * u, n)
        if c is None:
            misses += 1
            if misses >= MAX_MISSES:
                break
        else:
            misses = 0
            chords.append(c)
        t += step
    if not chords:
        return None
    ref = float(np.median([c[2] for c in chords[:REF_STATIONS]]))
    keep = len(chords)
    for i, c in enumerate(chords):
        if abs(c[2] - ref) > FLARE_TOL * ref:
            keep = i
            break
    return chords[:max(1, keep - BACK_OFF)], step, ref


def _exit(p, v, shape):
    """Where the ray from p along v leaves the image: (x, y), or None if it never does."""
    h, w = float(shape[0] - 1), float(shape[1] - 1)
    p = np.asarray(p, np.float64)
    best = None
    for axis, lim in ((0, 0.0), (0, w), (1, 0.0), (1, h)):
        if abs(v[axis]) < 1e-9:
            continue
        t = (lim - p[axis]) / v[axis]
        if t <= 0:
            continue
        q = p + t * v
        other = 1 - axis
        if -0.5 <= q[other] <= (w if other == 0 else h) + 0.5:
            if best is None or t < best[0]:
                best = (t, q)
    return None if best is None else (float(best[1][0]), float(best[1][1]))


def _corners(chords, step, width, u, shape, raw):
    """Where the arm crosses the image border, read off its SIDES instead of off the
    mask's own corner: (side 0's point, side 1's point), or None if there is too little
    clean shaft to fit.

    A segmentation rounds that corner off. The arm's side does not actually curve into the
    border -- it is a straight edge meeting a straight edge -- so the last stretch of mask
    before the border is the one place the mask is guaranteed to be wrong, and taking the
    corner from it puts the point a few pixels inside the picture, sitting on neither line.

    So each side of the shaft is fitted as the straight line a cylinder's silhouette is
    (skipping the rounded stretch, which is about the arm's own radius deep) and followed
    back out of the picture. Where it leaves IS the corner, and it needs no assumption
    about WHICH border that is -- an arm entering across an image corner has one side
    leaving through one border and the other side through the other, which is exactly
    where the mask's rounding is worst and the ends of a single border run are furthest
    from the truth.

    `raw` is the mask's own answer, kept as a sanity bound: a fit that puts the corner
    nowhere near where the mask does is not a better answer, it is a broken one."""
    skip = int(np.ceil(CORNER_SKIP_FRAC * width / step))
    skip = min(skip, max(0, len(chords) - SIDE_MIN_PTS))
    pts = chords[skip:]
    if len(pts) < SIDE_MIN_PTS:
        return None
    out = []
    for side in (0, 1):
        p = np.array([c[side] for c in pts], np.float32)
        # Huber rather than plain least squares: a nick in the mask boundary is one bad
        # point on an otherwise straight edge, and it must not tilt the whole line
        vx, vy, x0, y0 = cv2.fitLine(p, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        v = np.array([vx, vy])
        if float(v @ u) > 0:        # follow it BACK to the border, not on out along the arm
            v = -v
        q = _exit(np.array([x0, y0]), v, shape)
        if q is None or min(float(np.hypot(q[0] - r[0], q[1] - r[1]))
                            for r in raw) > CORNER_MAX_FRAC * width:
            return None
        out.append(q)
    return out[0], out[1]


def guess(comp):
    """The four points off the mask: (e0, e1, t0, t1), or None if there is no shaft here.

    Two passes: aim from the entry midpoint at the component's centroid, walk the shaft,
    then re-aim along what the walk found and walk it again. One re-aim is enough -- the
    centroid of a tube is already within a couple of degrees of its axis, and the second
    walk is aimed by the shaft itself.

    The two pairs are then PAIRED UP: e0 with t0 on one side of the arm, e1 with t1 on the
    other. They do not come out that way on their own -- the border run is read in the
    image's own order while a chord is read along the axis's normal, so which is 'first'
    flips with the direction the arm comes in from, and an arm entering from the right had
    its two sides drawn crossing over each other."""
    pair = border_pair(comp)
    if pair is None:
        return None
    e0, e1 = np.asarray(pair[0], np.float64), np.asarray(pair[1], np.float64)
    mid = 0.5 * (e0 + e1)
    ys, xs = np.nonzero(comp)
    u = _unit(np.array([xs.mean(), ys.mean()]) - mid)
    shaft = None
    for _ in range(2):
        shaft = _shaft(comp, mid, u)
        if shaft is None:
            return None
        tip = shaft[0][-1]
        u = _unit(0.5 * (np.asarray(tip[0]) + np.asarray(tip[1])) - mid)
    chords, step, width = shaft
    tip = chords[-1]
    # the corners come off the SIDES, not off the mask's rounded border run -- and when
    # they do they are already paired with the tip, both being read side-for-side
    corners = _corners(chords, step, width, u, comp.shape, (e0, e1))
    if corners is not None:
        return pair_up([corners[0], corners[1], tip[0], tip[1]])
    return pair_up([tuple(e0), tuple(e1), tip[0], tip[1]])


def pair_up(pts):
    """e0 with t0 on one side of the arm, e1 with t1 on the other: the four points with the
    tip pair swapped if the two sides would otherwise be drawn crossing over each other.

    Needed because the two pairs are read in unrelated orders -- the entry pair along the
    image border, the tip pair along the axis's own normal -- so which of the two counts as
    'first' flips with the direction the arm comes in from. Applied where the points are
    BUILT (off the mask, or off a saved record), never where they are moved: the tracker
    and the mouse both address a point by its index, and quietly renaming one mid-drag
    would be worse than an X on the screen."""
    p = [np.asarray(q, np.float64) for q in pts]
    n = _perp(_unit(0.5 * (p[2] + p[3]) - 0.5 * (p[0] + p[1])))
    if float((p[0] - p[1]) @ n) * float((p[2] - p[3]) @ n) < 0:
        p[2], p[3] = p[3], p[2]
    return tuple(tuple(float(z) for z in q) for q in p)


def edge_of(p, shape):
    """Which image edge p sits on (or is nearest): 0 left, 1 right, 2 top, 3 bottom."""
    h, w = shape[0], shape[1]
    return int(np.argmin([p[0], w - 1 - p[0], p[1], h - 1 - p[1]]))


def on_edge(p, edge, shape):
    """p put ON that image edge: the coordinate ACROSS the edge is pinned to it, the one
    ALONG it is kept (clamped into the picture).

    This is what makes an entry point one-dimensional. It is where the arm crosses the
    border, so it has exactly one degree of freedom -- where along that border -- and both
    the hand and the tracker are held to it. A tracker following a point at the picture's
    edge has half its template outside and creeps inward; a mouse cannot be dragged to a
    subpixel line at all. Neither has to be accurate across the edge, because across the
    edge is not a choice anybody gets to make."""
    h, w = shape[0], shape[1]
    x = float(np.clip(p[0], 0.0, w - 1))
    y = float(np.clip(p[1], 0.0, h - 1))
    return [(0.0, y), (float(w - 1), y), (x, 0.0), (x, float(h - 1))][edge]


# --- one frame's annotation -----------------------------------------------

class ArmMeasure:
    """One frame's robot arm: four points, and your verdict on the diameter they imply."""

    NAMES = ('e0', 'e1', 't0', 't1')     # entry pair, then tip pair

    def __init__(self, pts, trusted=None, trusted_by='auto', source='auto', manual=False):
        self.pts = [(float(p[0]), float(p[1])) for p in pts]
        if len(self.pts) != 4:
            raise ValueError('an arm is four points: entry pair then tip pair')
        self.source = source            # 'auto' | 'manual' | 'tracked' | 'hold'
        self.manual = bool(manual)      # placed by hand here -> a TRACK keyframe
        self.trusted_by = trusted_by    # 'auto' (the verdict) | 'user' (you said so)
        self.trusted = self.measured if trusted is None else bool(trusted and self.measured)
        # What this frame EXPORTS: its own reading, times the clip's calibration (see
        # clip_scale). None until the clip has been reconciled, and then the number that is
        # drawn and written out; self.diameter stays the uncalibrated reading, so the two
        # can always be compared.
        self.export_px = None

    # --- geometry ---------------------------------------------------------
    @property
    def start(self):
        """The entry: midpoint of the two points where the arm crosses the border."""
        return (0.5 * (self.pts[0][0] + self.pts[1][0]),
                0.5 * (self.pts[0][1] + self.pts[1][1]))

    @property
    def end(self):
        """The tip: midpoint of the two points at the end of the straight shaft."""
        return (0.5 * (self.pts[2][0] + self.pts[3][0]),
                0.5 * (self.pts[2][1] + self.pts[3][1]))

    @property
    def u(self):
        """Unit vector along the centerline, entry -> tip."""
        return _unit(np.asarray(self.end) - np.asarray(self.start))

    @property
    def n(self):
        """Unit vector square to the centerline -- the direction a diameter is read along."""
        return _perp(self.u)

    @property
    def length_px(self):
        return float(np.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1]))

    @property
    def diameter(self):
        """The tip pair's separation ACROSS the arm. The along-the-arm part of it is
        dropped, which is the whole reason this is a diameter and not just a distance: a
        tip point placed a little short or long still reads the same width."""
        v = np.asarray(self.pts[2]) - np.asarray(self.pts[3])
        return float(abs(v @ self.n))

    @property
    def measured(self):
        """There is a shaft here, wide enough and long enough to mean something."""
        return self.diameter >= MIN_DIAM_PX and self.length_px >= MIN_DIAM_PX

    @property
    def chord(self):
        """The scale line: the diameter, drawn square to the centerline at the tip."""
        c, half = np.asarray(self.end), 0.5 * self.diameter * self.n
        return (tuple(float(z) for z in c - half), tuple(float(z) for z in c + half))

    @property
    def mm_per_px(self):
        return ARM_MM / self.diameter if self.measured else 0.0

    # --- handles ----------------------------------------------------------
    def handles(self):
        """(name, point) pairs the mouse can grab -- all four, always."""
        return list(zip(self.NAMES, self.pts))

    def hit(self, x, y, radius=GRAB_PX):
        """Name of the handle under (x, y), or None."""
        best, best_d = None, float(radius)
        for name, (px, py) in self.handles():
            d = float(np.hypot(px - x, py - y))
            if d <= best_d:
                best, best_d = name, d
        return best

    def move(self, name, x, y, shape=None):
        """Drag one point. The frame becomes a keyframe: TRACK re-seeds here and never
        overwrites it.

        An ENTRY point only slides ALONG the image edge it is on -- one dimension, because
        that is all the arm's crossing of the border has. shape is (h, w); without it the
        point moves freely, which is only ever right for the tip pair."""
        i = self.NAMES.index(name)
        if i < 2 and shape is not None:
            self.pts[i] = on_edge((x, y), edge_of(self.pts[i], shape), shape)
        else:
            self.pts[i] = (float(x), float(y))
        self.manual = True
        self.source = 'manual'
        return self


    def points(self):
        """The four points as a list, for the tracker."""
        return list(self.pts)

    def moved_to(self, pts, source='tracked', shape=None):
        """The same annotation with its four points somewhere else -- what TRACK produces.

        Given shape (h, w), the entry pair is put back onto the edges THIS annotation's
        entry pair was on: the arm goes on entering the picture across the same border it
        crossed last frame, and a tracker is not entitled to an opinion about that.

        The verdict does NOT travel: it belongs to the frame it was given on, and the
        caller carries the target frame's own with adopt_verdict."""
        pts = list(pts)
        if shape is not None:
            for i in (0, 1):
                pts[i] = on_edge(pts[i], edge_of(self.pts[i], shape), shape)
        return ArmMeasure(pts, source=source)

    # --- the verdict ------------------------------------------------------
    def toggle(self, on=None):
        """Turn the diameter on/off for this frame. Your call sticks: from here on the
        automatic verdict never touches this frame again."""
        want = (not self.trusted) if on is None else bool(on)
        self.trusted = bool(want and self.measured)
        self.trusted_by = 'user'
        return self.trusted

    def adopt_verdict(self, other):
        """Carry a decision YOU made from an earlier measurement of the same frame.
        Re-measuring builds a new object, and a frame you switched off must not come back
        on because something re-ran over it. Automatic verdicts are not carried: those are
        exactly what a fresh measurement is entitled to redecide."""
        if other is not None and other.trusted_by == 'user':
            self.trusted_by = 'user'
            self.trusted = bool(other.trusted and self.measured)
        return self

    # --- export -----------------------------------------------------------
    @property
    def export_diameter(self):
        """The diameter that is drawn and written out: this frame's own, calibrated."""
        return float(self.export_px) if self.export_px else self.diameter

    @property
    def export_mm_per_px(self):
        d = self.export_diameter
        return ARM_MM / d if d >= MIN_DIAM_PX else 0.0

    def export_chord(self):
        """The scale line as exported: the chord at the tip, resized about its own midpoint
        to the calibrated diameter. Resized rather than redrawn so that what is on screen
        is exactly what is written out -- the one rule this overlay has."""
        c = np.asarray(self.end)
        half = 0.5 * self.export_diameter * self.n
        return (tuple(float(z) for z in c - half), tuple(float(z) for z in c + half))

    def to_scale_line(self):
        """The frame's ONE robot-arm reference as a scale_objects.ScaleLine (class 3,
        ARM_MM across), or None if this frame is not exported.

        This is what a depth-estimation loss consumes: the same record shape the ruler and
        the catheter tip write, so nothing downstream needs to know the arm was measured
        differently. It is the scale line you can see on the frame -- what is exported is
        what is drawn, and there is no second answer hiding behind the picture."""
        if not (self.trusted and self.measured):
            return None
        from gui import scale_objects
        p0, p1 = self.export_chord()
        return scale_objects.ScaleLine(3, p0, p1, ts=(0.0, 1.0), mm=ARM_MM,
                                       source='measured', conf=1.0)

    # --- persistence ------------------------------------------------------
    def to_dict(self):
        return {'entry': [list(self.pts[0]), list(self.pts[1])],
                'tip': [list(self.pts[2]), list(self.pts[3])],
                'start': list(self.start), 'end': list(self.end),
                'diameter_px': round(self.diameter, 4),
                'mm': ARM_MM, 'mm_per_px': round(self.mm_per_px, 8),
                'chord': [list(p) for p in self.chord] if self.measured else None,
                'span_px': round(self.length_px, 3),
                'source': self.source, 'manual': self.manual,
                'trusted': self.trusted, 'trusted_by': self.trusted_by}

    @staticmethod
    def from_dict(d):
        """None for a record without the four points -- an annotation left by the old
        edge-fitting tool. Re-measure the clip rather than guess at what it meant."""
        if not d or 'entry' not in d or 'tip' not in d:
            return None
        return ArmMeasure(pair_up(list(d['entry']) + list(d['tip'])), d.get('trusted'),
                          d.get('trusted_by', 'auto'), d.get('source', 'auto'),
                          d.get('manual', False))


def snap_width(ann, comp):
    """The same annotation with its tip pair put back onto the MASK's own chord, at the
    station along the arm the tracker left it at. Unchanged if there is nothing there.

    This is what makes a tracked clip measure anything at all, and it is the one thing the
    tracker must not be asked to do. A tracker follows a patch of image: it can say where
    the end of the shaft has moved to, but a cylinder's two edges are smooth and
    featureless ALONG their length, so nothing in the picture tells it how far apart they
    now are. Handed a pair of edge points it carries them along at the width it was given
    and reports the same diameter on every frame -- through a zoom that is visibly making
    the arm wider.

    So the two are given the jobs they are actually good at: the tracker chooses WHERE
    along the arm to measure, which is a question about position and needs no subpixel
    accuracy across the shaft, and the segmentation says how wide the arm is there, which
    is a question about the silhouette and is exactly what a mask is. The entry pair is
    left alone -- snap it to the border run too if the axis ever needs locking to the mask
    as well, but that would overwrite an entry you had placed by hand."""
    if comp is None:
        return ann
    c = chord_at(comp, ann.end, ann.n)
    if c is None or c[2] < MIN_DIAM_PX:
        return ann              # the tip is off the mask: keep what the tracker gave
    pts = pair_up(list(ann.pts[:2]) + [c[0], c[1]])
    ann.pts[2], ann.pts[3] = pts[2], pts[3]
    return ann


def station_frac(ann, comp):
    """Where this annotation's tip sits along the shaft, as a fraction in [0,1]: 0 at the
    border, 1 at the end of the straight shaft. None if the shaft cannot be walked here.

    A fraction rather than a distance, because a fraction is what survives the arm moving.
    Pixels do not: when the arm comes closer the whole silhouette scales, so the same
    physical cross-section sits further from the border in pixels than it did -- and
    holding the station at a fixed pixel depth would slide it down the arm exactly when the
    scale is changing. The fraction is the same physical place whatever the depth."""
    got = _shaft(comp, ann.start, ann.u) if comp is not None else None
    if got is None or len(got[0]) < 2:
        return None
    u, s = ann.u, np.asarray(ann.start, np.float64)
    last = got[0][-1]
    end = 0.5 * (np.asarray(last[0]) + np.asarray(last[1]))
    full = float((end - s) @ u)
    if full < 1.0:
        return None
    return float(np.clip(float((np.asarray(ann.end) - s) @ u) / full, 0.0, 1.0))


def at_frac(ann, comp, frac):
    """The same annotation with its tip pair moved to `frac` of the way along THIS frame's
    shaft, on the mask's own chord there. Unchanged if the shaft cannot be walked.

    The other half of station_frac, and the alternative to letting a tracker choose where
    to measure: the station is re-derived from the mask every frame at the same fraction
    along the arm, so it needs no texture to hold onto and cannot drift."""
    got = _shaft(comp, ann.start, ann.u) if comp is not None else None
    if got is None:
        return ann
    # the station is measured along the axis, not counted in chords: the walk samples every
    # SHAFT_STEP px, and snapping the station to that grid would quantise the reading and
    # let it hop a whole step whenever the shaft end moved slightly
    last = got[0][-1]
    s, u = np.asarray(ann.start, np.float64), ann.u
    full = float((0.5 * (np.asarray(last[0]) + np.asarray(last[1])) - s) @ u)
    if full < 1.0:
        return ann
    c = chord_at(comp, s + float(np.clip(frac, 0.0, 1.0)) * full * u, ann.n)
    if c is None or c[2] < MIN_DIAM_PX:
        return ann
    pts = pair_up(list(ann.pts[:2]) + [c[0], c[1]])
    ann.pts[2], ann.pts[3] = pts[2], pts[3]
    return ann


def entry_drift(prev, cur):
    """How far the entry pair moved between two frames: the larger of the two, in px.

    The one number that says the tracker has let go. The entry pair is pinned to the image
    border and slides along it slowly, so unlike the tip pair it has no legitimate reason
    to move far in one frame -- which makes it the honest early warning for the whole
    annotation, since everything else is measured from where it sits."""
    return max(float(np.hypot(cur.pts[i][0] - prev.pts[i][0],
                              cur.pts[i][1] - prev.pts[i][1])) for i in (0, 1))


def measure_frame(mask, seeds=()):
    """Guess this frame's four points off the segmentation: ArmMeasure, or None if there
    is no arm mask here, none of it reaches a border, or no shaft can be walked out."""
    comp = pick_component(arm_pixels(mask), seeds)
    if comp is None:
        return None
    pts = guess(comp)
    return None if pts is None else ArmMeasure(pts)


# --- reconciling the clip -------------------------------------------------

def _local_median(d, window):
    """Rolling median of d over `window` neighbouring entries (index space)."""
    half = max(1, int(window) // 2)
    return np.array([np.median(d[max(0, i - half):i + half + 1]) for i in range(len(d))])


def _anchor(manual):
    """Where the continuity chain starts: the first frame placed by hand, else the first
    measured one. A frame you placed IS the statement of what the arm looks like, which is
    exactly what a chain needs at its head -- and it is why the chain can run outward in
    both directions from it, instead of having to assume the clip was annotated forwards."""
    hands = [i for i, m in enumerate(manual) if m]
    return hands[0] if hands else 0


def _chain(d, frames, keep, manual, start, tol=STEP_TOL, gap_max=STEP_GAP_MAX):
    """Drop every frame whose reading has jumped away from the last one accepted before it.

    Walked outward from `start` in both directions, and the reference is the last ACCEPTED
    frame rather than the immediately previous one: a single bad frame is then dropped on
    its own, while a jump the tracker never comes back from drops everything past it --
    which is the point. If it does come back, the frames that agree with the anchor again
    are accepted again, because the reference never moved to the wrong value.

    Returns the positions dropped, in frame order."""
    dropped = []
    for order in (range(start, len(frames)), range(start, -1, -1)):
        ref, prev_t = d[start], frames[start]
        for i in order:
            if i == start:
                continue
            # the tolerance grows over frames nobody MEASURED, where the arm had time to
            # move unwatched -- never over frames just rejected, or a sustained jump would
            # widen its own way back in after three frames of being caught
            gap = min(abs(frames[i] - prev_t), gap_max)
            prev_t = frames[i]
            if manual[i]:               # a frame you placed by hand re-anchors the chain
                ref = d[i]
                continue
            if ref > 0 and abs(d[i] - ref) > tol * gap * ref:
                keep[i] = False
                dropped.append(i)
            elif keep[i]:
                ref = d[i]
    return sorted(dropped)


def _window(n, want=SMOOTH_WIN, frac=WINDOW_FRAC):
    """How many neighbouring measurements the local median may span, on an annotation of
    n of them. Never more than 1/frac of it: a window that covers the whole clip is the
    global median wearing a rolling median's clothes."""
    return max(3, min(int(want), max(1, int(n) // int(frac))))


def clip_scale(arm_by_frame, window=SMOOTH_WIN, k=OUTLIER_MAD, calib=1.0):
    """Decide which frames to export, and what each of them exports.

    Writes two things onto each ArmMeasure and returns a summary:

      ann.export_px       what this frame exports: its OWN diameter, times calib. None
                          where the frame has none.
      ann.trusted         whether this frame is exported at all. A frame has to pass two
                          tests: it must agree with its NEIGHBOURS (the outlier test, for
                          noise) and it must not have JUMPED away from the last frame
                          anybody believed (_chain, for the points coming off the shaft and
                          staying off). Frames whose verdict you gave yourself (trusted_by
                          == 'user') are left exactly as you left them either way.

    The neighbours decide WHETHER, never WHAT. Each frame's scale is paired with that
    frame's depth map, so a neighbourhood median would be handing a depth estimator the
    scale of a different moment -- and on a camera that is moving it would smooth away
    exactly the change the reference exists to record. Measured on a real 9-frame clip:
    the hand-drawn ruler showed the scene coming 9.9% closer over those frames, the arm's
    own readings showed 5.0% of it, and the smoothed export showed 0.8%.

    calib is a single multiplier for the whole clip -- the one knob for a mask that is
    systematically fat or thin, which no per-frame geometry can see. 1.0 leaves the
    measurement alone."""
    frames = sorted(t for t, a in arm_by_frame.items() if a is not None and a.measured)
    summary = {'n_measured': len(frames), 'n_exported': 0, 'n_rejected': 0,
               'median_px': 0.0, 'mm_per_px': 0.0, 'scatter_px': 0.0, 'scatter_pct': 0.0,
               'raw_scatter_pct': 0.0, 'calib': float(calib), 'window': 0,
               'n_broken': 0, 'break_at': None, 'anchor': None}
    for a in arm_by_frame.values():
        if a is not None:
            a.export_px = None
    if not frames:
        return summary
    d = np.array([arm_by_frame[t].diameter for t in frames], float)
    win = _window(len(frames), window)
    med = _local_median(d, win)
    resid = np.abs(d - med)
    mad = float(np.median(resid))
    tol = np.maximum(max(k * mad, OUTLIER_FLOOR_PX), OUTLIER_FLOOR_FRAC * med)
    keep = resid <= tol
    # ... and then the continuity chain, which is a different question: not "is this frame
    # noisy" but "is this still the same measurement". A frame has to pass both.
    manual = [arm_by_frame[t].manual for t in frames]
    start = _anchor(manual)
    broke = _chain(d, frames, keep, manual, start)
    summary.update(n_broken=len(broke), anchor=frames[start], window=win,
                   break_at=frames[broke[0]] if broke else None)
    for t, own, ok in zip(frames, d, keep):
        ann = arm_by_frame[t]
        ann.export_px = float(own) * float(calib)
        if ann.trusted_by != 'user':          # your own answer is never overruled
            ann.trusted = bool(ok)
    used = d[keep] * float(calib)
    summary.update(n_exported=int(sum(1 for t in frames if arm_by_frame[t].trusted)),
                   n_rejected=int((~keep).sum()),
                   median_px=float(np.median(used)) if used.size else 0.0,
                   scatter_px=float(used.std()) if used.size else 0.0,
                   raw_scatter_pct=float(100.0 * d.std() / d.mean()) if d.mean() else 0.0)
    if summary['median_px'] >= MIN_DIAM_PX:
        summary['mm_per_px'] = ARM_MM / summary['median_px']
        summary['scatter_pct'] = 100.0 * summary['scatter_px'] / summary['median_px']
    return summary


def ruler_check(arm_by_frame, scale_by_frame, ruler_cls=1):
    """Compare the arm's mm/px against a hand-drawn reference on the frames carrying both.

    REPORTED, never applied on its own, and the reason is physical rather than technical:
    mm/px is a property of a point in the picture, not of the picture. A ruler lying on
    tissue further from the camera than the arm genuinely has a different mm/px, and
    "correcting" the arm to match it would be scaling away a real depth difference. Where
    the two ARE at comparable depth this is the only independent check available -- the
    ruler is drawn by hand on something visible and carries none of the mask's bias.

    Returns None if no frame has both, else dict(n, ratio, arm_mm_per_px, ref_mm_per_px)
    where ratio is what the arm's mm/px must be multiplied by to agree with the reference."""
    pairs = []
    for ti, objs in (scale_by_frame or {}).items():
        line = (objs or {}).get(ruler_cls)
        ann = (arm_by_frame or {}).get(ti)
        if line is None or ann is None or not ann.measured:
            continue
        d = ann.export_diameter
        ref = getattr(line, 'mm_per_px', None)
        if callable(ref):
            ref = ref()
        if ref and d >= MIN_DIAM_PX:
            pairs.append((ARM_MM / d, float(ref)))
    if not pairs:
        return None
    arm_v = np.array([p[0] for p in pairs])
    ref_v = np.array([p[1] for p in pairs])
    return {'n': len(pairs), 'ratio': float(np.median(ref_v / arm_v)),
            'arm_mm_per_px': float(np.median(arm_v)),
            'ref_mm_per_px': float(np.median(ref_v))}


def diagnose(mask, seeds=()):
    """Why measure_frame gave up on this frame, in a sentence the annotator can act on.
    The steps fail for genuinely different reasons, and telling someone to go and segment
    the arm when they already have is how an afternoon gets lost."""
    px = arm_pixels(mask)
    if px is None:
        return (f'there is no "{ARM_NAME}" (object {ARM_OBJECT_ID}) mask on this frame at '
                'all. Segment the arm in Mask annotation and propagate it first -- this '
                'tool measures that mask, it does not find the arm.')
    if pick_component(px, seeds) is None:
        return (f'the "{ARM_NAME}" mask on this frame does not touch any image border. '
                'The arm enters the picture, so its mask has to run off the edge -- paint '
                'it all the way out. (If the picture has black bars down the sides, the '
                'arm stops at the picture and never reaches the frame edge: re-extract '
                'the video so the bars are cut.)')
    return ('the arm mask reaches a border, but no shaft could be walked out from it -- '
            'it is too small, or broken up along its length. Check that it is the arm you '
            'segmented and not a stray blob of instrument.')


# --- drawing ---------------------------------------------------------------

def draw(img, ann, comp=None, editing=False):
    """Draw one frame's arm annotation onto img in place: the mask outline, the two sides
    of the shaft, the centerline between them, and -- when the diameter is on -- the scale
    line at the tip, which is the thing that gets exported. editing adds the four grips."""
    if ann is None or not isinstance(img, np.ndarray) or img.dtype != np.uint8 \
            or img.ndim != 3:
        return
    live = ann.trusted and ann.measured
    color = ARM_COLOR if live else OFF_COLOR
    if comp is not None:
        cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, color, 1, cv2.LINE_AA)
    e0, e1, t0, t1 = ann.pts
    for p, q in ((e0, t0), (e1, t1), (ann.start, ann.end)):
        cv2.line(img, _pt(p), _pt(q), color, THICKNESS, cv2.LINE_AA)
    if live:
        p0, p1 = ann.export_chord()
        cv2.line(img, _pt(p0), _pt(p1), CHORD_COLOR, THICKNESS + 1, cv2.LINE_AA)
        v = ann.u * 4.0             # end ticks, along the axis, so it reads as a caliper
        for p in (p0, p1):
            q = np.asarray(p, np.float64)
            cv2.line(img, _pt(q - v), _pt(q + v), CHORD_COLOR, 1, cv2.LINE_AA)
    if editing:
        for _, p in ann.handles():
            px, py = _pt(p)
            r = HANDLE_R - 1
            cv2.rectangle(img, (px - r, py - r), (px + r, py + r), color, -1, cv2.LINE_AA)
            cv2.rectangle(img, (px - r, py - r), (px + r, py + r), (255, 255, 255), 1,
                          cv2.LINE_AA)
    label = (f'{ann.export_diameter:.1f} px = {ARM_MM:g} mm' if live else
             ('not exported' if ann.measured else 'no diameter here'))
    cv2.putText(img, label, (_pt(ann.end)[0] + 8, _pt(ann.end)[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color if live else HOLD_COLOR, 1,
                cv2.LINE_AA)


if __name__ == '__main__':
    def scene(w=320, h=240, y0=120, dy=0.0, width=20, jaws=True, shaft=200):
        """A frame's object-id mask: a tube entering the left border at y0, drifting by dy
        over its length, ending in a wrist that flares into two open jaws."""
        m = np.zeros((h, w), np.uint8)
        for x in range(shaft):
            yc = y0 + dy * x
            m[int(yc - width / 2):int(yc + width / 2) + 1, x] = ARM_OBJECT_ID
        if jaws:
            for x in range(shaft, min(w, shaft + 50)):
                s = (x - shaft) * 0.5
                for yc in (y0 + dy * x - s, y0 + dy * x + s):
                    m[max(0, int(yc - 4)):int(yc + 4) + 1, x] = ARM_OBJECT_ID
        return m

    # a straight tube: the entry pair is the border run's two ends, the tip pair sits on
    # the shaft and NOT out in the jaws, and the diameter is the tube's own width
    ann = measure_frame(scene())
    assert ann is not None and ann.measured
    assert abs(ann.pts[0][0]) < 1e-6 and abs(ann.pts[1][0]) < 1e-6, 'entry is on the border'
    assert abs(abs(ann.pts[0][1] - ann.pts[1][1]) - 21) <= 1, ann.pts
    assert abs(ann.diameter - 21) < 2, ann.diameter
    assert ann.end[0] < 200, f'tip pair walked into the jaws: {ann.end}'
    assert abs(ann.mm_per_px - ARM_MM / ann.diameter) < 1e-9

    # THE CORNERS. A segmentation does not draw the arm meeting the border at an angle,
    # it draws a curve into it, so the ends of the mask's border run sit well inside the
    # real corner -- and the deeper the rounding, the further in. The sides are straight
    # though, so fitting them and following them out of the picture puts the corner back.
    def curved(w=400, h=400, y0=200, half=45, shaft=340, r=25, dy=0.0):
        # an arm entering the left border whose two corners are CURVES, not angles
        m = np.zeros((h, w), np.uint8)
        for x in range(shaft):
            off = r - np.sqrt(max(0.0, r * r - (r - x) ** 2)) if x < r else 0.0
            yc = y0 + dy * x
            m[int(yc - half + off):int(yc + half - off) + 1, x] = ARM_OBJECT_ID
        return m

    for r in (15, 40):
        for dy in (0.0, -0.4):
            m = curved(r=r, dy=dy)
            raw = sorted(q[1] for q in border_pair(pick_component(arm_pixels(m))))
            fit = sorted(q[1] for q in measure_frame(m).pts[:2])
            want = (155.0, 245.0)          # where the un-rounded silhouette meets x = 0
            assert max(abs(q - t) for q, t in zip(raw, want)) >= r - 1, 'scene not rounded'
            assert max(abs(q - t) for q, t in zip(fit, want)) < 2.0, (r, dy, fit)

    # ... and it does not need to be told WHICH border: an arm crossing an image corner
    # has one side leaving through one border and the other through the other, which is
    # the case a single border run gets most wrong (it puts a corner at the image corner,
    # 50 px from the truth in this scene).
    ang = np.deg2rad(205.0)
    u_c = np.array([np.cos(ang), np.sin(ang)])
    n_c = np.array([-u_c[1], u_c[0]])
    base = np.array([399.0, 399.0])
    yy, xx = np.mgrid[0:400, 0:400]
    d_along = (xx - base[0]) * u_c[0] + (yy - base[1]) * u_c[1]
    d_across = np.abs((xx - base[0]) * n_c[0] + (yy - base[1]) * n_c[1])
    round_off = np.where(d_along < 30, 30 - np.sqrt(np.maximum(
        0.0, 900 - (30 - np.clip(d_along, 0, 30)) ** 2)), 0.0)
    m = (((d_along >= 0) & (d_along <= 360) & (d_across <= 45 - round_off))
         * ARM_OBJECT_ID).astype(np.uint8)
    truth = [_exit(base + 200 * u_c + side * 45 * n_c, -u_c, m.shape) for side in (1, -1)]
    got = list(measure_frame(m).pts[:2])
    worst = max(min(float(np.hypot(q[0] - t[0], q[1] - t[1])) for t in truth) for q in got)
    assert worst < 2.0, (got, truth)
    assert sorted(edge_of(q, m.shape) for q in got) == [1, 3], 'one corner per border'

    # the two sides must not cross, whichever border the arm comes in from. e0 pairs with
    # t0 and e1 with t1, so both have to sit on the SAME side of the centerline -- an arm
    # entering from the right used to come out with its side lines drawn in an X.
    for k in range(4):
        m = np.rot90(scene(w=400, h=400, y0=200, shaft=300), k)
        a = measure_frame(np.ascontiguousarray(m))
        assert a is not None, f'no arm after {k} quarter-turns'
        n, start, end = a.n, np.asarray(a.start), np.asarray(a.end)
        side_e = float((np.asarray(a.pts[0]) - start) @ n)
        side_t = float((np.asarray(a.pts[2]) - end) @ n)
        assert side_e * side_t > 0, f'sides crossed entering from border {k}: {a.pts}'
        assert abs(a.diameter - 21) < 2, (k, a.diameter)

    # THE property: a tilted arm is measured SQUARE to its own axis. The mask is drawn as
    # columns 21 px tall, so its true width across is 21*cos(atan(dy)) -- read down the
    # column instead and it comes out 12% too wide, which is a millimetre of scale error.
    tilt = measure_frame(scene(dy=0.5, jaws=False, shaft=180))
    want = 21 * np.cos(np.arctan(0.5))
    assert tilt is not None and abs(tilt.diameter - want) < 1.5, (tilt.diameter, want)

    # ... and that is exactly why the tip points are projected: slide one along the arm and
    # the reading does not move
    d0, u = tilt.diameter, tilt.u
    tilt.move('t0', tilt.pts[2][0] + 15 * u[0], tilt.pts[2][1] + 15 * u[1])
    assert abs(tilt.diameter - d0) < 0.5, (tilt.diameter, d0)
    assert tilt.manual and tilt.source == 'manual'

    # no arm, and an arm that never reaches a border, are different failures
    assert measure_frame(np.zeros((60, 80), np.uint8)) is None
    floating = np.zeros((60, 80), np.uint8)
    floating[20:40, 20:60] = ARM_OBJECT_ID
    assert measure_frame(floating) is None
    assert 'border' in diagnose(floating)

    # round-trip, and an old-format record loads as nothing rather than as a guess
    back = ArmMeasure.from_dict(ann.to_dict())
    assert back.pts == ann.pts and back.trusted == ann.trusted
    crossed = ann.to_dict()                     # a record saved with its sides crossed
    crossed['tip'] = crossed['tip'][::-1]       # comes back straight
    assert ArmMeasure.from_dict(crossed).pts == ann.pts
    assert ArmMeasure.from_dict({'start': [0, 0], 'end': [10, 0]}) is None

    # THE entry constraint: one dimension. Dragging an entry point sideways off its edge
    # only slides it along the edge; dragging a tip point moves it wherever you put it.
    shape = (240, 320)
    e = measure_frame(scene())
    e.move('e0', 40.0, 150.0, shape)
    assert e.pts[0] == (0.0, 150.0), e.pts[0]         # x pinned to the left edge it is on
    e.move('e0', -10.0, 999.0, shape)
    assert e.pts[0] == (0.0, 239.0), e.pts[0]         # ... and clamped into the picture
    e.move('t0', 40.0, 150.0, shape)
    assert e.pts[2] == (40.0, 150.0), e.pts[2]        # a tip point is free
    # the same holds for a tracker: the entry pair goes back on the edge it entered across,
    # however far the tracked position crept off it
    drifted = e.moved_to([(7.0, 120.0), (4.0, 140.0), (180.0, 110.0), (180.0, 130.0)],
                         shape=shape)
    assert drifted.pts[0] == (0.0, 120.0) and drifted.pts[1] == (0.0, 140.0), drifted.pts
    # an arm entering the bottom is held to the bottom, not to whatever is nearest
    bot = ArmMeasure([(100.0, 239.0), (120.0, 239.0), (110.0, 80.0), (130.0, 80.0)])
    assert bot.moved_to([(103.0, 232.0), (123.0, 231.0), (110.0, 70.0), (130.0, 70.0)],
                        shape=shape).pts[0] == (103.0, 239.0)

    # the arm comes closer, so it gets WIDER -- and a tracker carrying the tip pair along
    # cannot see that, because a cylinder's edges say nothing along their length. The
    # mask does, and snap_width is what reads it: same tracked station, this frame's width
    wide = np.zeros((240, 320), np.uint8)
    for x in range(200):
        wide[105:136, x] = ARM_OBJECT_ID                       # 31 px across, was 21
    carried = measure_frame(scene()).moved_to([(0.0, 110.0), (0.0, 130.0),
                                               (180.0, 110.0), (180.0, 130.0)])
    assert abs(carried.diameter - 20.0) < 0.01, carried.diameter
    snapped = snap_width(carried, pick_component(arm_pixels(wide)))
    assert abs(snapped.diameter - 31) < 2, snapped.diameter     # ... and now it is 31
    assert snapped.mm_per_px < 0.01 + ARM_MM / 31 * 1.05
    # a tip that has drifted off the mask keeps what the tracker gave, rather than
    # inventing a width from whatever else is under it
    off = measure_frame(scene()).moved_to([(0.0, 110.0), (0.0, 130.0),
                                           (300.0, 20.0), (300.0, 40.0)])
    assert snap_width(off, pick_component(arm_pixels(wide))).pts == off.pts

    # THE TWO TRACKING MODES, on the same arm. 'points' lets the tracker choose where
    # along the arm to measure; 'shaft' pins the station at a fraction of the way along
    # and re-derives it from the mask, so it needs no texture and cannot slide.
    tap = np.zeros((240, 320), np.uint8)                # a tapering arm: 30 px at the
    for x in range(220):                                # border, 16 px at the far end
        half = (30 - 14.0 * x / 220) / 2
        tap[int(120 - half):int(120 + half) + 1, x] = ARM_OBJECT_ID
    tcomp = pick_component(arm_pixels(tap))
    base = measure_frame(tap)
    f = station_frac(base, tcomp)
    assert f is not None and 0.9 < f <= 1.0, f          # the guess ends AT the shaft end
    # halfway along reads a width between the two ends, and reading it twice is stable
    half_way = at_frac(measure_frame(tap), tcomp, 0.5)
    assert 16 < half_way.diameter < 30, half_way.diameter
    assert abs(at_frac(measure_frame(tap), tcomp, 0.5).diameter - half_way.diameter) < 1e-9
    # the fraction round-trips: put the station at 0.5, ask where it is, get 0.5 back
    assert abs(station_frac(half_way, tcomp) - 0.5) < 0.06, station_frac(half_way, tcomp)
    # and it is monotone -- further along a tapering arm is narrower
    widths = [at_frac(measure_frame(tap), tcomp, q).diameter for q in (0.2, 0.5, 0.8)]
    assert widths[0] > widths[1] > widths[2], widths
    # a station pinned by fraction does not care that the tracker's tip has slid: the
    # same fraction on the same mask gives the same reading whatever the tip was doing
    slid = measure_frame(tap)
    slid.pts[2] = (slid.pts[2][0] - 60, slid.pts[2][1])
    slid.pts[3] = (slid.pts[3][0] - 60, slid.pts[3][1])
    assert abs(at_frac(slid, tcomp, 0.5).diameter - half_way.diameter) < 1.0

    # the entry pair's speed limit: it slides a few px a frame, so a big step is the
    # tracker having let go. Measured on the pair only -- the tip is free to move.
    base = measure_frame(scene())
    e0, e1 = base.pts[0], base.pts[1]

    def slid(de0, de1, tip=((250.0, 60.0), (250.0, 200.0))):
        return base.moved_to([(e0[0], e0[1] + de0), (e1[0], e1[1] + de1)] + list(tip),
                             shape=shape)

    crept = slid(3.0, -1.0)             # ... and a big tip move is not a let-go
    assert abs(entry_drift(base, crept) - 3.0) < 1e-6, entry_drift(base, crept)
    assert entry_drift(base, crept) < ENTRY_DRIFT_PX
    jumped = slid(40.0, 0.0, ((180.0, 110.0), (180.0, 130.0)))
    assert entry_drift(base, jumped) > ENTRY_DRIFT_PX, entry_drift(base, jumped)

    # the clip: one frame measuring something else is dropped, the rest are exported
    clip = {t: measure_frame(scene()) for t in range(20)}
    clip[7].pts[2] = (clip[7].pts[2][0], clip[7].pts[2][1] - 9)     # a fat mask on frame 7
    s = clip_scale(clip)
    assert s['n_measured'] == 20 and s['n_rejected'] == 1 and not clip[7].trusted
    assert abs(s['mm_per_px'] - ARM_MM / 21) < 0.02, s
    assert abs(clip_scale(clip, calib=1.10)['mm_per_px'] - ARM_MM / 23.1) < 0.02
    # a verdict you gave yourself survives the reconciliation
    clip[3].toggle(False)
    clip_scale(clip)
    assert not clip[3].trusted and clip[3].trusted_by == 'user'

    # WHAT a frame exports is its own reading, never its neighbours'. The neighbours only
    # decide WHETHER. This is the whole point of the reference: one scale per frame, to be
    # paired with that frame's depth map, so a camera moving in must come out in the log.
    zoom = {}
    for t in range(24):
        a = measure_frame(scene())
        x, y = a.pts[2]
        a.pts[2] = (x, y - 0.02 * t * a.diameter)      # the arm coming steadily closer
        zoom[t] = a
    raw = [zoom[t].diameter for t in range(24)]
    clip_scale(zoom)
    out = [zoom[t].export_diameter for t in range(24)]
    assert out == raw, 'the export must be the frame own reading, unsmoothed'
    assert len(set(round(v, 3) for v in out)) == 24, 'every frame its own number'
    trend = 100 * (out[-1] - out[0]) / out[0]
    assert trend > 30, f'a real zoom has to survive the export, got {trend:.1f}%'
    # ... and CALIB is the one thing that does scale it, whole-clip
    clip_scale(zoom, calib=1.10)
    assert all(abs(zoom[t].export_diameter - 1.10 * raw[t]) < 1e-9 for t in range(24))

    # the outlier window never spans the annotation: with 9 frames a 15-frame "rolling"
    # median is the global one, which is what made a short clip export a single constant
    assert _window(9, 15) == 3 and _window(30, 15) == 10 and _window(300, 15) == 15
    assert _window(2, 15) == 3, 'a floor, so the median always has something to say'
    assert clip_scale({t: measure_frame(scene()) for t in range(9)})['window'] == 3

    # THE continuity chain. A tracker that jumps at frame 12 and never comes back takes
    # its neighbours with it, so the rolling median is happy from 12 on -- only the STEP
    # away from frame 11 shows that the points left the arm.
    def run(n=24, jump_at=None, jump=1.5, until=None, hands=()):
        c = {}
        for t in range(n):
            a = measure_frame(scene())
            if jump_at is not None and t >= jump_at and (until is None or t < until):
                x, y = a.pts[2]
                a.pts[2] = (x, y - (jump - 1.0) * a.diameter)   # widen the tip chord
            a.manual = t in hands
            c[t] = a
        return c

    c = run(jump_at=12)
    s = clip_scale(c)
    assert all(c[t].trusted for t in range(12)), 'the good half must survive'
    assert not any(c[t].trusted for t in range(12, 24)), 'a jump drops what follows it'
    assert s['break_at'] == 12 and s['n_broken'] == 12 and s['anchor'] == 0

    # a jump the tracker DOES come back from costs only the frames it was away for: the
    # reference never moved onto the wrong value, so the recovered frames match it again
    c = run(jump_at=12, until=15)
    clip_scale(c)
    assert [t for t in range(24) if not c[t].trusted] == [12, 13, 14]

    # a real zoom is GRADUAL, and the chain must not fire on one: the arm gets 70% wider
    # over the clip, 3% at a time, and every frame is still believed
    c = {}
    for t in range(24):
        a = measure_frame(scene())
        x, y = a.pts[2]
        a.pts[2] = (x, y - 0.03 * t * a.diameter)
        c[t] = a
    clip_scale(c)
    assert all(c[t].trusted for t in range(24)), 'a gradual zoom is not a jump'

    # a frame you place by hand re-anchors the chain, and the side you signed for is the
    # side that is believed
    c = run(jump_at=12, hands=(12,))
    s = clip_scale(c)
    assert s['anchor'] == 12
    assert [t for t in range(24) if not c[t].trusted] == list(range(12))

    # the chain walks OUTWARD from the frame you placed by hand, not forwards from frame 0
    # -- so tracking backwards from your keyframe drops the wrong side of a jump, not the
    # right one
    c = run(jump_at=0, until=8, hands=(16,))
    s = clip_scale(c)
    assert s['anchor'] == 16
    assert [t for t in range(24) if not c[t].trusted] == list(range(8)),         [t for t in range(24) if not c[t].trusted]
    # ... and what is exported is exactly what is drawn
    line = clip[0].to_scale_line()
    assert abs(line.length_px - clip[0].export_diameter) < 1e-6
    assert abs(line.mm_per_px - clip[0].export_mm_per_px) < 1e-9

    vis = np.zeros((240, 320, 3), np.uint8)
    draw(vis, ann, comp=pick_component(arm_pixels(scene())), editing=True)
    assert vis.any(), 'the overlay drew nothing'
    assert ann.hit(*ann.pts[2]) == 't0' and ann.hit(0, 0, 3) is None

    print('robot_arm self-check OK')
