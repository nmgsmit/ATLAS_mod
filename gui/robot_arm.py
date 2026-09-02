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
The guess: the arm crosses the border, so on that border its mask is an interval and the
interval's two ends are the entry pair; from their midpoint the mask's perpendicular chord
is walked outward, and the shaft ends at the first station whose width leaves the shaft's
own (the wrist flaring, or the jaws opening).

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
and the catheter tip use -- and puts the entry pair back on its edges after each step. It
also holds the two sides to the SLOPE they were placed with (LOCK_SLOPE): the arm is a
straight rigid shaft, so the tracker may move each side across the arm and slide the tip
along it, but it may not pivot a side, and a pixel of tracker drift over a long arm is a
degree of pivot the arm never made. A frame you correct by hand is a keyframe: the tracker
re-seeds there -- taking the slope you just drew as the new one -- and TRACK never
overwrites it. The run STOPS the moment an entry point moves further in one frame than
the arm possibly could (ENTRY_DRIFT_PX): the entry pair is the far end of a long rigid
arm and slides a few pixels a frame at most, so a bigger step is the tracker having let
go, and every frame after it would be measured from the wrong place.

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
SMOOTH_WIN = 15                # frames of neighbouring MEASUREMENTS the local median is
                               # taken over. Counted in measured frames, not frame numbers.
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

# The one thing a tracker is not allowed to have an opinion about: the SLOPE of the arm's
# two edge lines. The arm is a straight rigid shaft, so each side is a line whose direction
# is set the moment you place the four points, and everything after that is the arm sliding
# and the camera closing in -- never the sides pivoting frame by frame. The four points sit
# on a smooth shaft with nothing to lock onto along its length, so each drifts a pixel or
# two a frame; a pixel of drift over a 200 px arm is a degree of rotation the arm never
# made, and the diameter is read SQUARE to the axis those points define, so that degree
# comes straight off the scale. Holding the slope costs nothing that is real: the height of
# each side and how far along the tip sits are still the tracker's to say, which is all a
# rigid arm can actually do on the screen.
LOCK_SLOPE = True              # False tracks all four points free, as before -- for a clip
                               # where the camera really does roll about the arm's axis

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

def _shaft_end(comp, mid, u):
    """(p0, p1): the ends of the last chord along +u from mid that is still the SHAFT.

    Chords are collected outward until the mask runs out, the shaft's own width is the
    median of the first few, and the shaft ends at the first station that departs from it
    by FLARE_TOL -- the wrist swelling, or the jaws opening into two half-width pieces
    that a single chord reads as one wide one."""
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
    c = chords[max(0, keep - 1 - BACK_OFF)]
    return c[0], c[1]


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
    tip = None
    for _ in range(2):
        found = _shaft_end(comp, mid, u)
        if found is None:
            return None
        tip = found
        u = _unit(0.5 * (np.asarray(tip[0]) + np.asarray(tip[1])) - mid)
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


def side_dirs(ann):
    """The direction of each of the arm's two edge lines, entry -> tip: (d0, d1).

    The slope the rest of the clip is held to. It belongs to the arm and to where the
    camera is looking from, not to any one frame's tracking, so it is captured where the
    four points were PLACED -- the frame you set up, or the last keyframe you corrected."""
    return (_unit(np.asarray(ann.pts[2]) - np.asarray(ann.pts[0])),
            _unit(np.asarray(ann.pts[3]) - np.asarray(ann.pts[1])))


def hold_slope(pts, dirs, edges, shape):
    """The four tracked points put back onto edge lines of the given directions.

    Each side keeps its slope and gives up nothing else. The line is re-fitted to that
    side's two tracked points -- a line of fixed direction through their midpoint, which is
    the least-squares fit to two points -- so the tracker still says how far the side has
    moved ACROSS the arm (the height) and how far along the shaft the tip sits, and only
    the pivot it was never entitled to is removed.

    The entry point then falls out of the geometry rather than being tracked at all: it is
    where that line crosses the border the arm entered across, which is by construction
    where that edge of the arm enters the picture. It is clamped into the frame, so an arm
    whose side line leaves through a corner still lands on its border.

    edges is the border index (edge_of) each entry point belongs to, one per side."""
    out = [(float(p[0]), float(p[1])) for p in pts]
    h, w = float(shape[0]), float(shape[1])
    for i, (d, edge) in enumerate(zip(dirs, edges)):
        d = _unit(d)
        e = np.asarray(pts[i], np.float64)
        t = np.asarray(pts[i + 2], np.float64)
        c = 0.5 * (e + t)                       # the line: this direction, this height
        axis, want = (0, (0.0, w - 1.0)[edge]) if edge < 2 else (1, (0.0, h - 1.0)[edge - 2])
        if abs(d[axis]) > 1e-6:
            hit = c + ((want - c[axis]) / d[axis]) * d
        else:
            hit = c + float((e - c) @ d) * d    # the side runs ALONG that border: keep the
                                                # tracked position, projected onto the line
        out[i] = on_edge(hit, edge, shape)
        q = c + float((t - c) @ d) * d
        out[i + 2] = (float(q[0]), float(q[1]))
    return out


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
        # What this frame EXPORTS: its own reading reconciled against the neighbouring
        # frames and scaled by the clip's calibration (see clip_scale). None until the clip
        # has been reconciled; self.diameter stays the raw reading, so the two can always
        # be compared.
        self.clip_diameter = None

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

    def moved_to(self, pts, source='tracked', shape=None, dirs=None):
        """The same annotation with its four points somewhere else -- what TRACK produces.

        Given shape (h, w), the entry pair is put back onto the edges THIS annotation's
        entry pair was on: the arm goes on entering the picture across the same border it
        crossed last frame, and a tracker is not entitled to an opinion about that.

        Given dirs -- the two edge-line directions from side_dirs, captured where the
        points were placed -- it is not entitled to an opinion about the SLOPE of the two
        sides either: each side is put back on a line of its own locked direction, keeping
        only the height and the tip's position along the shaft that the tracker found. See
        hold_slope, and LOCK_SLOPE for why.

        The verdict does NOT travel: it belongs to the frame it was given on, and the
        caller carries the target frame's own with adopt_verdict."""
        pts = list(pts)
        if shape is not None:
            edges = [edge_of(self.pts[i], shape) for i in (0, 1)]
            if dirs is not None:
                pts = hold_slope(pts, dirs, edges, shape)
            else:
                for i in (0, 1):
                    pts[i] = on_edge(pts[i], edges[i], shape)
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
        """The diameter that is drawn and written out (clip-reconciled where available)."""
        return float(self.clip_diameter) if self.clip_diameter else self.diameter

    @property
    def export_mm_per_px(self):
        d = self.export_diameter
        return ARM_MM / d if d >= MIN_DIAM_PX else 0.0

    def export_chord(self):
        """The scale line as exported: the chord at the tip, resized about its own midpoint
        to the clip-reconciled diameter. Resized rather than redrawn so that what is on
        screen is exactly what is written out -- the one rule this overlay has."""
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


def clip_scale(arm_by_frame, window=SMOOTH_WIN, k=OUTLIER_MAD, calib=1.0):
    """Reconcile every frame's diameter against the rest of the clip.

    Writes two things onto each ArmMeasure and returns a summary:

      ann.clip_diameter   the diameter to EXPORT -- the local median of the neighbouring
                          frames' readings, times calib. None where the frame has none.
      ann.trusted         whether this frame is exported at all. A frame has to pass two
                          tests: it must agree with its NEIGHBOURS (the outlier test, for
                          noise) and it must not have JUMPED away from the last frame
                          anybody believed (_chain, for the points coming off the shaft and
                          staying off). Frames whose verdict you gave yourself (trusted_by
                          == 'user') are left exactly as you left them either way.

    calib is a single multiplier for the whole clip -- the one knob for a mask that is
    systematically fat or thin, which no per-frame geometry can see. 1.0 leaves the
    measurement alone."""
    frames = sorted(t for t, a in arm_by_frame.items() if a is not None and a.measured)
    summary = {'n_measured': len(frames), 'n_exported': 0, 'n_rejected': 0,
               'median_px': 0.0, 'mm_per_px': 0.0, 'scatter_px': 0.0, 'scatter_pct': 0.0,
               'raw_scatter_pct': 0.0, 'calib': float(calib),
               'n_broken': 0, 'break_at': None, 'anchor': None}
    for a in arm_by_frame.values():
        if a is not None:
            a.clip_diameter = None
    if not frames:
        return summary
    d = np.array([arm_by_frame[t].diameter for t in frames], float)
    med = _local_median(d, window)
    resid = np.abs(d - med)
    mad = float(np.median(resid))
    tol = np.maximum(max(k * mad, OUTLIER_FLOOR_PX), OUTLIER_FLOOR_FRAC * med)
    keep = resid <= tol
    # ... and then the continuity chain, which is a different question: not "is this frame
    # noisy" but "is this still the same measurement". A frame has to pass both.
    manual = [arm_by_frame[t].manual for t in frames]
    start = _anchor(manual)
    broke = _chain(d, frames, keep, manual, start)
    summary.update(n_broken=len(broke), anchor=frames[start],
                   break_at=frames[broke[0]] if broke else None)
    for t, m, ok in zip(frames, med, keep):
        ann = arm_by_frame[t]
        ann.clip_diameter = float(m) * float(calib)
        if ann.trusted_by != 'user':          # your own answer is never overruled
            ann.trusted = bool(ok)
    used = med[keep] * float(calib)
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

    # THE slope lock: over a run the tracker may move each side ACROSS the arm and slide
    # the tip ALONG it, and may not pivot it. Every point below has drifted a couple of px
    # off in its own direction -- the two sides still come back on exactly the directions
    # they were placed with, and the entry pair on the border it entered across.
    lock = measure_frame(scene())
    dirs = side_dirs(lock)
    noisy = [(3.0, lock.pts[0][1] - 2.0), (2.0, lock.pts[1][1] + 1.0),
             (lock.pts[2][0] + 2.0, lock.pts[2][1] - 3.0),
             (lock.pts[3][0] - 2.0, lock.pts[3][1] + 3.0)]
    held = lock.moved_to(noisy, shape=shape, dirs=dirs)
    assert max(abs(float(a @ _perp(b))) for a, b in zip(side_dirs(held), dirs)) < 1e-9
    assert held.pts[0][0] == 0.0 and held.pts[1][0] == 0.0, held.pts
    free = lock.moved_to(noisy, shape=shape)
    assert max(abs(float(a @ _perp(b))) for a, b in zip(side_dirs(free), dirs)) > 1e-3, \
        'the free tracker keeps the pivot -- otherwise this test proves nothing'

    # ... and it is not a freeze. The camera closing in genuinely widens the arm, and that
    # comes through, because how far each side sits ACROSS the arm is still the tracker's
    # to say -- only the direction it runs in is not.
    wider = lock.moved_to([(0.0, lock.pts[0][1] - 2.0), (0.0, lock.pts[1][1] + 2.0),
                           (lock.pts[2][0], lock.pts[2][1] - 2.0),
                           (lock.pts[3][0], lock.pts[3][1] + 2.0)],
                          shape=shape, dirs=dirs)
    assert abs(wider.diameter - (lock.diameter + 4.0)) < 1e-6, wider.diameter
    # the tip still slides freely ALONG the shaft, and a slide along it is not a reading
    slid = lock.moved_to([lock.pts[0], lock.pts[1]]
                         + [(lock.pts[2 + i][0] + 20.0 * dirs[i][0],
                             lock.pts[2 + i][1] + 20.0 * dirs[i][1]) for i in (0, 1)],
                         shape=shape, dirs=dirs)
    assert abs(slid.diameter - lock.diameter) < 0.1, slid.diameter   # a tenth of a pixel:
    # the two sides are held to their OWN directions, and a guessed pair is a few
    # thousandths of a radian from parallel, so a shaft read 20 px longer picks up that
    # much of its taper. The point is that sliding the tip does not move the READING.
    assert slid.length_px > lock.length_px + 15.0

    # WHY: sixty frames of ordinary sub-pixel tracker noise on a smooth shaft. Free, the
    # sides have pivoted by degrees the arm never moved -- and the diameter is read square
    # to the axis they define, so that pivot is scale error. Locked, they have not moved.
    rng = np.random.default_rng(0)
    a_free, a_lock = measure_frame(scene()), measure_frame(scene())
    for _ in range(60):
        j = rng.normal(0.0, 0.7, (4, 2))
        a_free = a_free.moved_to([(q[0] + k[0], q[1] + k[1])
                                  for q, k in zip(a_free.pts, j)], shape=shape)
        a_lock = a_lock.moved_to([(q[0] + k[0], q[1] + k[1])
                                  for q, k in zip(a_lock.pts, j)], shape=shape, dirs=dirs)
    tilt_free = max(abs(float(a @ _perp(b))) for a, b in zip(side_dirs(a_free), dirs))
    tilt_lock = max(abs(float(a @ _perp(b))) for a, b in zip(side_dirs(a_lock), dirs))
    assert tilt_lock < 1e-9 and tilt_free > 0.01, (tilt_free, tilt_lock)

    # an arm entering the bottom: the locked side line is intersected with THAT border, not
    # with whichever one the tracked point drifted nearest to
    up = ArmMeasure([(100.0, 239.0), (120.0, 239.0), (110.0, 80.0), (130.0, 80.0)])
    ud = side_dirs(up)
    got = up.moved_to([(103.0, 232.0), (123.0, 231.0), (113.0, 70.0), (133.0, 70.0)],
                      shape=shape, dirs=ud)
    assert got.pts[0][1] == 239.0 and got.pts[1][1] == 239.0, got.pts
    assert max(abs(float(a @ _perp(b))) for a, b in zip(side_dirs(got), ud)) < 1e-9

    # the entry pair's speed limit: it slides a few px a frame, so a big step is the
    # tracker having let go. Measured on the pair only -- the tip is free to move.
    base = measure_frame(scene())
    crept = base.moved_to([(0.0, 113.0), (0.0, 133.0), (250.0, 60.0), (250.0, 200.0)],
                          shape=shape)
    assert entry_drift(base, crept) == 3.0, entry_drift(base, crept)
    assert entry_drift(base, crept) < ENTRY_DRIFT_PX      # a big tip move is not a let-go
    jumped = base.moved_to([(0.0, 150.0), (0.0, 133.0), (180.0, 110.0), (180.0, 130.0)],
                           shape=shape)
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
