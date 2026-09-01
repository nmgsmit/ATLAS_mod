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

TRACK carries the four points with segmenter.PointTracker -- the same tracker the ruler
and the catheter tip use. Entry points that were sitting on the image border are put back
onto it after each step, because that is what an entry point is. A frame you correct by
hand is a keyframe: the tracker re-seeds there, and TRACK never overwrites it.

Trust is yours. Every frame carries a verdict, the automatic one only sets the initial
state, and only frames left ON are exported. clip_scale() then reports what the clip
agrees on and drops frames that disagree with their neighbours -- the arm is rigid, so its
apparent width can only change as fast as the camera-arm distance does, which is slowly.

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

BORDER_SNAP_PX = 6.0           # px; a tracked entry point this close to the image edge is
                               # put back onto it

# Reconciling the clip. The arm is rigid and 8 mm, so its apparent width changes only as
# fast as the camera-arm distance does: a frame that disagrees with its neighbours is
# telling you about its own mask, not about the arm.
SMOOTH_WIN = 15                # frames of neighbouring MEASUREMENTS the local median is
                               # taken over. Counted in measured frames, not frame numbers.
OUTLIER_MAD = 3.0              # a frame this many MADs off its local median is not exported
OUTLIER_FLOOR_PX = 0.5         # ... but never reject over less than this: half a pixel of
                               # disagreement on a rasterised boundary is not a fault.
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
    walk is aimed by the shaft itself."""
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
    return (tuple(e0), tuple(e1), tip[0], tip[1])


def snap_border(p, w, h, tol=BORDER_SNAP_PX):
    """A point within tol of the image edge, put back ON that edge.

    An entry point belongs on the border by definition, but a tracker following one has
    half its template outside the picture and creeps inward. A point dragged well clear of
    the edge is left alone -- that is somebody saying the arm does not reach it there."""
    x, y = float(p[0]), float(p[1])
    d = [x, w - 1 - x, y, h - 1 - y]
    i = int(np.argmin(d))
    if d[i] > tol:
        return (x, y)
    return [(0.0, y), (float(w - 1), y), (x, 0.0), (x, float(h - 1))][i]


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

    def move(self, name, x, y):
        """Drag one point. The frame becomes a keyframe: TRACK re-seeds here and never
        overwrites it."""
        self.pts[self.NAMES.index(name)] = (float(x), float(y))
        self.manual = True
        self.source = 'manual'
        return self

    def points(self):
        """The four points as a list, for the tracker."""
        return list(self.pts)

    def moved_to(self, pts, source='tracked'):
        """The same annotation with its four points somewhere else -- what TRACK produces.
        The verdict does NOT travel: it belongs to the frame it was given on, and the
        caller carries the target frame's own with adopt_verdict."""
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
        return ArmMeasure(list(d['entry']) + list(d['tip']), d.get('trusted'),
                          d.get('trusted_by', 'auto'), d.get('source', 'auto'),
                          d.get('manual', False))


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


def clip_scale(arm_by_frame, window=SMOOTH_WIN, k=OUTLIER_MAD, calib=1.0):
    """Reconcile every frame's diameter against the rest of the clip.

    Writes two things onto each ArmMeasure and returns a summary:

      ann.clip_diameter   the diameter to EXPORT -- the local median of the neighbouring
                          frames' readings, times calib. None where the frame has none.
      ann.trusted         whether this frame is exported at all. Set from the outlier test,
                          EXCEPT on frames whose verdict you gave yourself (trusted_by ==
                          'user'), which are left exactly as you left them.

    calib is a single multiplier for the whole clip -- the one knob for a mask that is
    systematically fat or thin, which no per-frame geometry can see. 1.0 leaves the
    measurement alone."""
    frames = sorted(t for t, a in arm_by_frame.items() if a is not None and a.measured)
    summary = {'n_measured': len(frames), 'n_exported': 0, 'n_rejected': 0,
               'median_px': 0.0, 'mm_per_px': 0.0, 'scatter_px': 0.0, 'scatter_pct': 0.0,
               'raw_scatter_pct': 0.0, 'calib': float(calib)}
    for a in arm_by_frame.values():
        if a is not None:
            a.clip_diameter = None
    if not frames:
        return summary
    d = np.array([arm_by_frame[t].diameter for t in frames], float)
    med = _local_median(d, window)
    resid = np.abs(d - med)
    mad = float(np.median(resid))
    tol = max(k * mad, OUTLIER_FLOOR_PX)
    keep = resid <= tol
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
    assert ArmMeasure.from_dict({'start': [0, 0], 'end': [10, 0]}) is None

    # a tracked entry point gets put back on the border it crept off; one dragged well
    # clear of the edge is left where it was put
    assert snap_border((2.0, 130.0), 320, 240) == (0.0, 130.0)
    assert snap_border((160.0, 130.0), 320, 240) == (160.0, 130.0)

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
    # ... and what is exported is exactly what is drawn
    line = clip[0].to_scale_line()
    assert abs(line.length_px - clip[0].export_diameter) < 1e-6
    assert abs(line.mm_per_px - clip[0].export_mm_per_px) < 1e-9

    vis = np.zeros((240, 320, 3), np.uint8)
    draw(vis, ann, comp=pick_component(arm_pixels(scene())), editing=True)
    assert vis.any(), 'the overlay drew nothing'
    assert ann.hit(*ann.pts[2]) == 't0' and ann.hit(0, 0, 3) is None

    print('robot_arm self-check OK')
