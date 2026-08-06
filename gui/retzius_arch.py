"""Retzius arch overlay for the ATLAS GUI (ARCH button).

A parabolic arc marking the arch of the cave of Retzius. Each arc is stored as
the two base points plus a scalar tip height -- NOT as three free points:

    left, right : the sides of the arch, dragged onto the cave walls
    height      : signed distance of the tip from the base-chord midpoint,
                  measured along the chord's perpendicular bisector

The tip is derived as midpoint + height * normal, so it always sits on the
mid-line between the two sides (the urethra), and moving a side re-centres it
automatically. Because the arc is sampled in this chord-aligned frame and only
clipped at draw time, the parabola stays stable even when the tip lies outside
the image.

The arch is stored PER FRAME. TRACK carries it forward or backward through the
video in a pass of its own (it never touches the masks): probe points are sampled
along the visible part of the arc, followed with segmenter.PointTracker (CSRT/LK
+ occlusion guard), and each frame the parabola is re-fitted to the surviving
probes -- a weighted similarity transform (scale clamped per step so the arch
cannot 'breathe'), then a weighted-least-squares height refresh in which mid-arc
probes count most. Frames the user adjusted by hand are keyframes
(source='manual'); a TRACK run re-seeds on them as it passes.

The GUI wiring (clicks, dragging, the TRACK loop, persistence triggers) lives in
main_controller.py; arcs persist to <workspace>/arches.json.
"""
import json
from os import path

import cv2
import numpy as np

# --- CONFIG ---------------------------------------------------------------
ARC_COLOR = (0, 255, 255)      # RGB; arc line + side handles (cyan pops on tissue)
TIP_COLOR = (255, 0, 255)      # RGB; tip handle + mid-line guide (magenta)
HOLD_COLOR = (170, 170, 170)   # RGB; arc drawn gray on frames where tracking lost the tissue
SHAKY_COLOR = (255, 165, 0)    # RGB; arc drawn amber where it tracked but the probes disagreed
THICKNESS = 2                  # px; arc line width
HANDLE_R = 5                   # px; handle dot radius (image space)
GRAB_PX = 12                   # px; handle grab radius at zoom 1 (image space)
N_SAMPLES = 100                # points sampled along the arc polyline
# Shape exponent p in v(u) = height*(1-|u|^p). p=2 is the plain parabola; below it the
# tip pinches to a point, above it the crown flattens and the arms turn steep. Sides and
# tip are pinned for every p, so the knob only trades tip sharpness against arm splay.
DEFAULT_POWER = 2.0
POWER_RANGE = (1.5, 3.0)
ARCHES_FILE = 'arches.json'    # saved inside the workspace
PROBE_N = 7                    # tracked probe points along the visible arc
PROBE_MARGIN = 8               # px; probes stay this far inside the image border
SCALE_CLIP = (0.9, 1.1)        # per-frame-step bounds on the fitted scale change
MAX_HOLD_FRAMES = 30           # consecutive lost frames before the arch gives up (~1s @30fps;
                               # the masks keep propagating regardless)

# How much to trust each probe. Two independent signals, multiplied:
#   appearance -- does the patch still look like the target? (the tracker's NCC conf)
#   agreement  -- does it sit where the parabola the OTHER probes vote for says it should?
# Appearance alone cannot catch a probe that slid onto identical-looking tissue: it
# still matches, it is just in the wrong place. Agreement catches exactly that, so the
# fit is iteratively reweighted (IRLS) with a redescending Tukey loss -- a probe far
# off the consensus ends at weight 0 and cannot drag the arch. The scale is estimated
# per frame from the residuals themselves (MAD), so honest tissue deformation widens
# the tolerance while a lone jump still gets rejected.
IRLS_ITERS = 3                 # reweight-and-refit passes
TUKEY_C = 4.685                # Tukey biweight constant, in robust sigmas
SIGMA_FLOOR = 2.0              # px; residual scatter below this is tracker jitter, not signal
SIGMA_CEIL_FRAC = 0.08         # ceiling on that scale, as a fraction of the arch half-chord.
                               # The MAD scale is RELATIVE -- it spots one probe contradicting
                               # the rest, but if every probe is wrong there is no
                               # disagreement to spot and it would happily fit garbage. This
                               # ceiling is the absolute veto: a probe this far off the arc it
                               # was on is rejected no matter how many others are equally off.
MIN_EFF_PROBES = 2.0           # effective (weight-discounted) probes needed to fit at all
LOW_CONF = 0.45                # below this the arc is drawn amber: tracked, but shaky

# Occluder (robot instrument) masking -- the third trust factor, alongside appearance
# and agreement. There is nothing to see BEHIND an instrument; the mask does not
# recover the hidden tissue, it just says which probes to stop believing, so the
# remaining ones and the parabola carry the arch across. Annotate instruments as this
# object id and propagate them; the arch reads those masks from the same pass
# automatically, and is a no-op until then.
OCCLUDER_OBJECT_ID = 5         # 'Non-anatomical' in palette.custom_names
BLOCK_MARGIN = 10              # px; half-window around a probe that must be clear of the
                               # occluder (~ the tracker's appearance patch half-size)
SEED_MIN_VIS = 0.9             # only seed a probe on tissue at least this clear
# --------------------------------------------------------------------------


class Arch:

    def __init__(self, left, right, height, source='manual', conf=1.0, power=DEFAULT_POWER):
        self.left = (float(left[0]), float(left[1]))
        self.right = (float(right[0]), float(right[1]))
        self.height = float(height)
        self.power = float(np.clip(power, *POWER_RANGE))
        self.source = source    # 'manual' (user keyframe) | 'tracked' | 'hold' (lost)
        self.conf = float(conf)  # [0,1] mean probe trust behind this fit (1 if manual)

    def _frame(self):
        """Chord-aligned frame: midpoint, unit tangent t, unit normal n, half-length d."""
        left = np.asarray(self.left, np.float64)
        right = np.asarray(self.right, np.float64)
        mid = (left + right) / 2
        chord = right - left
        d = float(np.linalg.norm(chord)) / 2
        t = chord / (2 * d) if d > 1e-6 else np.array([1.0, 0.0])
        n = np.array([t[1], -t[0]])   # image y grows downward: +height bulges screen-up
        return mid, t, n, d

    @property
    def apex(self):
        mid, _, n, _ = self._frame()
        return tuple(mid + self.height * n)

    def set_apex(self, x, y):
        # only the height changes: project the cursor onto the mid-line, so the
        # tip stays centred between the sides even when dragged off the image
        mid, _, n, _ = self._frame()
        self.height = float(np.dot(np.asarray([x, y], np.float64) - mid, n))

    def shape(self, us):
        """Profile v(u)/height in [0,1]: 1 at the tip (u=0), 0 at both sides (u=+-1)."""
        return 1 - np.abs(np.asarray(us, np.float64)) ** self.power

    def at(self, us):
        """(n,2) points on the arc at chord parameters us in [-1,1];
        v(u) = height*(1-|u|^p) pins both sides exactly (u=-1 left, u=+1 right)."""
        mid, t, n, d = self._frame()
        u = np.asarray(us, np.float64).reshape(-1, 1)
        return mid + (u * d) * t + (self.shape(u) * self.height) * n

    def points(self, n_samples=N_SAMPLES):
        return self.at(np.linspace(-1.0, 1.0, n_samples))

    def handles(self):
        return (('left', self.left), ('right', self.right), ('apex', self.apex))


def hit_test(arches, x, y, radius=GRAB_PX):
    """Nearest (arch, handle_name) within radius of (x, y), else None."""
    best, best_d = None, float(radius)
    for arch in arches:
        for name, (px, py) in arch.handles():
            dist = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
            if dist <= best_d:
                best, best_d = (arch, name), dist
    return best


def move_handle(arch, name, x, y, ux, uy):
    """Drag a handle: the sides take the in-image (clamped) position, the tip projects
    the raw cursor onto the mid-line so it can travel past the image border."""
    if name == 'apex':
        arch.set_apex(ux, uy)
    else:
        setattr(arch, name, (float(x), float(y)))


def probe_visibility(blocked, pts, margin=BLOCK_MARGIN):
    """Per-probe visibility in [0,1]: the fraction of its patch that is clear of the
    occluder mask. 1 = clean tissue, 0 = fully behind an instrument. blocked is an
    HxW bool array (True = occluder); None means nothing is masked."""
    pts = list(pts)
    if blocked is None or not pts:
        return np.ones(len(pts), np.float64)
    h, w = blocked.shape
    vis = []
    for p in pts:
        x, y = int(round(p[0])), int(round(p[1]))
        x0, x1 = max(0, x - margin), min(w, x + margin + 1)
        y0, y1 = max(0, y - margin), min(h, y + margin + 1)
        patch = blocked[y0:y1, x0:x1]
        vis.append(0.0 if patch.size == 0 else 1.0 - float(np.mean(patch)))
    return np.asarray(vis, np.float64)


def sample_probes(arch, w, h, n=PROBE_N, margin=PROBE_MARGIN, blocked=None):
    """(u_list, point_list) of up to n probes on the VISIBLE part of the arc, for
    tracking. Probes on the off-image part (e.g. around an off-image tip) have no
    pixels to follow, so they are skipped, as are probes sitting on the occluder mask
    -- better not to seed on an instrument than to detect the drift afterwards. The
    survivors are spread evenly over whatever is left."""
    us = np.linspace(-1.0, 1.0, 41)
    pts = arch.at(us)
    ok = ((pts[:, 0] >= margin) & (pts[:, 0] <= w - 1 - margin) &
          (pts[:, 1] >= margin) & (pts[:, 1] <= h - 1 - margin))
    if blocked is not None:
        ok &= probe_visibility(blocked, pts) >= SEED_MIN_VIS
    us, pts = us[ok], pts[ok]
    if len(us) < 2:
        return [], []
    idx = np.unique(np.round(np.linspace(0, len(us) - 1, min(n, len(us)))).astype(int))
    return us[idx].tolist(), [(float(p[0]), float(p[1])) for p in pts[idx]]


def fit_similarity(src, dst, weights=None):
    """Weighted 2D similarity (scale s, rotation R, translation T) minimizing
    sum w_i |s R src_i + T - dst_i|^2 (no reflection). Exact for 2 points."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    w = np.ones(len(src)) if weights is None else np.asarray(weights, np.float64)
    w = w / w.sum()
    ms, md = (w[:, None] * src).sum(0), (w[:, None] * dst).sum(0)
    s0, d0 = src - ms, dst - md
    # complex form of 2D weighted Procrustes: z encodes rotation+scale jointly
    zs = s0[:, 0] + 1j * s0[:, 1]
    zd = d0[:, 0] + 1j * d0[:, 1]
    denom = float((w * np.abs(zs) ** 2).sum())
    if denom < 1e-12:                      # all sources coincide: translation only
        return 1.0, np.eye(2), md - ms
    z = (w * np.conj(zs) * zd).sum() / denom
    s = float(abs(z))
    if s < 1e-12:
        return 1.0, np.eye(2), md - ms
    c, sn = z.real / s, z.imag / s
    R = np.array([[c, -sn], [sn, c]])
    T = md - s * (R @ ms)
    return s, R, T


def _robust_sigma(r):
    """MAD-based scale of the residuals: adapts to the frame. If the tissue really
    deformed, every probe is a bit off and the tolerance widens with them; if all but
    one agree, the scale stays tight and that one gets rejected."""
    return 1.4826 * float(np.median(np.abs(r)))


def _tukey(r, sigma):
    """Redescending weight in [0,1]: 1 for a probe on the consensus, exactly 0 beyond
    TUKEY_C sigmas -- an outlier is discarded, not merely discounted."""
    c = TUKEY_C * max(float(sigma), 1e-6)
    u = np.clip(np.abs(np.asarray(r, np.float64)) / c, 0.0, 1.0)
    return (1.0 - u ** 2) ** 2


def _fit_once(arch, old, us, P, w, scale_clip):
    """One weighted fit of the arch onto probe positions P.

    Stage 1: weighted similarity from the probes' old positions to the new ones,
    applied to (left, right, apex) -- keeps the parabola perfect and clamps the
    per-step scale so the arch cannot 'breathe' when the evidence bunches up.
    Stage 2: weighted-least-squares height refresh, v_i ~ h*(1-u_i^2): probes near
    the tip carry the most height information, probes at the sides none -- so a
    mid-arc bend updates the curvature without touching the arm extent."""
    s, R, T = fit_similarity(old, P, w)
    s = float(np.clip(s, *scale_clip))
    wn = w / w.sum()                       # re-derive T so centroids match at clipped scale
    T = (wn[:, None] * P).sum(0) - s * (R @ (wn[:, None] * old).sum(0))
    left = s * (R @ np.asarray(arch.left)) + T
    right = s * (R @ np.asarray(arch.right)) + T
    apex = s * (R @ np.asarray(arch.apex)) + T
    out = Arch(left, right, 0.0, source='tracked', power=arch.power)
    out.set_apex(*apex)

    mid2, t2, n2, d2 = out._frame()
    v = ((P - (mid2 + (us[:, None] * d2) * t2)) * n2).sum(1)
    b = out.shape(us)
    den = float((w * b * b).sum())
    if den > 1e-9 and den / w.sum() > 0.05:   # enough mid-arc evidence to trust
        out.height = float((w * v * b).sum() / den)
    return out


def refit(arch, us, new_pts, weights, scale_clip=SCALE_CLIP, iters=IRLS_ITERS):
    """New Arch fitted to tracked probes, or None if the probes disagree too much to
    fit at all (caller should hold the previous pose). us are the probes' arc
    parameters (from sample_probes), new_pts their tracked positions, weights their
    appearance confidences from the tracker.

    Robust: fit, measure each probe's distance from the fitted parabola, downweight
    the ones that disagree with the consensus, refit. Final trust per probe is
    appearance * agreement, so a probe that still looks right but jumped somewhere
    the others contradict ends up ignored. out.conf is the mean surviving trust."""
    us = np.asarray(us, np.float64)
    P = np.asarray(new_pts, np.float64)
    w0 = np.clip(np.asarray(weights, np.float64), 0.0, 1.0)
    old = arch.at(us)
    # tolerance is adaptive but bounded: never tighter than tracker jitter, never
    # looser than a fraction of the arch's own size (see SIGMA_CEIL_FRAC)
    sigma_ceil = max(SIGMA_CEIL_FRAC * arch._frame()[3], SIGMA_FLOOR)

    w = w0.copy()
    out = None
    for _ in range(max(1, iters)):
        if w.sum() < 1e-6:
            return None                    # nothing left to fit to
        out = _fit_once(arch, old, us, P, w, scale_clip)
        r = np.linalg.norm(out.at(us) - P, axis=1)
        sigma = float(np.clip(_robust_sigma(r), SIGMA_FLOOR, sigma_ceil))
        w = w0 * _tukey(r, sigma)

    if out is None or w.sum() < 1e-6:
        return None
    # Kish effective sample size: how many probes are really voting once the
    # disagreers are discounted. Two is the minimum a similarity fit needs.
    n_eff = float(w.sum() ** 2 / (w ** 2).sum())
    if n_eff < MIN_EFF_PROBES:
        return None
    out.conf = float(np.clip(w.sum() / len(w), 0.0, 1.0))
    return out


def _pt(p):
    return (int(round(p[0])), int(round(p[1])))


def _arc_color(arch):
    """Cyan = trusted, amber = tracked but the probes disagreed (nudge it),
    gray = tracking lost the tissue and this frame is holding the last pose."""
    source = getattr(arch, 'source', 'manual')
    if source == 'hold':
        return HOLD_COLOR
    if source == 'tracked' and getattr(arch, 'conf', 1.0) < LOW_CONF:
        return SHAKY_COLOR
    return ARC_COLOR


def draw(img, arches, editing=False, pending=()):
    """Draw the arcs onto img in place (cv2 clips anything off-image, incl. the tip).
    editing adds drag handles + a mid-line cue; pending are the placement clicks."""
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3:
        return  # e.g. torch tensor from the fast 'image' mode -- nothing to draw on
    for arch in arches:
        arc_color = _arc_color(arch)
        pts = np.round(arch.points()).astype(np.int32)
        cv2.polylines(img, [pts], False, arc_color, THICKNESS, cv2.LINE_AA)
        if editing:
            mid, _, _, _ = arch._frame()
            cv2.line(img, _pt(mid), _pt(arch.apex), TIP_COLOR, 1, cv2.LINE_AA)
            for name, p in arch.handles():
                color = TIP_COLOR if name == 'apex' else arc_color
                cv2.circle(img, _pt(p), HANDLE_R, color, -1, cv2.LINE_AA)
                cv2.circle(img, _pt(p), HANDLE_R, (255, 255, 255), 1, cv2.LINE_AA)
    # placement preview: the clicked sides, then base chord + mid-line guide for the tip
    for p in pending:
        cv2.circle(img, _pt(p), HANDLE_R - 1, ARC_COLOR, -1, cv2.LINE_AA)
    if len(pending) == 2:
        (ax, ay), (bx, by) = pending
        cv2.line(img, _pt(pending[0]), _pt(pending[1]), ARC_COLOR, 1, cv2.LINE_AA)
        n = np.array([by - ay, -(bx - ax)], np.float64)
        norm = float(np.linalg.norm(n))
        if norm > 1e-6:
            n /= norm
            mid = np.array([(ax + bx) / 2, (ay + by) / 2])
            reach = img.shape[0] + img.shape[1]  # long enough to cross the whole image
            cv2.line(img, _pt(mid - n * reach), _pt(mid + n * reach), TIP_COLOR, 1, cv2.LINE_AA)


def save(workspace, arch_by_frame):
    """Persist {frame_index: Arch} to <workspace>/arches.json (format version 2)."""
    frames = {str(ti): {'left': a.left, 'right': a.right, 'height': a.height,
                        'power': a.power, 'source': a.source, 'conf': round(a.conf, 4)}
              for ti, a in sorted(arch_by_frame.items())}
    with open(path.join(workspace, ARCHES_FILE), 'w') as f:
        json.dump({'version': 2, 'frames': frames}, f, indent=2)


def load(workspace):
    """Load {frame_index: Arch}. A legacy v1 file (single global arch as a list)
    is migrated onto frame 0 so the annotation is not lost."""
    file = path.join(workspace, ARCHES_FILE)
    if not path.exists(file):
        return {}
    try:
        with open(file) as f:
            data = json.load(f)
        if isinstance(data, list):         # legacy v1: one arch shown on all frames
            if not data:
                return {}
            a = data[0]
            print('[retzius_arch] legacy single-arch file migrated onto frame 0 -- '
                  're-place it on your keyframe and TRACK.')
            return {0: Arch(a['left'], a['right'], a['height'])}
        return {int(ti): Arch(a['left'], a['right'], a['height'],
                              a.get('source', 'manual'), a.get('conf', 1.0),
                              a.get('power', DEFAULT_POWER))   # pre-power files: parabola
                for ti, a in data['frames'].items()}
    except (ValueError, KeyError, TypeError, IndexError) as e:
        print(f'[retzius_arch] could not read {file}: {e}')
        return {}


if __name__ == '__main__':
    # geometry: tip projects onto the mid-line, sides stay pinned, off-image tip is stable
    a = Arch((100, 200), (300, 220), 0.0)
    a.set_apex(240, -50)                       # off-axis click far above the image
    mid = np.array([200.0, 210.0])
    chord = np.array([200.0, 20.0])
    assert abs(np.dot(np.array(a.apex) - mid, chord)) < 1e-6   # tip exactly on the mid-line
    assert a.apex[1] < 0                                       # tip off-image is allowed
    pts = a.points()
    assert np.allclose(pts[0], a.left) and np.allclose(pts[-1], a.right)  # sides exact
    assert np.all(np.isfinite(pts))

    # moving a side keeps the same height and re-centres the tip on the new mid-line
    h0 = a.height
    move_handle(a, 'left', 120, 260, 120, 260)
    assert a.height == h0
    new_mid = np.array([210.0, 240.0])
    new_chord = np.array([180.0, -40.0])
    assert abs(np.dot(np.array(a.apex) - new_mid, new_chord)) < 1e-6

    # drawing clips the off-image tip without complaint; handles + pending preview render
    img = np.zeros((240, 320, 3), np.uint8)
    draw(img, [a], editing=True, pending=[(10, 10), (50, 50)])
    assert img.any()
    draw(np.zeros((2, 2, 3), np.uint8), [Arch((0, 0), (0, 0), 5)])  # degenerate chord: no crash

    # shape exponent: sides+tip pinned for every p, and a lower p means a sharper tip
    for p in (POWER_RANGE[0], DEFAULT_POWER, POWER_RANGE[1]):
        q = Arch((60, 150), (260, 140), 80, power=p)
        qs = q.points()
        assert np.allclose(qs[0], q.left) and np.allclose(qs[-1], q.right)
        assert np.allclose(q.at([0.0])[0], q.apex)
    sharp, flat = Arch((0, 100), (200, 100), 50, power=1.5), Arch((0, 100), (200, 100), 50, power=3)
    assert sharp.at([0.5])[0][1] > flat.at([0.5])[0][1]   # +y is down: sharp sits lower
    assert Arch((0, 0), (1, 1), 1, power=99).power == POWER_RANGE[1]   # clamped

    # save/load roundtrip (v2 per-frame format, source + conf preserved)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        a.source, a.conf, a.power = 'tracked', 0.42, 2.5
        save(d, {7: a})
        loaded = load(d)
        b = loaded[7]
        assert list(loaded) == [7] and b.source == 'tracked' and abs(b.conf - 0.42) < 1e-4
        assert b.left == a.left and b.right == a.right and abs(b.height - a.height) < 1e-9
        assert b.power == 2.5
        # legacy v1 list migrates onto frame 0; a file without 'power' is a plain parabola
        with open(path.join(d, ARCHES_FILE), 'w') as f:
            json.dump([{'left': [1, 2], 'right': [3, 4], 'height': 5}], f)
        legacy = load(d)
        assert list(legacy) == [0] and legacy[0].left == (1.0, 2.0)
        with open(path.join(d, ARCHES_FILE), 'w') as f:
            json.dump({'version': 2, 'frames': {'3': {'left': [1, 2], 'right': [3, 4],
                                                      'height': 5}}}, f)
        assert load(d)[3].power == DEFAULT_POWER
    a.power = DEFAULT_POWER

    # colour reflects trust: cyan tracked, amber shaky, gray held, cyan manual
    assert _arc_color(Arch((0, 0), (10, 0), 5, 'tracked', 0.9)) == ARC_COLOR
    assert _arc_color(Arch((0, 0), (10, 0), 5, 'tracked', 0.1)) == SHAKY_COLOR
    assert _arc_color(Arch((0, 0), (10, 0), 5, 'hold', 0.0)) == HOLD_COLOR
    assert _arc_color(Arch((0, 0), (10, 0), 5, 'manual', 1.0)) == ARC_COLOR

    # hit test grabs the nearest handle
    assert hit_test([a], a.left[0] + 3, a.left[1] - 3, radius=8)[1] == 'left'
    assert hit_test([a], -500, -500, radius=8) is None

    # probe sampling: only visible arc points, spread out, u/pos consistent
    c = Arch((60, 150), (260, 140), 200)          # tip far above a 240x320 image
    us, pts = sample_probes(c, 320, 240)
    assert 2 <= len(us) <= PROBE_N
    for u, p in zip(us, pts):
        assert PROBE_MARGIN <= p[0] <= 320 - 1 - PROBE_MARGIN
        assert PROBE_MARGIN <= p[1] <= 240 - 1 - PROBE_MARGIN
        mid_c, t_c, n_c, d_c = c._frame()
        q = mid_c + u * d_c * t_c + (1 - u ** 2) * c.height * n_c
        assert np.allclose(p, q)
    # fully off-image arch yields no probes
    assert sample_probes(Arch((-500, -500), (-400, -500), 10), 320, 240) == ([], [])

    # occluder mask: probes are never seeded on an instrument, and visibility grades
    blk = np.zeros((240, 320), bool)
    blk[:, 150:200] = True                        # an 'instrument' band across the arc
    us_b, pts_b = sample_probes(c, 320, 240, blocked=blk)
    assert len(us_b) >= 2
    for p in pts_b:
        assert not blk[int(round(p[1])), int(round(p[0]))], p
        assert not (150 - BLOCK_MARGIN < p[0] < 200 + BLOCK_MARGIN), p   # margin honoured
    assert probe_visibility(blk, [(175, 100)])[0] == 0.0        # dead centre of the band
    assert probe_visibility(blk, [(20, 100)])[0] == 1.0         # far clear of it
    assert 0.0 < probe_visibility(blk, [(150 - BLOCK_MARGIN // 2, 100)])[0] < 1.0  # edge
    assert probe_visibility(None, [(175, 100)])[0] == 1.0       # no mask -> all visible
    assert len(probe_visibility(None, [])) == 0
    # an arch fully behind the occluder can't be tracked at all
    blk_all = np.ones((240, 320), bool)
    assert sample_probes(c, 320, 240, blocked=blk_all) == ([], [])

    # fit_similarity recovers a known transform (weighted, no reflection)
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 100, (6, 2))
    ang, sc, tr = 0.3, 1.07, np.array([12.0, -7.0])
    Rm = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = (sc * (Rm @ src.T)).T + tr
    s_f, R_f, T_f = fit_similarity(src, dst, weights=rng.uniform(0.5, 1.0, 6))
    assert abs(s_f - sc) < 1e-9 and np.allclose(R_f, Rm) and np.allclose(T_f, tr)

    # refit: probes moved by a small rigid transform -> the whole arch follows it,
    # stays a perfect parabola, and keeps its chord length (scale within clip)
    c2 = Arch((60, 150), (260, 140), 60)
    us2, pts2 = sample_probes(c2, 320, 240)
    moved = [(p[0] + 9, p[1] - 4) for p in pts2]
    r2 = refit(c2, us2, moved, np.ones(len(us2)))
    assert np.allclose(r2.left, (60 + 9, 150 - 4), atol=1e-6)
    assert np.allclose(r2.right, (260 + 9, 140 - 4), atol=1e-6)
    assert abs(r2.height - 60) < 1e-6 and r2.source == 'tracked'
    # ... and a non-parabolic arch keeps its shape through tracking
    c2p = Arch((60, 150), (260, 140), 60, power=2.5)
    us2p, pts2p = sample_probes(c2p, 320, 240)
    r2p = refit(c2p, us2p, [(p[0] + 9, p[1] - 4) for p in pts2p], np.ones(len(us2p)))
    assert r2p.power == 2.5 and abs(r2p.height - 60) < 1e-6
    assert r2.conf > 0.99, r2.conf          # everyone agreed

    # ROBUSTNESS: one probe jumps far away with a PERFECT appearance score (the
    # failure NCC cannot see). The consensus must ignore it entirely.
    rogue = list(moved)
    rogue[len(rogue) // 2] = (rogue[len(rogue) // 2][0] + 55,
                              rogue[len(rogue) // 2][1] - 45)
    r5 = refit(c2, us2, rogue, np.ones(len(us2)))
    assert np.allclose(r5.left, (69, 146), atol=0.5), r5.left     # same as the clean fit
    assert np.allclose(r5.right, (269, 136), atol=0.5), r5.right
    assert r5.conf < r2.conf, 'a disagreeing probe must lower the frame confidence'
    # ... and a low-appearance probe is discounted even when it agrees
    r6 = refit(c2, us2, moved, [0.2] * len(us2))
    assert r6 is not None and r6.conf < 0.25

    # too much disagreement -> None (caller holds the previous pose rather than
    # inventing a fit); here every probe is scattered randomly
    scatter = [(p[0] + rng.uniform(-90, 90), p[1] + rng.uniform(-90, 90)) for p in pts2]
    bad = refit(c2, us2, scatter, np.ones(len(us2)), iters=IRLS_ITERS)
    assert bad is None or bad.conf < LOW_CONF
    assert refit(c2, us2, moved, np.zeros(len(us2))) is None      # no appearance trust at all
    # mid-arc probes pushed outward -> height grows (stage 1 absorbs the mean shift
    # as translation, stage 2 turns the remaining mid-arc bulge into curvature)
    mid2_, t2_, n2_, _ = c2._frame()
    bulged = [tuple(np.asarray(p) + (1 - u ** 2) * 10 * n2_) for u, p in zip(us2, pts2)]
    r3 = refit(c2, us2, bulged, np.ones(len(us2)))
    assert r3.height > c2.height + 1.5
    # a suspicious spread is clamped, not followed: probes 1.3x apart may grow the
    # chord by at most SCALE_CLIP -- the arch must not 'breathe'
    chord = lambda A: np.linalg.norm(np.subtract(A.right, A.left))
    spread = [tuple(mid2_ + 1.3 * (np.asarray(p) - mid2_)) for p in pts2]
    r4 = refit(c2, us2, spread, np.ones(len(us2)))
    assert r4 is not None
    assert chord(r4) <= chord(c2) * SCALE_CLIP[1] + 1e-6, chord(r4) / chord(c2)
    # an ABSURD spread is rejected outright (residual ceiling) rather than clamped:
    # nothing in one frame step doubles the arch, so hold instead of inventing a fit
    absurd = [tuple(mid2_ + 2.0 * (np.asarray(p) - mid2_)) for p in pts2]
    assert refit(c2, us2, absurd, np.ones(len(us2))) is None

    print('[retzius_arch] ok | apex=(%.1f, %.1f) height=%.1f | probes=%d'
          % (*a.apex, a.height, len(us)))
