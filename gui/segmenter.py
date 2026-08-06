"""Standalone, configurable segmentation + measurement helpers for the ATLAS GUI.

Everything tweakable lives in the CONFIG block below: the checkpoint, the input
size, the anchor's physical size, where the SurgeNet model code is, and how the
model's class ids map onto the GUI's object ids (see gui/cutie/utils/palette.py).

The GUI calls three things from here:
    segment(image_rgb, device) -> HxW uint8 mask in GUI object ids   (SEGMENT button)
    euclidean(p, q)            -> pixel distance between two points
    mm_per_pixel(anchor_px)    -> scale from the 6 mm ANCHOR measurement
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# --- CONFIG ---------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[1]          # ...\ATLAS-Interactive
CHECKPOINT = _REPO.parent / "backbones" / "best.pth"  # ..\backbones\best.pth
IMG_SIZE = 512   # MUST match the seg training --img-size (finetune_segmentation.py).
                 # 512 matches the current best.pth; boundaries are already smoothed by
                 # the bilinear logit-upsample in segment(). For finer PREDICTION detail
                 # (thin urethra/catheter), retrain at 1024 and set this to 1024 to match.
ANCHOR_MM = 5.333                                     # physical size of the anchor
ROBOT_ANCHOR_MM = 8.0                                  # physical size of the robot anchor

# Where metaformer.py (MetaFormerFPN) lives. Adjust if you move the repo.
SURGENET_DIR = _REPO.parent / "code" / "third_party" / "surgenet"

# model class id (argmax channel) -> GUI object id.
# model classes (from training RAW_NAMES): 0=bg 1=catheter 2=prostate 3=urethra 4=apicalvesicle
# GUI object ids (palette.custom_names):    1=Urethra 2=Prostate 4=Catheter 5=Non-anatomical
CLASS_TO_OBJECT = {1: 4, 2: 2, 3: 1, 4: 5}

# LIVE SUL point tracker (one OpenCV CSRT tracker per point + occlusion guard).
TRACK_BOX = 31             # px; size of the CSRT box tracked around each point
TRACK_APPEAR_THRESH = 0.4  # [0,1] a CSRT move is only accepted if the new patch still
                           #        matches the target template this well (rejects the
                           #        box sliding onto an occluding object)
TRACK_NCC_THRESH = 0.5     # [0,1] template-match score needed to re-acquire after loss
TRACK_REFRESH_NCC = 0.8    # [0,1] only refresh the template on a match this strong, so
                           #        an occluded/ambiguous patch never overwrites the target
TRACK_TEMPLATE = 21        # px; appearance patch size saved per point
TRACK_SEARCH = 61          # px; window searched when re-acquiring a lost point
TRACK_MAX_LOST = 30        # frames a point may stay lost before it's 'failed' (~1s @30fps)
# --------------------------------------------------------------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

_model = None  # ponytail: lazy single global model, fine for a one-checkpoint GUI


def _load(device):
    global _model
    if _model is None:
        sys.path.insert(0, str(SURGENET_DIR))
        from metaformer import MetaFormerFPN
        sd = torch.load(CHECKPOINT, map_location=device)
        nc = sd["FPN.segmentation_head.0.bias"].shape[0]
        # pretrained_weights=None -> builds from random init (no download), then we load sd
        model = MetaFormerFPN(num_classes=nc, pretrained="ImageNet", pretrained_weights=None)
        model.load_state_dict(sd)
        _model = model.to(device).eval()
    return _model


@torch.no_grad()
def segment(image_rgb: np.ndarray, device: str = "cpu") -> np.ndarray:
    """image_rgb: HxWx3 uint8 (RGB). Returns HxW uint8 mask in GUI object ids."""
    model = _load(device)
    h, w = image_rgb.shape[:2]
    img = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device)
    logits = model(x)                                  # (1, C, IMG_SIZE, IMG_SIZE)
    # upsample class logits to full frame BEFORE argmax -> smooth boundaries at native
    # resolution, instead of blocky NEAREST upscaling of a 512-grid argmax
    logits = F.interpolate(logits, (h, w), mode="bilinear", align_corners=False)
    pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    out = np.zeros((h, w), np.uint8)
    for cls, obj in CLASS_TO_OBJECT.items():
        out[pred == cls] = obj
    return out


def euclidean(p, q) -> float:
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


# SUL-depth: the plain SUL is the in-plane (projected) length; SUL depth adds the
# out-of-plane term from the depth map so a line that dives toward/away from the camera
# reads longer. EndoDAC depth is RELATIVE (normalized disparity in [0,1]), so DEPTH_SCALE_PX
# is a calibration knob, not physics: it maps the full [0,1] depth range to this many pixels
# of z. Calibrate it once against a known 3D length (raise it until the depth SUL of a
# reference matches reality); 0 makes SUL depth identical to plain SUL.
DEPTH_SCALE_PX = 300.0
DEPTH_WIN = 5  # px half-window; depth at a clicked point is the median over this window


def depth_at(depth_map: np.ndarray, p, win: int = DEPTH_WIN) -> float:
    """Median depth in a small window around p (robust to per-pixel noise). 0 if none valid."""
    h, w = depth_map.shape
    x, y = int(round(p[0])), int(round(p[1]))
    patch = depth_map[max(0, y - win):y + win + 1, max(0, x - win):x + win + 1]
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size else 0.0


def euclidean3d(p, q, dp: float, dq: float, depth_scale: float = DEPTH_SCALE_PX) -> float:
    """In-plane distance plus an out-of-plane term from the depth difference at p,q
    (dp,dq are the [0,1] depths). Same pixel units as euclidean(), so anchor mm/px applies."""
    dz = (dp - dq) * depth_scale
    return (euclidean(p, q) ** 2 + dz ** 2) ** 0.5


def solve_depth_scale(p, q, dp: float, dq: float, real_mm: float, mm_per_px: float) -> float:
    """Solve DEPTH_SCALE_PX from a reference object of known real_mm length, tilted so it
    has a real depth difference (dp,dq at p,q; both from depth_at() on a DEPTH-computed frame).
    mm_per_px comes from the ANCHOR click on the same frame/video. Raises ValueError if the
    object has no usable depth separation (flat-on) or the geometry doesn't fit the model
    (real_mm too short for the observed in-plane pixel span)."""
    d_depth = dp - dq
    if abs(d_depth) < 1e-6:
        raise ValueError('reference has no depth separation (flat-on) -- pick a tilted object')
    target_px2 = (real_mm / mm_per_px) ** 2
    inplane_px2 = euclidean(p, q) ** 2
    if target_px2 <= inplane_px2:
        raise ValueError('real_mm is too short for the in-plane pixel span -- '
                          'check mm_per_px / real_mm / point picks')
    return (target_px2 - inplane_px2) ** 0.5 / abs(d_depth)


def mm_per_pixel(anchor_px: float, anchor_mm: float = ANCHOR_MM) -> float:
    """mm per pixel given the anchor span (px) maps to anchor_mm millimetres."""
    return anchor_mm / anchor_px if anchor_px else 0.0


def _gray(frame: np.ndarray) -> np.ndarray:
    return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def _blocked_at(blocked: np.ndarray, p) -> bool:
    """True if p sits on an occluder pixel (or outside the frame)."""
    x, y = int(round(p[0])), int(round(p[1]))
    h, w = blocked.shape
    return not (0 <= x < w and 0 <= y < h) or bool(blocked[y, x])


def _make_csrt():
    """A CSRT tracker instance, or None on OpenCV builds without one (e.g. 5.x
    dropped the contrib trackers). Callers fall back to LK optical flow then."""
    for factory in (getattr(cv2, 'TrackerCSRT_create', None),
                    getattr(getattr(cv2, 'TrackerCSRT', None), 'create', None),
                    getattr(getattr(cv2, 'legacy', None), 'TrackerCSRT_create', None)):
        if factory is not None:
            try:
                return factory()
            except cv2.error:
                pass
    return None


class PointTracker:
    """Multi-point tracker for LIVE SUL and for the Retzius arch's probes.

    Motion model: one OpenCV CSRT tracker per point (a discriminative
    correlation-filter tracker that copes with tissue deformation, scale and
    lighting). On OpenCV builds without CSRT (5.x dropped the contrib trackers)
    it falls back to pyramidal Lucas-Kanade optical flow, which is core OpenCV;
    the appearance/occlusion logic below is identical for both backends.

    Occlusion guard: a proposed move is only accepted if the new patch still
    matches the saved target template (NCC >= appear_thresh). If it doesn't --
    the point has drifted onto an occluding object -- it HOLDS its last good
    position and the motion model is reset to that spot, so it can never chase
    the occluder. It re-locks (via a template search around the hold point) when
    the target reappears nearby. The appearance template is only refreshed on a
    strong match, so an occluded patch never overwrites the target's look.

    step() returns, per point: pos (x,y), conf in [0,1] (the NCC match score),
    state in {'tracked','reacquired','lost'}, and failed=True once it has been
    lost longer than max_lost frames.
    """

    def __init__(self, box=TRACK_BOX, appear_thresh=TRACK_APPEAR_THRESH,
                 ncc_thresh=TRACK_NCC_THRESH, refresh_ncc=TRACK_REFRESH_NCC,
                 template=TRACK_TEMPLATE, search=TRACK_SEARCH, max_lost=TRACK_MAX_LOST):
        self.b = box // 2
        self.appear_thresh = appear_thresh
        self.ncc_thresh = ncc_thresh
        self.refresh_ncc = refresh_ncc
        self.t = template // 2
        self.s = search // 2
        self.max_lost = max_lost

    def init(self, frame, points):
        gray = _gray(frame)
        self.shape = gray.shape
        self.prev_gray = gray
        self.pts = np.array(points, np.float32).reshape(-1, 2)
        self.templates = [self._patch(gray, p) for p in self.pts]
        self.lost = [0] * len(self.pts)
        self.use_csrt = _make_csrt() is not None
        self.trackers = ([self._new_tracker(frame, p) for p in self.pts]
                         if self.use_csrt else None)

    def _new_tracker(self, frame, p):
        tr = _make_csrt()
        tr.init(frame, self._box(p))
        return tr

    def _box(self, p):
        H, W = self.shape
        side = 2 * self.b + 1
        x = int(min(max(0, round(p[0]) - self.b), W - side))
        y = int(min(max(0, round(p[1]) - self.b), H - side))
        return (x, y, side, side)

    def _patch(self, gray, p):
        x, y = int(round(p[0])), int(round(p[1]))
        h, w = gray.shape
        x0, x1 = max(0, x - self.t), min(w, x + self.t + 1)
        y0, y1 = max(0, y - self.t), min(h, y + self.t + 1)
        return gray[y0:y1, x0:x1].copy()

    def _match_score(self, gray, p, template):
        """NCC of the patch centred on p against template, [-1,1]. 0 if off-edge."""
        th, tw = template.shape
        x, y = int(round(p[0])), int(round(p[1]))
        x0, y0 = x - tw // 2, y - th // 2
        h, w = gray.shape
        if x0 < 0 or y0 < 0 or x0 + tw > w or y0 + th > h:
            return 0.0
        patch = gray[y0:y0 + th, x0:x0 + tw]
        return float(cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)[0, 0])

    def _reacquire(self, gray, p, template, blocked=None):
        x, y = int(round(p[0])), int(round(p[1]))
        h, w = gray.shape
        x0, x1 = max(0, x - self.s), min(w, x + self.s + 1)
        y0, y1 = max(0, y - self.s), min(h, y + self.s + 1)
        win = gray[y0:y1, x0:x1]
        th, tw = template.shape
        if win.shape[0] < th or win.shape[1] < tw:
            return None, 0.0
        res = cv2.matchTemplate(win, template, cv2.TM_CCOEFF_NORMED)
        if blocked is not None:
            # never re-lock onto the occluder: res[i,j] is the score for the template
            # centred at (x0+j+tw//2, y0+i+th//2), so veto the centres that sit on it
            bwin = blocked[y0:y1, x0:x1]
            cent = bwin[th // 2:th // 2 + res.shape[0], tw // 2:tw // 2 + res.shape[1]]
            if cent.shape == res.shape:
                res = np.where(cent, -1.0, res)
        _, score, _, loc = cv2.minMaxLoc(res)
        nx = x0 + loc[0] + tw // 2
        ny = y0 + loc[1] + th // 2
        return np.array([nx, ny], np.float32), float(score)

    def step(self, frame, blocked=None):
        """Advance every point one frame. blocked is an optional HxW bool array marking
        occluders (robot instruments): a point that lands on one is treated as occluded
        -- it holds instead of following whatever is drawn on the instrument."""
        gray = _gray(frame)
        if not self.use_csrt:
            # LK fallback: flow all points from the previous frame in one call
            p0 = self.pts.reshape(-1, 1, 2).astype(np.float32)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, p0, None,
                winSize=(2 * self.b + 1, 2 * self.b + 1), maxLevel=3)
        out = []
        for i in range(len(self.pts)):
            if self.use_csrt:
                ok, box = self.trackers[i].update(frame)
                cand = (box[0] + box[2] / 2, box[1] + box[3] / 2) if ok else None
            else:
                ok = bool(st[i][0])
                cand = (float(p1[i, 0, 0]), float(p1[i, 0, 1])) if ok else None
            # the mask, when present, is ground truth: a point that landed on an
            # instrument is occluded no matter how well its patch happens to score
            if ok and blocked is not None and _blocked_at(blocked, cand):
                ok = False
            # appearance gate: the motion model saying "ok" is not enough -- the patch
            # must still look like the target, else the point slid onto an occluder.
            appear = self._match_score(gray, cand, self.templates[i]) if ok else -1.0
            if appear >= self.appear_thresh:                # genuinely on the target
                self.pts[i] = np.array(cand, np.float32)
                self.lost[i] = 0
                conf, state = appear, 'tracked'
                if appear >= self.refresh_ncc:              # refresh only on a strong match
                    self.templates[i] = self._patch(gray, self.pts[i])
            else:
                # occluded/drifted: HOLD the last good position and look for the target
                # reappearing there; never follow the occluding object.
                pos, score = self._reacquire(gray, self.pts[i], self.templates[i], blocked)
                if pos is not None and score >= self.ncc_thresh:
                    self.pts[i] = pos
                    self.lost[i] = 0
                    conf, state = score, 'reacquired'
                else:
                    self.lost[i] += 1
                    conf, state = 0.0, 'lost'
                if self.use_csrt:
                    # reset CSRT to the hold point so its model can't chase the occluder
                    # (LK needs no reset: it always flows from self.pts / prev_gray)
                    self.trackers[i] = self._new_tracker(frame, self.pts[i])
            out.append({'pos': (float(self.pts[i][0]), float(self.pts[i][1])),
                        'conf': float(conf), 'state': state,
                        'failed': self.lost[i] > self.max_lost})
        self.prev_gray = gray
        return out


if __name__ == "__main__":
    # self-check: measurement math + real checkpoint load/inference on a dummy frame
    assert abs(euclidean((0, 0), (3, 4)) - 5.0) < 1e-9
    assert abs(mm_per_pixel(60) - ANCHOR_MM / 60) < 1e-9          # anchor scale
    assert abs(euclidean((0, 0), (60, 0)) * mm_per_pixel(60) - ANCHOR_MM) < 1e-6
    # SUL depth: equal depths -> same as 2D; a depth gap adds an out-of-plane term
    assert abs(euclidean3d((0, 0), (3, 4), 0.5, 0.5) - 5.0) < 1e-9
    assert euclidean3d((0, 0), (3, 4), 0.7, 0.2) > 5.0
    _dm = np.zeros((20, 20), np.float32); _dm[8:12, 8:12] = 0.6
    assert abs(depth_at(_dm, (10, 10)) - 0.6) < 1e-6 and depth_at(_dm, (0, 0)) == 0.0
    # solve_depth_scale recovers the constant used to build its own test case
    _p, _q, _dp, _dq, _scale, _mmpx = (0, 0), (3, 4), 0.7, 0.2, 300.0, 0.5
    _real_mm = euclidean3d(_p, _q, _dp, _dq, _scale) * _mmpx
    assert abs(solve_depth_scale(_p, _q, _dp, _dq, _real_mm, _mmpx) - _scale) < 1e-6
    dummy = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
    m = segment(dummy, "cpu")
    assert m.shape == (240, 320) and m.dtype == np.uint8
    assert set(np.unique(m)).issubset({0, *CLASS_TO_OBJECT.values()})
    print("[segmenter] ok | mask classes:", np.unique(m))

    # point tracker self-check: follow a textured blob, then lose it under occlusion
    rng = np.random.default_rng(0)
    def scene(cx):
        img = (rng.random((100, 120, 3)) * 40).astype(np.uint8)   # faint texture
        cv2.circle(img, (cx, 50), 7, (240, 60, 60), -1)
        cv2.circle(img, (cx, 50), 3, (60, 240, 240), -1)          # bit of structure
        return img
    tr = PointTracker()
    tr.init(scene(40), [(40, 50)])
    r1 = tr.step(scene(46))[0]
    assert abs(r1["pos"][0] - 46) < 4 and r1["state"] == "tracked", r1
    # occluder slides through: target gone, a bright bar elsewhere.
    # the point must HOLD near the target (~46), not jump onto the occluder.
    occ = (rng.random((100, 120, 3)) * 40).astype(np.uint8)
    cv2.rectangle(occ, (85, 35), (115, 65), (255, 255, 255), -1)
    r2 = tr.step(occ)[0]
    assert r2["state"] != "tracked" and r2["pos"][0] < 65, r2

    # occluder mask: the blob is still perfectly visible and the patch matches, so
    # appearance alone would happily say "tracked" -- but the mask declares the place
    # it moved to an instrument, so the point must HOLD instead of following it there.
    tr2 = PointTracker()
    tr2.init(scene(40), [(40, 50)])               # seeded on clear tissue
    blocked = np.zeros((100, 120), bool)
    blocked[:, 44:70] = True                      # instrument covers the destination
    r3 = tr2.step(scene(46), blocked=blocked)[0]
    assert r3["state"] != "tracked", r3           # mask overrides a good-looking match
    assert r3["pos"][0] < 44, r3                  # never enters the masked region
    # it may re-lock onto the sliver of target still showing at the instrument edge
    # (~43 here) rather than sit exactly still; that probe's patch is then half masked,
    # so retzius_arch's visibility grading halves its weight and the agreement vote
    # drops it altogether if it starts creeping along the edge.
    assert 38 <= r3["pos"][0] < 44, r3
    # the mask is the only thing that changed: unmasked, the same frame tracks fine
    tr3 = PointTracker()
    tr3.init(scene(40), [(40, 50)])
    assert tr3.step(scene(46))[0]["state"] == "tracked"
    print("[tracker]  ok | track conf=%.2f, occluded -> state=%s pos_x=%.0f (held) | "
          "masked -> state=%s (held off the instrument)"
          % (r1["conf"], r2["state"], r2["pos"][0], r3["state"]))
