"""EndoDAC monocular depth estimation for the ATLAS GUI.

EndoDAC is DINOv2-ViT + DV-LoRA + a DPT depth head, fine-tuned for endoscopy.
The released checkpoint (depth_model.pth) is a single model: backbone_size="base"
(ViT-B/14), dvlora rank 4, input 256x320 -- there is NO small/large released
checkpoint, so the encoder size is fixed here. depth_model.pth holds the full
state (backbone + lora + head), so we build with pretrained_path=None and load it
directly. Loading logic mirrors EndoDAC/test_simple.py.
"""
import os
import sys
import logging
import warnings
from pathlib import Path

# EndoDAC's DINOv2 backbone is fine without xFormers: the fallbacks are the same
# math (MemEffAttention -> standard attention, SwiGLUFFN -> pure torch), and ViT-B
# correctly uses an MLP FFN. Disable xFormers and mute the resulting import noise.
os.environ.setdefault("XFORMERS_DISABLED", "1")
warnings.filterwarnings("ignore", message=".*xFormers.*")
logging.getLogger("dinov2").setLevel(logging.WARNING)  # mute "using MLP layer as FFN"

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# --- CONFIG ---------------------------------------------------------------
_REPO       = Path(__file__).resolve().parents[1]               # ...\ATLAS-Interactive
ENDODAC_DIR = _REPO.parent / "backbones" / "EndoDAC" / "EndoDAC"  # inner pkg dir (has models/)
# RARP-finetuned checkpoint (self-supervised EndoDAC finetune on our RARP clips, see
# code/DEPTH_REPORT.md, run 24270397). Warm-started from the released depth_model.pth.
CHECKPOINT  = _REPO.parent / "backbones" / "EndoDAC" / "depth_model.pth"
# released-checkpoint hyperparams (EndoDAC/test_simple.py defaults)
LORA_RANK   = 4
LORA_TYPE   = "dvlora"
RESIDUAL_BLOCKS = [2, 5, 8, 11]
# Internal inference resolution (h, w). MUST match the checkpoint's training resolution --
# the residual conv blocks are pinned to this patch grid. (392,490) is jobs/finetune_depth.sh's
# --image-shape default, what the deliverable RARP finetune was trained at (DEPTH_REPORT.md).
# The plain released depth_model.pth instead needs (224,280) -- swap this back if you revert.
IMAGE_SHAPE = (392, 490)
# Hard crop of N bottom rows BEFORE inference, to drop the baked-in da Vinci instrument
# banner so it can't corrupt the depth. Empirically the depth looks better cropping it out
# than feeding the full frame and masking it (training feeds the full frame, but at
# inference the banner still perturbs the ViT's global features). Cropped rows are padded
# as far. Set 0 for full-bleed clips with no banner.
BOTTOM_CROP_PX = 70
# Console-GUI overlay masking: da Vinci/CMR consoles bake in logos, instrument banners,
# and corner widgets that aren't anatomy. They're STATIC across a clip while tissue moves,
# so a temporal-variance mask over frames SAMPLED ACROSS THE CLIP isolates them -- and
# unlike the dark mask below it also catches BRIGHT widgets (e.g. CMR Versius corner icons
# on full-bleed video). One trick handles full-bleed, pillarbox+banner, vignette and corner
# logos. Set MASK_OVERLAY = False to disable. Mirrors scripts/finetune_depth.py (which masks
# the full frame; here the bottom banner is already removed by BOTTOM_CROP_PX, so the mask
# runs on the cropped ROI and just needs to catch the side/corner/vignette overlays).
MASK_OVERLAY = True
_OV_N = 16             # frames sampled across the clip to estimate the mask
_OV_STD = 6.0          # per-pixel temporal std (0-255) below which a pixel is "overlay"
_OV_GRID = (128, 160)  # (h, w) the variance is computed at
# --------------------------------------------------------------------------

_model = None       # ponytail: lazy global, fine for single-model GUI


def _load(device: str):
    global _model
    if _model is None:
        sys.path.insert(0, str(ENDODAC_DIR))  # ponytail: matches segmenter.py path-hack
        import models.endodac as endodac
        import models.backbones as backbones
        sd = torch.load(str(CHECKPOINT), map_location=device)
        # endodac() doesn't thread image_shape to the backbone's residual blocks
        # (they're pinned to input_size=(224,280)); inject IMAGE_SHAPE so they match.
        _orig = backbones.vits.vit_base
        backbones.vits.vit_base = lambda **kw: _orig(input_size=IMAGE_SHAPE, **kw)
        try:
            m = endodac.endodac(
                backbone_size="base", r=LORA_RANK, lora_type=LORA_TYPE,
                image_shape=IMAGE_SHAPE, pretrained_path=None,
                residual_block_indexes=RESIDUAL_BLOCKS, include_cls_token=True)
        finally:
            backbones.vits.vit_base = _orig
        model_dict = m.state_dict()
        m.load_state_dict({k: v for k, v in sd.items() if k in model_dict})
        _model = m.to(device).eval()
    return _model


def _overlay_valid(frames_roi):
    """Temporal-variance overlay mask. The console GUI (logos, instrument banner, corner
    widgets, vignette) is constant across a clip while anatomy moves, so per-pixel std over
    frames sampled across the clip isolates anatomy (valid = std > _OV_STD).
    frames_roi: list of HxWx3 uint8 ROI frames (same crop as the frame being scored).
    Returns a bool mask at ROI resolution (True = anatomy), or None if too few frames or
    the scene is too static to trust. Mirrors scripts/finetune_depth.py's per-clip mask."""
    if len(frames_roi) < 6:
        return None
    gh, gw = _OV_GRID
    stack = np.stack([cv2.resize(f, (gw, gh), interpolation=cv2.INTER_AREA).mean(2)
                      for f in frames_roi]).astype(np.float32)
    valid = stack.std(0) > _OV_STD
    if valid.mean() < 0.25:                        # mostly static -> mask unreliable, skip
        return None
    ch, w = frames_roi[0].shape[:2]
    return cv2.resize(valid.astype(np.uint8), (w, ch),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


@torch.no_grad()
def estimate_depth(image_rgb: np.ndarray, device: str = "cpu",
                   overlay_frames=None) -> np.ndarray:
    """image_rgb: HxWx3 uint8 (RGB). Returns HxW float32 disparity in [0,1]
    (higher = closer), normalized with a 95th-pct ceiling like EndoDAC's own viz.
    overlay_frames: optional list of HxWx3 uint8 frames sampled across the clip; used
    to build the static-overlay mask (needs >=6). Without them the overlay mask is
    skipped and only the dark-vignette mask applies."""
    model = _load(device)
    h, w = image_rgb.shape[:2]
    crop = min(BOTTOM_CROP_PX, h - 1)
    roi = image_rgb[:h - crop] if crop else image_rgb   # drop overlay before inference
    ch = roi.shape[0]

    ih, iw = IMAGE_SHAPE
    img = cv2.resize(roi, (iw, ih), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    disp = model(x)[("disp", 0)]                       # EndoDAC has no ImageNet norm
    disp = F.interpolate(disp, (ch, w), mode="bilinear", align_corners=False)
    disp = disp[0, 0].cpu().numpy()

    # Endoscope frames have a black circular vignette; the model reads that border
    # as "near" and it dominates the colormap. Normalize over the valid region only
    # and push the border to far, so the anatomy gets the full color range.
    valid = roi.mean(2) > 10
    if valid.sum() < valid.size * 0.05:                # no real vignette -> use all
        valid = np.ones_like(valid)
    if MASK_OVERLAY and overlay_frames:                # also drop static console-GUI widgets
        frames_roi = [f[:f.shape[0] - crop] if crop else f for f in overlay_frames]
        ov = _overlay_valid(frames_roi)
        if (ov is not None and ov.shape == valid.shape
                and (valid & ov).sum() > valid.size * 0.05):
            valid = valid & ov
    lo, hi = disp[valid].min(), np.percentile(disp[valid], 95)
    out = np.clip((disp - lo) / (hi - lo + 1e-8), 0, 1)
    out[~valid] = 0
    if crop:                                           # pad the overlay strip as far
        full = np.zeros((h, w), np.float32)
        full[:ch] = out
        out = full
    return out.astype(np.float32)


if __name__ == "__main__":
    # overlay-mask self-check (pure numpy): a static corner widget stays masked while a
    # moving region is kept.
    _rng = np.random.default_rng(0)
    frames = []
    for _ in range(_OV_N):
        f = (_rng.random((200, 260, 3)) * 255).astype(np.uint8)   # moving "anatomy"
        f[:40, :60] = 80                                          # static "widget"
        frames.append(f)
    ov = _overlay_valid(frames)
    assert ov is not None, "overlay mask should be available with enough frames"
    assert ov[:40, :60].mean() < 0.2, "static corner widget should be masked out"
    assert ov[120:, 150:].mean() > 0.8, "moving region should be kept"
    print("[depth] overlay-mask self-check ok")

    # self-check: real checkpoint load + inference on a dummy frame
    dummy = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
    d = estimate_depth(dummy, "cpu")
    assert d.shape == (240, 320) and d.dtype == np.float32, (d.shape, d.dtype)
    assert 0.0 <= d.min() and d.max() <= 1.0, (d.min(), d.max())
    print("[depth] ok | image_shape=%s out=%s range=[%.3f, %.3f]"
          % (IMAGE_SHAPE, d.shape, d.min(), d.max()))
