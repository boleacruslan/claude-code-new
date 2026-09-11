"""Natural skin retouching via three-band frequency separation.

The usual one-pass blur destroys pores and gives the plastic look. Here the
face is split into three bands:

  base  - tone and shading (edge-preserving filtered at a fixed working
          resolution, so behaviour does not change with megapixels)
  mid   - blotches, shadows and wrinkle lines; this is the band we attenuate,
          and dark mid-frequency detail is pushed down hardest because that
          is what a line or an eye bag actually is
  fine  - pores, lashes, stubble; kept nearly intact so skin still reads as skin

Only face regions are touched, and only inside a feathered skin mask that
excludes eyes, brows, lips, hair and background.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from . import imageio_utils as io
from .faces import Face, FaceAnalyzer, primary_faces, skin_mask

DETECT_SIZE = 1280   # landmarks are found on a downscaled copy
WORK_WIDTH = 900     # ROI width used for the base layer


@dataclass
class RetouchConfig:
    strength: float = 0.6        # overall blend of the retouched result
    texture_keep: float = 0.90   # fine band - keep pores
    blotch_keep: float = 0.55    # bright mid band
    wrinkle_keep: float = 0.35   # dark mid band - lines and bags
    smooth_radius: float = 60.0  # base-layer reach, in canonical ROI pixels
    # Contrast the base layer refuses to flatten. The single most important
    # knob: too low and the filter treats every wrinkle as an edge worth
    # keeping, so nothing happens; too high and it starts eating the shading
    # that gives a face its shape. 0.45 was picked by eye against a portrait.
    smooth_threshold: float = 0.45
    under_eye: float = 0.0       # optional extra lift under the eyes, 0..1
    min_face_frac: float = 0.004
    face_rel_to_largest: float = 0.15   # retouch guests too, not just the couple
    tiles: int = 2                      # tiled sweep so group shots get treated too
    quality: int = 95


def _shift_face(face: Face, dx: int, dy: int) -> Face:
    pts = face.landmarks.copy()
    pts[:, 0] -= dx
    pts[:, 1] -= dy
    x, y, w, h = face.bbox
    return dataclasses.replace(face, landmarks=pts, bbox=(x - dx, y - dy, w, h))


def _scale_face(face: Face, scale: float) -> Face:
    pts = face.landmarks * scale
    x, y, w, h = face.bbox
    box = (int(x * scale), int(y * scale), int(w * scale), int(h * scale))
    return dataclasses.replace(face, landmarks=pts, bbox=box)


def _roi_bounds(face: Face, shape, pad: float = 0.25) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    x, y, fw, fh = face.bbox
    px, py = int(fw * pad), int(fh * pad)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(w, x + fw + px), min(h, y + fh + py)
    return x0, y0, x1, y1


SIGMA_FINE = 1.2   # pore scale, in canonical ROI pixels


def _smooth_roi(roi: np.ndarray, mask: np.ndarray, cfg: RetouchConfig) -> np.ndarray:
    """Apply the three-band smoothing to one face ROI.

    The band maths runs on a copy resized to a canonical width - in both
    directions, so a 300px face and a 3000px face are filtered at the same
    scale relative to the face rather than to the sensor. Only the resulting
    low-frequency *change* is upsampled back, which keeps full-resolution work
    down to one resize and a few array ops, and leaves real pore detail
    untouched by any blur.
    """
    src = roi.astype(np.float32)
    h, w = roi.shape[:2]

    scale = WORK_WIDTH / float(max(w, 1))
    small = cv2.resize(roi, (WORK_WIDTH, max(8, round(h * scale))),
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

    # Edge-preserving base: tone and shading only. Measured to flatten skin
    # mid-frequencies about three times harder than a bilateral filter, at
    # constant cost regardless of radius.
    base = cv2.edgePreservingFilter(
        small, flags=cv2.RECURS_FILTER,
        sigma_s=float(cfg.smooth_radius), sigma_r=float(cfg.smooth_threshold),
    ).astype(np.float32)

    small_f = small.astype(np.float32)
    low_old = cv2.GaussianBlur(small_f, (0, 0), SIGMA_FINE)   # base + mid
    mid = low_old - base
    mid_out = np.where(mid < 0, mid * cfg.wrinkle_keep, mid * cfg.blotch_keep)

    # Everything below is the low/mid-frequency change, so it upsamples cleanly.
    delta = cv2.resize(base + mid_out - low_old, (w, h), interpolation=cv2.INTER_LINEAR)
    fine = src - cv2.resize(low_old, (w, h), interpolation=cv2.INTER_LINEAR)
    smoothed = src + delta + fine * (cfg.texture_keep - 1.0)

    alpha = np.clip(mask * cfg.strength, 0.0, 1.0)[:, :, None]
    return src * (1.0 - alpha) + smoothed * alpha


def _under_eye_lift(out: np.ndarray, face: Face, mask: np.ndarray,
                    amount: float) -> np.ndarray:
    """Gentle brightening of the tear-trough area."""
    if amount <= 0:
        return out
    h, w = out.shape[:2]
    region = np.zeros((h, w), np.uint8)
    # Landmark rings just below each eye.
    for idx in ([230, 229, 228, 31, 228, 229, 230, 120, 119, 118, 117],
                [450, 449, 448, 261, 448, 449, 450, 349, 348, 347, 346]):
        pts = face.landmarks[idx].astype(np.int32)
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 1] >= 0)]
        if len(pts) >= 3:
            cv2.fillConvexPoly(region, cv2.convexHull(pts), 255)
    grow = max(3, int(face.bbox[2] * 0.05)) | 1
    region = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    soft = cv2.GaussianBlur(region, (grow, grow), 0).astype(np.float32) / 255.0
    soft = np.clip(soft * mask * amount, 0.0, 1.0)[:, :, None]
    lifted = np.clip(out * 1.0 + 14.0, 0, 255)
    return out * (1.0 - soft) + lifted * soft


def retouch_image(bgr: np.ndarray, analyzer: FaceAnalyzer,
                  cfg: RetouchConfig) -> tuple[np.ndarray, int]:
    """Return the retouched image and the number of faces treated."""
    h, w = bgr.shape[:2]
    scale = min(1.0, DETECT_SIZE / float(max(h, w)))
    if scale < 1.0:
        small = cv2.resize(bgr, (round(w * scale), round(h * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = bgr

    faces = analyzer.detect(small, tiles=cfg.tiles)
    faces = primary_faces(faces, cfg.min_face_frac, cfg.face_rel_to_largest)
    if not faces:
        return bgr, 0

    out = bgr.astype(np.float32)
    inv = 1.0 / scale if scale < 1.0 else 1.0
    treated = 0
    for face in faces:
        full = _scale_face(face, inv) if inv != 1.0 else face
        x0, y0, x1, y1 = _roi_bounds(full, bgr.shape)
        if x1 - x0 < 24 or y1 - y0 < 24:
            continue
        local = _shift_face(full, x0, y0)
        roi = bgr[y0:y1, x0:x1]
        mask = skin_mask(roi.shape, local, roi)
        if mask.max() <= 0.01:
            continue
        result = _smooth_roi(roi, mask, cfg)
        result = _under_eye_lift(result, local, mask, cfg.under_eye)
        out[y0:y1, x0:x1] = result
        treated += 1

    return np.clip(out, 0, 255).astype(np.uint8), treated


def side_by_side(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Before/after strip for dialling in the strength."""
    gap = np.full((before.shape[0], 12, 3), 32, np.uint8)
    return np.hstack([before, gap, after])


_ANALYZER: Optional[FaceAnalyzer] = None


def _analyzer() -> FaceAnalyzer:
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = FaceAnalyzer(max_faces=8)
    return _ANALYZER


def retouch_one(args: tuple) -> dict:
    """Worker entry point: process one file, write the output."""
    path, out_dir, cfg_dict, suffix, preview, max_side = args
    cfg = RetouchConfig(**cfg_dict)
    rec = {"path": path, "filename": os.path.basename(path), "faces": 0, "error": ""}

    img = io.load_bgr(path, max_side=max_side)
    if img is None:
        rec["error"] = "unreadable"
        return rec
    try:
        result, treated = retouch_image(img, _analyzer(), cfg)
        rec["faces"] = treated
        stem, _ = os.path.splitext(os.path.basename(path))
        dest = os.path.join(out_dir, f"{stem}{suffix}.jpg")
        io.save_jpeg(dest, result, cfg.quality, io.read_exif_bytes(path))
        rec["output"] = dest
        if preview:
            prev_dir = os.path.join(out_dir, "_preview")
            io.save_jpeg(os.path.join(prev_dir, f"{stem}_ba.jpg"),
                         side_by_side(img, result), 88)
    except Exception as exc:
        rec["error"] = str(exc)
    return rec
