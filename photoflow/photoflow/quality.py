"""Pure measurement helpers: sharpness, exposure, contrast, perceptual hash.

Every metric is computed on a canonical resolution so values stay comparable
between a 45MP body and a downscaled preview.
"""

from __future__ import annotations

import cv2
import numpy as np

SHARP_CANON = 1024   # long side used for global sharpness
FACE_CANON = 256     # long side used for face-crop sharpness


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _fit(gray: np.ndarray, long_side: int) -> np.ndarray:
    h, w = gray.shape[:2]
    scale = long_side / float(max(h, w))
    if abs(scale - 1.0) < 0.02:
        return gray
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(gray, (max(8, round(w * scale)), max(8, round(h * scale))),
                      interpolation=interp)


def laplacian_var(gray: np.ndarray, long_side: int) -> float:
    """Variance of the Laplacian at a fixed scale - the classic blur measure."""
    g = _fit(gray, long_side)
    g = cv2.GaussianBlur(g, (3, 3), 0)  # kill sensor noise, which inflates the score
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def tenengrad(gray: np.ndarray, long_side: int) -> float:
    """Sobel gradient energy - second opinion on focus, robust to fine noise."""
    g = _fit(gray, long_side)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def global_sharpness(bgr: np.ndarray) -> float:
    return laplacian_var(to_gray(bgr), SHARP_CANON)


def region_sharpness(bgr: np.ndarray, box: tuple[int, int, int, int],
                     cap: int = 1400) -> float:
    """Focus acutance of a crop, measured at the source file's own pixel scale.

    Deliberately *not* normalised to a fixed crop size. Downscaling a region
    before measuring hides exactly the defect we are hunting: on a 45MP frame
    reduced to 1600px, a focus miss that is obvious at 1:1 smears into a
    couple of pixels and reads as sharp. Keeping native pixels means a soft
    frame stays soft.

    The consequence is that values are comparable *within* a shoot (same body,
    same lens) rather than across cameras - which is exactly how the caller
    uses them, since scoring normalises against the shoot's own percentiles.

    ``cap`` only bounds compute on an enormous close-up crop.
    """
    x, y, w, h = box
    h_img, w_img = bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w_img, x + w), min(h_img, y + h)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return 0.0
    crop = to_gray(bgr[y0:y1, x0:x1])
    longest = max(crop.shape[:2])
    if longest > cap:
        crop = _fit(crop, cap)
    crop = cv2.GaussianBlur(crop, (3, 3), 0)
    return float(cv2.Laplacian(crop, cv2.CV_64F).var())


def exposure_stats(bgr: np.ndarray) -> dict:
    """Clipping, brightness and contrast over the whole frame."""
    gray = to_gray(bgr)
    total = gray.size
    clipped_high = float(np.count_nonzero(gray >= 250) / total)
    clipped_low = float(np.count_nonzero(gray <= 6) / total)
    p1, p50, p99 = np.percentile(gray, [1, 50, 99])
    return {
        "mean_luma": float(gray.mean()),
        "median_luma": float(p50),
        "clipped_high": clipped_high,
        "clipped_low": clipped_low,
        "contrast": float(p99 - p1),
        "std_luma": float(gray.std()),
    }


def masked_luma(bgr: np.ndarray, mask: np.ndarray) -> float:
    """Median luminance inside a binary mask; -1 when the mask is empty."""
    gray = to_gray(bgr)
    sel = gray[mask > 0]
    if sel.size < 32:
        return -1.0
    return float(np.median(sel))


def dhash(bgr: np.ndarray, size: int = 8) -> np.ndarray:
    """64-bit difference hash, packed into 8 bytes."""
    gray = to_gray(bgr)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    return np.packbits(diff.flatten())


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.unpackbits(np.bitwise_xor(a, b)).sum())
