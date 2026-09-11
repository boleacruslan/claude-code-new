"""Describe a frame as it was BEFORE editing, so the edit can be predicted.

Everything here answers one question: what would a photographer look at when
deciding where to put the sliders? Brightness and where it sits in the
histogram, the colour cast, whether there is a face and how the skin is lit,
the shooting parameters, and the character of the shoot as a whole.

For RAW the render is deliberately fixed and dumb - no auto brightness, camera
white balance, nothing adaptive. An adaptive render would silently correct the
very differences the model is supposed to learn.
"""

from __future__ import annotations

import datetime as _dt
import math
import os
from typing import Optional

import cv2
import numpy as np

from .. import imageio_utils as io
from ..faces import FaceAnalyzer, primary_faces, skin_mask
from ..quality import to_gray

FEATURE_SIZE = 1024      # long side for statistics
FACE_DETECT_SIZE = 1280


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def render_raw(path: str, half_size: bool = True) -> tuple[Optional[np.ndarray], dict]:
    """Render a raw file neutrally and report its as-shot white balance.

    ``no_auto_bright`` matters: LibRaw would otherwise stretch each frame's
    histogram to taste, which erases the exposure differences that the
    Exposure2012 target is supposed to explain.

    The camera's white-balance multipliers come back as features because
    Lightroom's Temperature target is in Kelvin measured against exactly them -
    without the as-shot reference, absolute Kelvin is not predictable.
    """
    try:
        import rawpy
    except ImportError as exc:
        raise RuntimeError(
            "reading NEF needs rawpy: pip install rawpy. Alternatively export "
            "JPEGs from Lightroom with all develop settings zeroed and point "
            "photoflow at those."
        ) from exc

    try:
        with rawpy.imread(path) as raw:
            wb = list(raw.camera_whitebalance or [])
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                half_size=half_size,
                output_bps=8,
                gamma=(2.222, 4.5),
            )
    except Exception as exc:                      # LibRaw raises many types
        return None, {"error": str(exc)}

    meta = {}
    if len(wb) >= 3 and wb[1]:
        meta["wb_r"] = float(wb[0]) / float(wb[1])
        meta["wb_b"] = float(wb[2]) / float(wb[1])
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), meta


def load_for_features(path: str) -> tuple[Optional[np.ndarray], dict]:
    """Load any supported file into a neutral BGR image."""
    if os.path.splitext(path)[1].lower() in io.RAW_EXTS:
        return render_raw(path)
    image = io.load_bgr(path, max_side=2048)
    return image, {}


# --------------------------------------------------------------------------
# Feature blocks
# --------------------------------------------------------------------------

def image_stats(bgr: np.ndarray) -> dict:
    """Histogram shape and colour cast - the bulk of what drives a tone edit."""
    h, w = bgr.shape[:2]
    scale = FEATURE_SIZE / float(max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (round(w * scale), round(h * scale)),
                         interpolation=cv2.INTER_AREA)

    gray = to_gray(bgr)
    p = np.percentile(gray, [1, 5, 25, 50, 75, 95, 99])
    blue, green, red = (bgr[:, :, i].astype(np.float32) for i in range(3))
    green_mean = float(green.mean()) or 1.0

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    out = {
        "luma_p1": p[0], "luma_p5": p[1], "luma_p25": p[2], "luma_p50": p[3],
        "luma_p75": p[4], "luma_p95": p[5], "luma_p99": p[6],
        "luma_mean": float(gray.mean()), "luma_std": float(gray.std()),
        "contrast": float(p[6] - p[0]),
        "clip_high": float(np.count_nonzero(gray >= 250) / gray.size),
        "clip_low": float(np.count_nonzero(gray <= 5) / gray.size),
        "dark_mass": float(np.count_nonzero(gray < 64) / gray.size),
        "light_mass": float(np.count_nonzero(gray > 192) / gray.size),
        # Grey-world cast: what a photographer reads as "too warm / too green".
        "cast_rg": float(red.mean()) / green_mean,
        "cast_bg": float(blue.mean()) / green_mean,
        "red_p95": float(np.percentile(red, 95)),
        "blue_p95": float(np.percentile(blue, 95)),
        "sat_mean": float(hsv[:, :, 1].mean()),
        "sat_p90": float(np.percentile(hsv[:, :, 1], 90)),
    }
    return {k: float(v) for k, v in out.items()}


def face_stats(bgr: np.ndarray, analyzer: FaceAnalyzer) -> dict:
    """How the subject's skin is lit - photographers expose for skin, not for
    the average of the frame, so this often carries the exposure decision."""
    h, w = bgr.shape[:2]
    scale = min(1.0, FACE_DETECT_SIZE / float(max(h, w)))
    small = (cv2.resize(bgr, (round(w * scale), round(h * scale)),
                        interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr)

    empty = {"has_face": 0.0, "n_faces": 0.0, "face_area": 0.0,
             "skin_luma": -1.0, "skin_cr": -1.0, "skin_cb": -1.0,
             "skin_vs_frame": 0.0}
    faces = analyzer.detect(small)
    mains = primary_faces(faces)
    if not mains:
        return empty

    face = mains[0]
    mask = skin_mask(small.shape, face, small) > 0.5
    if mask.sum() < 64:
        empty.update(has_face=1.0, n_faces=float(len(mains)),
                     face_area=float(face.area_frac))
        return empty

    ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    frame_luma = float(to_gray(small).mean()) or 1.0
    skin_luma = float(np.median(ycrcb[:, :, 0][mask]))
    return {
        "has_face": 1.0,
        "n_faces": float(len(mains)),
        "face_area": float(face.area_frac),
        "skin_luma": skin_luma,
        "skin_cr": float(np.median(ycrcb[:, :, 1][mask])),
        "skin_cb": float(np.median(ycrcb[:, :, 2][mask])),
        "skin_vs_frame": skin_luma / frame_luma,
    }


_CAMERA_VOCAB: dict[str, int] = {}
_LENS_VOCAB: dict[str, int] = {}


def _code(vocab: dict[str, int], value: str, grow: bool) -> float:
    if not value:
        return -1.0
    if value not in vocab:
        if not grow:
            return -1.0
        vocab[value] = len(vocab)
    return float(vocab[value])


def exif_stats(path: str, grow_vocab: bool = True) -> dict:
    """Shooting parameters. Read with exifread so NEF works, not just JPEG."""
    out = {"iso": 0.0, "aperture": 0.0, "shutter_log": 0.0, "focal": 0.0,
           "exposure_bias": 0.0, "flash": 0.0, "hour": -1.0,
           "camera": -1.0, "lens": -1.0}
    try:
        import exifread
        with open(path, "rb") as fh:
            tags = exifread.process_file(fh, details=False)
    except Exception:
        return out

    def number(key: str) -> Optional[float]:
        tag = tags.get(key)
        if tag is None:
            return None
        try:
            values = tag.values
            value = values[0] if isinstance(values, list) else values
            if hasattr(value, "num"):
                return float(value.num) / float(value.den or 1)
            return float(value)
        except Exception:
            return None

    out["iso"] = number("EXIF ISOSpeedRatings") or 0.0
    out["aperture"] = number("EXIF FNumber") or 0.0
    shutter = number("EXIF ExposureTime")
    # Shutter spans 1/8000..30s; the log makes that range usable by a model.
    out["shutter_log"] = float(math.log2(shutter)) if shutter and shutter > 0 else 0.0
    out["focal"] = number("EXIF FocalLength") or 0.0
    out["exposure_bias"] = number("EXIF ExposureBiasValue") or 0.0
    flash = tags.get("EXIF Flash")
    out["flash"] = 1.0 if flash and "fired" in str(flash).lower() \
        and "not" not in str(flash).lower() else 0.0

    stamp = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
    if stamp:
        try:
            when = _dt.datetime.strptime(str(stamp).strip(), "%Y:%m:%d %H:%M:%S")
            out["hour"] = float(when.hour + when.minute / 60.0)
        except ValueError:
            pass

    out["camera"] = _code(_CAMERA_VOCAB, str(tags.get("Image Model", "")).strip(),
                          grow_vocab)
    out["lens"] = _code(_LENS_VOCAB, str(tags.get("EXIF LensModel", "")).strip(),
                        grow_vocab)
    return out


def extract_one(path: str, analyzer: FaceAnalyzer,
                grow_vocab: bool = True) -> Optional[dict]:
    """All per-frame features for one file, or None if it cannot be read."""
    image, meta = load_for_features(path)
    if image is None:
        return None
    row = {"_path": path}
    row.update(image_stats(image))
    row.update(face_stats(image, analyzer))
    row.update(exif_stats(path, grow_vocab))
    row["wb_r"] = float(meta.get("wb_r", 0.0))
    row["wb_b"] = float(meta.get("wb_b", 0.0))
    return row


# --------------------------------------------------------------------------
# Shoot-level context
# --------------------------------------------------------------------------

SHOOT_FEATURES = ["shoot_luma_p50", "shoot_luma_p10", "shoot_luma_p90",
                  "shoot_cast_rg", "shoot_cast_bg", "shoot_size",
                  "position_in_shoot"]


def add_shoot_aggregates(rows: list[dict], shoot_of) -> None:
    """Describe each frame's shoot, in place.

    A wedding is edited as a body of work: if the whole reception was dim and
    tungsten-lit, every frame in it moves the same way. Unlike a per-shoot
    correction term, these aggregates are computable on a brand new shoot,
    where no answers exist yet - which is the only reason they are usable at
    prediction time.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(shoot_of(row), []).append(row)

    for members in groups.values():
        luma = np.array([m["luma_p50"] for m in members], dtype=float)
        rg = float(np.median([m["cast_rg"] for m in members]))
        bg = float(np.median([m["cast_bg"] for m in members]))
        p10, p50, p90 = np.percentile(luma, [10, 50, 90])
        ordered = sorted(members, key=lambda m: (m.get("hour", -1), m["_path"]))
        for index, row in enumerate(ordered):
            row["shoot_luma_p50"] = float(p50)
            row["shoot_luma_p10"] = float(p10)
            row["shoot_luma_p90"] = float(p90)
            row["shoot_cast_rg"] = rg
            row["shoot_cast_bg"] = bg
            row["shoot_size"] = float(len(members))
            row["position_in_shoot"] = index / max(len(members) - 1, 1)


def feature_names(sample: dict) -> list[str]:
    return sorted(k for k in sample if not k.startswith("_"))


def to_matrix(rows: list[dict], names: list[str]) -> np.ndarray:
    return np.array([[float(row.get(n, 0.0)) for n in names] for row in rows],
                    dtype=np.float64)
