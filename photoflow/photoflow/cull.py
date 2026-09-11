"""Two-pass culling: measure every frame, then rank within the shoot."""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Optional

import cv2
import numpy as np

from . import imageio_utils as io
from . import quality as q
from .faces import FaceAnalyzer, eye_region_box, primary_faces, skin_mask
from .grouping import group_bursts, mark_group_picks
from .scoring import (CullConfig, auto_blur_floor, decide, score_record,
                      sharpness_reference)

ANALYSIS_SIZE = 1600   # long side used for all measurements
THUMB_SIZE = 420

_ANALYZER: Optional[FaceAnalyzer] = None


def _analyzer(max_faces: int) -> FaceAnalyzer:
    """One FaceMesh per worker process, built on first use."""
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = FaceAnalyzer(max_faces=max_faces)
    return _ANALYZER


def _scale_box(box: tuple[int, int, int, int], factor: float) -> tuple[int, int, int, int]:
    """Map a box from the analysis copy back onto full-resolution pixels."""
    if factor == 1.0:
        return box
    x, y, w, h = box
    return (int(x * factor), int(y * factor), int(w * factor), int(h * factor))


def analyze_one(args: tuple) -> dict:
    """Measure a single frame. Runs inside a worker process."""
    path, cfg_dict, thumb_dir, max_faces = args
    cfg = CullConfig(**cfg_dict)
    rec: dict = {
        "path": path,
        "filename": os.path.basename(path),
        "capture_time": io.read_capture_time(path),
        "error": "",
    }

    # Full resolution is needed for focus: see quality.region_sharpness.
    # Detection, exposure balance and hashing all run on a downscaled copy.
    full = io.load_bgr(path)
    if full is None:
        rec.update(error="unreadable", verdict="error", score=0.0, reasons=["unreadable"],
                   hard_fail=True, group=-1, faces_total=0, faces_primary=0)
        return rec

    full_h, full_w = full.shape[:2]
    scale = min(1.0, ANALYSIS_SIZE / float(max(full_h, full_w)))
    if scale < 1.0:
        img = cv2.resize(full, (round(full_w * scale), round(full_h * scale)),
                         interpolation=cv2.INTER_AREA)
    else:
        img = full
    up = 1.0 / scale if scale < 1.0 else 1.0

    h, w = img.shape[:2]
    # Clipping is counted on the original pixels - downscaling averages blown
    # highlights away and would under-report them.
    expo = q.exposure_stats(full)
    rec.update(width=full_w, height=full_h,
               global_sharp=round(q.global_sharpness(img), 2),
               mean_luma=round(expo["mean_luma"], 1),
               clipped_high=round(expo["clipped_high"], 4),
               clipped_low=round(expo["clipped_low"], 4),
               contrast=round(expo["contrast"], 1))

    faces = _analyzer(max_faces).detect(img, tiles=cfg.tiles)
    mains = primary_faces(faces, cfg.min_face_frac, cfg.face_rel_to_largest)

    counts = {"open": 0, "closed": 0, "squint": 0, "profile": 0}
    face_sharp = 0.0
    face_luma = -1.0
    min_ear = -1.0
    for i, face in enumerate(mains):
        counts[face.eye_state(cfg.ear_closed, cfg.ear_open)] += 1
        eye_box = eye_region_box(face)
        sharp = q.region_sharpness(full, _scale_box(eye_box, up))
        if sharp <= 0.0:   # eyes cropped off or too small - fall back to the face
            sharp = q.region_sharpness(full, _scale_box(face.bbox, up))
        face_sharp = max(face_sharp, sharp)
        if i == 0:  # exposure is judged on the main subject only
            mask = skin_mask(img.shape, face, img)
            face_luma = q.masked_luma(img, (mask > 0.5).astype(np.uint8))
        min_ear = face.ear if min_ear < 0 else min(min_ear, face.ear)
    del full

    rec.update(faces_total=len(faces), faces_primary=len(mains),
               face_sharp=round(face_sharp, 2),
               face_luma=round(face_luma, 1),
               min_ear=round(min_ear, 3),
               eyes_open=counts["open"], eyes_closed=counts["closed"],
               eyes_squint=counts["squint"], eyes_profile=counts["profile"])
    rec["_hash"] = q.dhash(img)

    if thumb_dir:
        rec["thumb"] = _write_thumb(img, thumb_dir, rec["filename"])
    return rec


def _write_thumb(img: np.ndarray, thumb_dir: str, filename: str) -> str:
    os.makedirs(thumb_dir, exist_ok=True)
    h, w = img.shape[:2]
    scale = THUMB_SIZE / float(max(h, w))
    small = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                       interpolation=cv2.INTER_AREA)
    name = os.path.splitext(filename)[0] + ".jpg"
    out = os.path.join(thumb_dir, name)
    # Two source files can share a basename across subfolders.
    stem, ext = os.path.splitext(out)
    n = 1
    while os.path.exists(out):
        out = f"{stem}_{n}{ext}"
        n += 1
    cv2.imwrite(out, small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return os.path.basename(out)


def analyze_folder(paths: list[str], cfg: CullConfig, jobs: int = 4,
                   thumb_dir: Optional[str] = None, max_faces: int = 8,
                   progress: Optional[Callable[[int, int], None]] = None) -> list[dict]:
    """Pass 1 - measure every frame, in parallel."""
    args = [(p, cfg.as_dict(), thumb_dir, max_faces) for p in paths]
    records: list[dict] = []
    if jobs <= 1:
        for i, a in enumerate(args, 1):
            records.append(analyze_one(a))
            if progress:
                progress(i, len(args))
        return records

    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for i, rec in enumerate(pool.map(analyze_one, args, chunksize=4), 1):
            records.append(rec)
            if progress:
                progress(i, len(args))
    return records


def rank(records: list[dict], cfg: CullConfig) -> tuple[list[dict], tuple[float, float, float]]:
    """Pass 2 - score, group bursts, pick winners, decide verdicts."""
    good = [r for r in records if not r.get("error")]
    sharp_lo, sharp_hi, sharp_median = sharpness_reference(good)
    blur_floor = auto_blur_floor(cfg, sharp_median)
    for rec in good:
        score_record(rec, cfg, sharp_lo, sharp_hi, blur_floor)

    good.sort(key=lambda r: (r["capture_time"] is None,
                             r["capture_time"] or 0, r["filename"]))
    group_bursts(good, cfg.burst_gap_seconds, cfg.burst_hash_distance)
    mark_group_picks(good)
    for rec in good:
        decide(rec, cfg)

    for rec in records:
        if rec.get("error"):
            rec.setdefault("group", -1)
            rec.setdefault("group_size", 1)
            rec.setdefault("is_pick", False)
    return records, (sharp_lo, sharp_hi, blur_floor)


def apply_actions(records: list[dict], out_dir: str, action: str = "copy",
                  dry_run: bool = False) -> dict[str, int]:
    """Place each frame into keep/ maybe/ reject/ by verdict.

    Originals are never deleted: the worst case is a move into reject/, and
    the default action only copies.
    """
    tally: dict[str, int] = {}
    for rec in records:
        verdict = rec.get("verdict", "error")
        tally[verdict] = tally.get(verdict, 0) + 1
        if action == "none" or dry_run:
            continue
        dest_dir = os.path.join(out_dir, verdict)
        os.makedirs(dest_dir, exist_ok=True)
        dest = _unique(os.path.join(dest_dir, rec["filename"]))
        try:
            if action == "copy":
                shutil.copy2(rec["path"], dest)
            elif action == "move":
                shutil.move(rec["path"], dest)
            elif action == "link":
                os.symlink(os.path.abspath(rec["path"]), dest)
        except OSError as exc:
            rec["error"] = f"{action}_failed: {exc}"
    return tally


def _unique(path: str) -> str:
    stem, ext = os.path.splitext(path)
    n = 1
    while os.path.exists(path):
        path = f"{stem}_{n}{ext}"
        n += 1
    return path
