"""Join edit records to image files, build the dataset, train, and apply."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Optional

import numpy as np

from .. import imageio_utils as io
from ..faces import FaceAnalyzer
from . import features as F
from . import model as M
from . import xmp

_ANALYZER: Optional[FaceAnalyzer] = None


def _analyzer() -> FaceAnalyzer:
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = FaceAnalyzer(max_faces=6)
    return _ANALYZER


def _extract_worker(path: str) -> Optional[dict]:
    try:
        return F.extract_one(path, _analyzer())
    except Exception:
        return None


def build_index(roots: list[str]) -> dict[str, str]:
    """Map basename -> path, so catalog entries survive a moved archive.

    Lightroom stores absolute paths. Photographers move drives, rename volumes
    and work across machines, so a path that does not resolve is the normal
    case rather than an error; matching on filename recovers it.
    """
    index: dict[str, str] = {}
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith((".", "_"))]
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext in io.IMAGE_EXTS or ext in io.RAW_EXTS:
                    index.setdefault(name.lower(), os.path.join(dirpath, name))
    return index


def resolve_paths(records: list[dict], images_root: Optional[str]) -> tuple[list[dict], int]:
    """Attach a readable file to each edit record; drop those we cannot find."""
    index = build_index([images_root]) if images_root else {}
    resolved, missing = [], 0
    for record in records:
        path = record.get("path") or ""
        if path and os.path.exists(path):
            record["resolved"] = path
        else:
            candidate = index.get(os.path.basename(path).lower()) if index else None
            if not candidate:
                missing += 1
                continue
            record["resolved"] = candidate
        resolved.append(record)
    return resolved, missing


def shoot_key(record: dict) -> str:
    """Which wedding a frame belongs to - the unit validation splits on."""
    folder = record.get("folder") or os.path.dirname(record.get("resolved", ""))
    return folder or "?"


def extract_features(paths: list[str], jobs: int = 4,
                     progress: Optional[Callable[[int, int], None]] = None) -> list[Optional[dict]]:
    rows: list[Optional[dict]] = []
    if jobs <= 1:
        for i, path in enumerate(paths, 1):
            rows.append(_extract_worker(path))
            if progress:
                progress(i, len(paths))
        return rows
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for i, row in enumerate(pool.map(_extract_worker, paths, chunksize=2), 1):
            rows.append(row)
            if progress:
                progress(i, len(paths))
    return rows


def build_dataset(records: list[dict], jobs: int = 4,
                  progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """Features, targets and shoot labels, aligned row for row."""
    paths = [r["resolved"] for r in records]
    feature_rows = extract_features(paths, jobs=jobs, progress=progress)

    kept_rows, kept_records = [], []
    for row, record in zip(feature_rows, records):
        if row is not None:
            kept_rows.append(row)
            kept_records.append(record)
    if not kept_rows:
        raise RuntimeError("no images could be read - check --images-root")

    keys = [shoot_key(r) for r in kept_records]
    for row, key in zip(kept_rows, keys):
        row["_shoot"] = key          # leading underscore keeps it out of the matrix
    F.add_shoot_aggregates(kept_rows, lambda row: row["_shoot"])

    names = F.feature_names(kept_rows[0])
    matrix = F.to_matrix(kept_rows, names)

    targets: dict[str, np.ndarray] = {}
    present: dict[str, np.ndarray] = {}
    for param in xmp.ALL_PARAMS:
        values = np.zeros(len(kept_records))
        mask = np.zeros(len(kept_records), dtype=bool)
        for i, record in enumerate(kept_records):
            if param in record["settings"]:
                values[i] = record["settings"][param]
                mask[i] = True
        if mask.any():
            targets[param] = values
            present[param] = mask

    return {
        "matrix": matrix,
        "names": names,
        "targets": targets,
        "present": present,
        "groups": np.array(keys),
        "n_dropped": len(records) - len(kept_records),
    }


def predict_folder(folder: str, style: M.StyleModel, jobs: int = 4,
                   progress: Optional[Callable[[int, int], None]] = None) -> list[dict]:
    """Predict settings for every image in a new shoot."""
    paths = io.list_images(folder)
    paths += [p for p in _list_raw(folder)]
    paths = sorted(set(paths))
    if not paths:
        return []

    rows = extract_features(paths, jobs=jobs, progress=progress)
    usable = [(p, r) for p, r in zip(paths, rows) if r is not None]
    if not usable:
        return []

    feature_rows = [r for _, r in usable]
    # The whole folder is treated as one shoot, which is what it is.
    F.add_shoot_aggregates(feature_rows, lambda row: "new_shoot")
    matrix = F.to_matrix(feature_rows, style.feature_names)
    predictions = style.predict(matrix)

    out = []
    for i, (path, _) in enumerate(usable):
        out.append({
            "path": path,
            "settings": {name: float(values[i]) for name, values in predictions.items()},
        })
    return out


def _list_raw(folder: str) -> list[str]:
    found = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "_"))]
        for name in filenames:
            if os.path.splitext(name)[1].lower() in io.RAW_EXTS:
                found.append(os.path.join(dirpath, name))
    return found
