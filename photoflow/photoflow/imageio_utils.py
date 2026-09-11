"""Image discovery, loading, EXIF handling and saving."""

from __future__ import annotations

import datetime as _dt
import os
from typing import Iterator, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}

# Nikon/Canon raw files are listed so we can warn instead of silently skipping.
RAW_EXTS = {".nef", ".cr2", ".cr3", ".arw", ".raf", ".orf", ".rw2", ".dng"}


def list_images(root: str, recursive: bool = True) -> list[str]:
    """Return sorted image paths under *root*."""
    out: list[str] = []
    if os.path.isfile(root):
        return [root]
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip our own output folders so re-runs stay idempotent.
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "_"))]
        for name in filenames:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                out.append(os.path.join(dirpath, name))
        if not recursive:
            break
    return sorted(out)


def count_raw(root: str, recursive: bool = True) -> int:
    """Count raw files so the CLI can tell the user they were ignored."""
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "_"))]
        n += sum(1 for f in filenames if os.path.splitext(f)[1].lower() in RAW_EXTS)
        if not recursive:
            break
    return n


def load_bgr(path: str, max_side: Optional[int] = None) -> Optional[np.ndarray]:
    """Load an image as BGR uint8, honouring the EXIF orientation flag.

    Returns None when the file cannot be decoded, so a corrupt frame never
    aborts a batch run.
    """
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            if max_side:
                w, h = im.size
                scale = max_side / float(max(w, h))
                if scale < 1.0:
                    im = im.resize(
                        (max(1, round(w * scale)), max(1, round(h * scale))),
                        Image.BILINEAR,
                    )
            rgb = np.asarray(im)
    except Exception:
        return None
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def read_exif_bytes(path: str) -> Optional[bytes]:
    """Raw EXIF blob, for copying metadata onto a processed output file."""
    try:
        with Image.open(path) as im:
            return im.info.get("exif")
    except Exception:
        return None


_EXIF_DATETIME_TAGS = (36867, 36868, 306)  # DateTimeOriginal, DateTimeDigitized, DateTime


def read_capture_time(path: str) -> Optional[_dt.datetime]:
    """Capture time from EXIF, falling back to file mtime."""
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            for tag in _EXIF_DATETIME_TAGS:
                value = exif.get(tag)
                if value:
                    try:
                        return _dt.datetime.strptime(str(value).strip(), "%Y:%m:%d %H:%M:%S")
                    except ValueError:
                        continue
    except Exception:
        pass
    try:
        return _dt.datetime.fromtimestamp(os.path.getmtime(path))
    except OSError:
        return None


def save_jpeg(path: str, bgr: np.ndarray, quality: int = 95,
              exif: Optional[bytes] = None) -> None:
    """Write BGR image as JPEG, carrying EXIF across when available."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    kwargs = {"quality": int(quality), "subsampling": 0, "optimize": True}
    if exif:
        kwargs["exif"] = exif
    im.save(path, "JPEG", **kwargs)
