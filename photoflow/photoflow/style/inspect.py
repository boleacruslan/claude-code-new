"""Survey a Lightroom archive before trying to learn anything from it.

This runs first and on purpose. The catalog schema is undocumented and the
develop settings sit inside a serialised Lua blob, so the reader is an
educated guess until it has met a real catalog. Training a model on data that
was parsed wrong would produce confident nonsense, so this command answers a
narrow question: did we actually read the edits, and which of them vary?
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Optional

import numpy as np

from . import xmp

# A parameter whose spread is below this is effectively a constant: the
# photographer set it once (or never touched it), so there is nothing to learn.
VARIATION_EPS = {
    "Exposure2012": 0.05,
    "Temperature": 60.0,
    "Tint": 1.0,
}
DEFAULT_EPS = 0.5


def collect(catalog: Optional[str] = None, folder: Optional[str] = None,
            limit: Optional[int] = None) -> dict:
    """Gather edit records from a catalog or from XMP sidecars."""
    if catalog:
        rows, variant = xmp.read_catalog(catalog, limit=limit)
        source = f"catalog ({variant} query)"
    elif folder:
        rows = []
        for image_path, sidecar in xmp.find_pairs(folder):
            settings = xmp.read_sidecar(sidecar)
            if settings:
                rows.append({
                    "path": image_path,
                    "folder": os.path.dirname(image_path),
                    "capture_time": None,
                    "settings": settings,
                })
            if limit and len(rows) >= limit:
                break
        source = "xmp sidecars"
    else:
        raise ValueError("need either a catalog or a folder")
    return {"rows": rows, "source": source}


def summarise(rows: list[dict]) -> dict:
    """Per-parameter spread, shoot breakdown, and how many files we can find."""
    values: dict[str, list[float]] = {p: [] for p in xmp.ALL_PARAMS}
    for row in rows:
        for name, value in row["settings"].items():
            values[name].append(value)

    varying, constant, absent = [], [], []
    for name in xmp.ALL_PARAMS:
        series = values[name]
        if not series:
            absent.append(name)
            continue
        array = np.asarray(series, dtype=float)
        spread = float(np.percentile(array, 90) - np.percentile(array, 10))
        entry = {
            "name": name,
            "count": len(series),
            "coverage": len(series) / max(len(rows), 1),
            "p5": float(np.percentile(array, 5)),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "spread": spread,
        }
        if spread >= VARIATION_EPS.get(name, DEFAULT_EPS):
            varying.append(entry)
        else:
            constant.append(entry)

    shoots = Counter(row["folder"] or "?" for row in rows)
    found = sum(1 for row in rows if row["path"] and os.path.exists(row["path"]))
    return {
        "total": len(rows),
        "files_on_disk": found,
        "varying": sorted(varying, key=lambda e: -e["spread"]),
        "constant": constant,
        "absent": absent,
        "shoots": shoots,
    }


def render(summary: dict, source: str) -> str:
    """Format the survey for the terminal."""
    out = [f"source: {source}",
           f"edited frames found: {summary['total']}",
           f"originals present on disk: {summary['files_on_disk']}"]

    if summary["total"] and not summary["files_on_disk"]:
        out.append("  ! paths from the catalog do not resolve on this machine - "
                   "the archive may live on another drive, or be mounted elsewhere")

    shoots = summary["shoots"]
    out.append(f"\nshoots (folders): {len(shoots)}")
    for folder, count in shoots.most_common(12):
        out.append(f"   {count:6d}  {folder[-70:]}")
    if len(shoots) > 12:
        out.append(f"   ... and {len(shoots) - 12} more")
    if len(shoots) < 3:
        out.append("  ! fewer than 3 shoots: honest validation needs held-out "
                   "weddings, so gather more catalogs before training")

    out.append(f"\nparameters that vary ({len(summary['varying'])}) - "
               f"these are what a model can learn:")
    out.append(f"   {'parameter':<28}{'n':>7}{'p5':>10}{'p50':>10}{'p95':>10}{'spread':>10}")
    for entry in summary["varying"]:
        out.append(f"   {entry['name']:<28}{entry['count']:>7}"
                   f"{entry['p5']:>10.2f}{entry['p50']:>10.2f}"
                   f"{entry['p95']:>10.2f}{entry['spread']:>10.2f}")

    if summary["constant"]:
        names = ", ".join(e["name"] for e in summary["constant"][:12])
        out.append(f"\neffectively constant ({len(summary['constant'])}): {names}"
                   + (" ..." if len(summary["constant"]) > 12 else ""))
        out.append("   these get copied through as fixed values, not predicted")
    if summary["absent"]:
        out.append(f"\nnever present ({len(summary['absent'])}) - ignored")

    if not summary["varying"]:
        out.append("\n! nothing varies. Either the parse failed or the archive "
                   "is unedited - do not train on this.")
    return "\n".join(out)
