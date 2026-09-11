"""Read and write Lightroom develop settings.

Two sources are supported:

* **XMP sidecars** - what Lightroom writes with *Metadata > Save Metadata to
  File*. Plain XML, documented namespace, safe to parse and to write back.
  This is the path to prefer in both directions.
* **A ``.lrcat`` catalog** - SQLite. Used read-only and best-effort, for
  photographers who never saved sidecars. The settings live in a Lua table
  serialised into a text column, so values are pulled out by pattern rather
  than by a real Lua parse.

Predictions are written as sidecars, never baked into pixels: every value
lands on a Lightroom slider the photographer can still move.
"""

from __future__ import annotations

import os
import re
import sqlite3
import xml.etree.ElementTree as ET
from typing import Iterator, Optional

CRS_NS = "http://ns.adobe.com/camera-raw-settings/1.0/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
XMP_EXT = ".xmp"

# The develop parameters worth learning. Local adjustments (brushes, gradients,
# AI masks) are deliberately absent: they are per-photo artistic decisions with
# no stable mapping from image statistics, and faking them would be worse than
# leaving them to the photographer.
TONE_PARAMS = [
    "Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012",
    "Whites2012", "Blacks2012",
]
PRESENCE_PARAMS = ["Texture", "Clarity2012", "Dehaze", "Vibrance", "Saturation"]
WB_PARAMS = ["Temperature", "Tint"]
CURVE_PARAMS = ["ParametricShadows", "ParametricDarks", "ParametricLights",
                "ParametricHighlights"]
DETAIL_PARAMS = ["Sharpness", "LuminanceSmoothing", "ColorNoiseReduction",
                 "PostCropVignetteAmount", "GrainAmount"]

HSL_COLORS = ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple",
              "Magenta"]
HSL_PARAMS = [f"{kind}Adjustment{color}"
              for kind in ("Hue", "Saturation", "Luminance")
              for color in HSL_COLORS]

ALL_PARAMS = (TONE_PARAMS + PRESENCE_PARAMS + WB_PARAMS + CURVE_PARAMS
              + DETAIL_PARAMS + HSL_PARAMS)

# Ranges used to sanity-check parsed values and to clamp predictions.
PARAM_RANGE = {p: (-5.0, 5.0) for p in ["Exposure2012"]}
PARAM_RANGE.update({p: (-100.0, 100.0) for p in
                    TONE_PARAMS[1:] + PRESENCE_PARAMS + CURVE_PARAMS + HSL_PARAMS})
PARAM_RANGE.update({"Temperature": (2000.0, 50000.0), "Tint": (-150.0, 150.0)})
PARAM_RANGE.update({p: (0.0, 150.0) for p in DETAIL_PARAMS})
PARAM_RANGE["PostCropVignetteAmount"] = (-100.0, 100.0)


def sidecar_path(image_path: str) -> str:
    return os.path.splitext(image_path)[0] + XMP_EXT


def _to_float(text: str) -> Optional[float]:
    if text is None:
        return None
    text = text.strip().lstrip("+")
    try:
        return float(text)
    except ValueError:
        return None


def read_sidecar(path: str) -> dict[str, float]:
    """Parse develop settings out of one XMP sidecar.

    Lightroom writes settings either as attributes on rdf:Description or as
    child elements, depending on version and on what wrote the file, so both
    forms are collected.
    """
    try:
        tree = ET.parse(path)
    except (ET.ParseError, OSError):
        return {}

    wanted = set(ALL_PARAMS)
    found: dict[str, float] = {}
    crs_prefix = f"{{{CRS_NS}}}"
    for element in tree.iter():
        for key, value in element.attrib.items():
            if not key.startswith(crs_prefix):
                continue
            name = key[len(crs_prefix):]
            if name in wanted:
                number = _to_float(value)
                if number is not None:
                    found.setdefault(name, number)
        tag = element.tag
        if tag.startswith(f"{{{CRS_NS}}}"):
            name = tag.split("}")[-1]
            if name in wanted and element.text:
                number = _to_float(element.text)
                if number is not None:
                    found.setdefault(name, number)
    return {k: v for k, v in found.items() if _in_range(k, v)}


def _in_range(name: str, value: float) -> bool:
    lo, hi = PARAM_RANGE.get(name, (-1e6, 1e6))
    return lo <= value <= hi


def find_pairs(root: str) -> Iterator[tuple[str, str]]:
    """Yield (image_path, sidecar_path) for every image that has a sidecar."""
    from ..imageio_utils import IMAGE_EXTS, RAW_EXTS

    known = IMAGE_EXTS | RAW_EXTS
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "_"))]
        sidecars = {os.path.splitext(f)[0] for f in filenames
                    if f.lower().endswith(XMP_EXT)}
        if not sidecars:
            continue
        for name in filenames:
            stem, ext = os.path.splitext(name)
            if ext.lower() in known and stem in sidecars:
                yield (os.path.join(dirpath, name),
                       os.path.join(dirpath, stem + XMP_EXT))


# --------------------------------------------------------------------------
# Catalog reading
# --------------------------------------------------------------------------

_LUA_PAIR = re.compile(r'([A-Za-z0-9_]+)\s*=\s*(-?[\d.]+|true|false)')


def catalog_tables(lrcat_path: str) -> list[str]:
    """Table names in a catalog - diagnostics for when the schema surprises us."""
    uri = f"file:{os.path.abspath(lrcat_path)}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        return [r[0] for r in rows]
    finally:
        conn.close()


# Adobe does not document the catalog schema, and it shifts between Lightroom
# versions, so the join is attempted in descending order of richness and the
# first one that runs is used. Whatever succeeds, the develop settings come out
# of the same serialised Lua blob.
_CATALOG_QUERIES = [
    ("full", """
        SELECT rf.absolutePath, fo.pathFromRoot, f.baseName, f.extension,
               i.captureTime, ds.text
        FROM Adobe_imageDevelopSettings ds
        JOIN Adobe_images i         ON i.id_local = ds.image
        JOIN AgLibraryFile f        ON f.id_local = i.rootFile
        JOIN AgLibraryFolder fo     ON fo.id_local = f.folder
        JOIN AgLibraryRootFolder rf ON rf.id_local = fo.rootFolder
        WHERE ds.text IS NOT NULL
    """),
    ("no_root", """
        SELECT '', fo.pathFromRoot, f.baseName, f.extension,
               i.captureTime, ds.text
        FROM Adobe_imageDevelopSettings ds
        JOIN Adobe_images i     ON i.id_local = ds.image
        JOIN AgLibraryFile f    ON f.id_local = i.rootFile
        JOIN AgLibraryFolder fo ON fo.id_local = f.folder
        WHERE ds.text IS NOT NULL
    """),
    ("bare", """
        SELECT '', '', f.baseName, f.extension, i.captureTime, ds.text
        FROM Adobe_imageDevelopSettings ds
        JOIN Adobe_images i  ON i.id_local = ds.image
        JOIN AgLibraryFile f ON f.id_local = i.rootFile
        WHERE ds.text IS NOT NULL
    """),
    ("settings_only", """
        SELECT '', '', '', '', NULL, ds.text
        FROM Adobe_imageDevelopSettings ds
        WHERE ds.text IS NOT NULL
    """),
]


def read_catalog(lrcat_path: str, limit: Optional[int] = None) -> tuple[list[dict], str]:
    """Pull develop settings out of a Lightroom catalog.

    Opened read-only through a URI, so an open Lightroom is never disturbed and
    the catalog cannot be modified. Values come from a regex sweep of the
    serialised Lua settings blob - best effort, and only values landing inside
    each parameter's legal range are kept.

    Returns the rows and the name of the query variant that worked, so the
    caller can tell the user how much of the schema actually matched.
    """
    uri = f"file:{os.path.abspath(lrcat_path)}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise RuntimeError(f"cannot open catalog: {exc}") from exc

    errors = []
    try:
        for variant, query in _CATALOG_QUERIES:
            sql = query + (f" LIMIT {int(limit)}" if limit else "")
            try:
                cursor = conn.execute(sql)
            except sqlite3.Error as exc:
                errors.append(f"{variant}: {exc}")
                continue

            rows: list[dict] = []
            for root, sub, base, ext, capture, text in cursor:
                settings = _parse_lua_settings(text)
                if not settings:
                    continue
                name = f"{base}.{ext}" if base and ext else (base or "")
                rows.append({
                    "path": os.path.join(root or "", sub or "", name),
                    "folder": os.path.join(root or "", sub or ""),
                    "capture_time": capture,
                    "settings": settings,
                })
            return rows, variant
    finally:
        conn.close()

    raise RuntimeError(
        "catalog schema not as expected, every query variant failed:\n  "
        + "\n  ".join(errors)
        + "\n\nFall back to sidecars: in Lightroom select the photos and use "
          "Metadata > Save Metadata to File, then point photoflow at the folder."
    )


def _parse_lua_settings(text: str) -> dict[str, float]:
    wanted = set(ALL_PARAMS)
    out: dict[str, float] = {}
    for name, raw in _LUA_PAIR.findall(text or ""):
        if name not in wanted or name in out:
            continue
        if raw in ("true", "false"):
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if _in_range(name, value):
            out[name] = value
    return out


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

_SIDECAR_TEMPLATE = """<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="{rdf}">
  <rdf:Description rdf:about=""
    xmlns:crs="{crs}"
    crs:Version="15.0"
    crs:ProcessVersion="11.0"
    crs:HasSettings="True"
{attributes}/>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
"""


def format_value(name: str, value: float) -> str:
    if name == "Exposure2012":
        return f"{value:+.2f}"
    if name in ("Temperature",):
        return f"{int(round(value))}"
    return f"{int(round(value)):+d}" if value else "0"


def write_sidecar(path: str, settings: dict[str, float],
                  overwrite: bool = False) -> str:
    """Write predicted settings as an XMP sidecar.

    Refuses to clobber an existing sidecar unless asked: that file holds the
    photographer's real edit, and it is the training data.
    """
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(
            f"{path} already exists - refusing to overwrite an existing edit")
    lines = []
    for name in ALL_PARAMS:
        if name in settings:
            lines.append(f'    crs:{name}="{format_value(name, settings[name])}"')
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_SIDECAR_TEMPLATE.format(rdf=RDF_NS, crs=CRS_NS,
                                          attributes="\n".join(lines)))
    return path
