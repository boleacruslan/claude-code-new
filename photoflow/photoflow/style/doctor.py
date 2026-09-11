"""One command that reports whether this machine can run the style pipeline.

The two paths that cannot be tested without the photographer's own files -
decoding a NEF and reading a Lightroom catalog - are exactly the two that
break first. This prints a compact report designed to be copied straight back
into a conversation, so failures can be diagnosed without shipping gigabytes
of raw files anywhere.
"""

from __future__ import annotations

import os
import platform
import sys
import traceback

LINE = "-" * 64


def _versions() -> list[str]:
    out = [f"python      {sys.version.split()[0]} on {platform.system()} "
           f"{platform.machine()}"]
    for module, label in [("numpy", "numpy"), ("cv2", "opencv"),
                          ("mediapipe", "mediapipe"), ("rawpy", "rawpy"),
                          ("sklearn", "scikit-learn"), ("exifread", "exifread"),
                          ("PIL", "pillow")]:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "installed")
            out.append(f"{label:<14}{version}")
        except Exception as exc:
            out.append(f"{label:<14}MISSING ({exc.__class__.__name__})")
    return out


def _check_raw(path: str) -> list[str]:
    from . import features

    out = [LINE, f"RAW: {os.path.basename(path)}"]
    if not os.path.exists(path):
        return out + [f"  file not found: {path}"]
    out.append(f"  size {os.path.getsize(path) / 1e6:.1f} MB")
    try:
        image, meta = features.render_raw(path)
    except Exception:
        return out + ["  DECODE FAILED"] + ["  " + l for l in
                                            traceback.format_exc().splitlines()[-6:]]
    if image is None:
        return out + [f"  DECODE FAILED: {meta.get('error')}"]

    out.append(f"  decoded {image.shape[1]}x{image.shape[0]} (half-size render)")
    out.append(f"  as-shot white balance r/g={meta.get('wb_r', 0):.3f} "
               f"b/g={meta.get('wb_b', 0):.3f}")
    if not meta.get("wb_r"):
        out.append("  ! no white balance read - Temperature cannot be predicted")
    try:
        stats = features.image_stats(image)
        out.append(f"  median luma {stats['luma_p50']:.0f}, cast "
                   f"r/g={stats['cast_rg']:.3f} b/g={stats['cast_bg']:.3f}")
    except Exception as exc:
        out.append(f"  stats failed: {exc}")

    exif = features.exif_stats(path)
    out.append(f"  exif ISO {exif['iso']:.0f} f/{exif['aperture']:.1f} "
               f"{exif['focal']:.0f}mm hour {exif['hour']:.1f}")
    if exif["iso"] == 0:
        out.append("  ! EXIF not read - check exifread is installed")
    return out


def _check_catalog(path: str, limit: int) -> list[str]:
    from . import inspect as style_inspect
    from . import xmp

    out = [LINE, f"CATALOG: {os.path.basename(path)}"]
    if not os.path.exists(path):
        return out + [f"  file not found: {path}"]
    out.append(f"  size {os.path.getsize(path) / 1e6:.1f} MB")
    try:
        rows, variant = xmp.read_catalog(path, limit=limit)
    except Exception as exc:
        out.append("  READ FAILED")
        out += ["  " + l for l in str(exc).splitlines()[:12]]
        try:
            out.append("  tables actually present:")
            names = xmp.catalog_tables(path)
            for i in range(0, len(names), 4):
                out.append("    " + ", ".join(names[i:i + 4]))
                if i > 40:
                    out.append("    ...")
                    break
        except Exception as inner:
            out.append(f"  could not list tables: {inner}")
        return out

    out.append(f"  query variant that worked: {variant}")
    out.append(f"  records with parsed settings: {len(rows)}")
    if not rows:
        out.append("  ! settings blob parsed to nothing - the Lua format differs")
        return out

    summary = style_inspect.summarise(rows)
    out.append(f"  originals found on disk: {summary['files_on_disk']}")
    out.append(f"  shoots: {len(summary['shoots'])}")
    out.append(f"  parameters that vary: {len(summary['varying'])}")
    for entry in summary["varying"][:8]:
        out.append(f"    {entry['name']:<26}n={entry['count']:<6}"
                   f"p50={entry['p50']:>9.2f}  spread={entry['spread']:>8.2f}")
    if len(summary["varying"]) > 8:
        out.append(f"    ... and {len(summary['varying']) - 8} more")
    sample = rows[0]
    out.append(f"  sample path from catalog: {sample.get('path', '')[:70]}")
    return out


def run(raw: str | None, catalog: str | None, limit: int = 400) -> str:
    report = ["=" * 64, "photoflow style doctor", "=" * 64]
    report += _versions()
    if raw:
        report += _check_raw(raw)
    if catalog:
        report += _check_catalog(catalog, limit)
    if not raw and not catalog:
        report += [LINE,
                   "pass --raw <file.NEF> and --catalog <file.lrcat> to test "
                   "the two paths that actually break"]
    report.append("=" * 64)
    return "\n".join(report)
