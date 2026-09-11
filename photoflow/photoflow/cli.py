"""Command line interface: photoflow cull | retouch | calibrate."""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from . import imageio_utils as io
from . import report as rp
from .cull import analyze_folder, apply_actions, rank
from .retouch import RetouchConfig, retouch_one
from .scoring import CullConfig


def _default_jobs() -> int:
    return max(1, min(8, (os.cpu_count() or 2)))


def _progress(prefix: str):
    start = time.time()

    def show(done: int, total: int) -> None:
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        sys.stderr.write(
            f"\r{prefix} {done}/{total}  {rate:4.1f} img/s  ETA {eta/60:4.1f} min   ")
        sys.stderr.flush()
        if done == total:
            sys.stderr.write("\n")

    return show


def _collect(src: str) -> list[str]:
    paths = io.list_images(src)
    raw = io.count_raw(src)
    if raw:
        print(f"note: {raw} raw file(s) ignored - photoflow reads JPEG/TIFF/PNG. "
              f"Export JPEGs (or in-camera JPEG) for culling.", file=sys.stderr)
    if not paths:
        print(f"no images found in {src}", file=sys.stderr)
    return paths


def cmd_cull(args: argparse.Namespace) -> int:
    paths = _collect(args.source)
    if not paths:
        return 1

    cfg = CullConfig(
        keep_score=args.keep_score, reject_score=args.reject_score,
        ear_closed=args.ear_closed, ear_open=args.ear_open,
        blur_floor=args.blur_floor, min_face_frac=args.min_face_frac,
        burst_gap_seconds=args.burst_gap, burst_hash_distance=args.burst_distance,
        duplicate_verdict=args.duplicates, tiles=args.tiles,
    )

    out_dir = args.output or os.path.join(args.source, "_cull")
    report_dir = os.path.join(out_dir, "_report")
    thumb_dir = None if args.no_thumbs else os.path.join(report_dir, "thumbs")

    print(f"analysing {len(paths)} image(s) with {args.jobs} worker(s)...")
    records = analyze_folder(paths, cfg, jobs=args.jobs, thumb_dir=thumb_dir,
                             progress=_progress("  measuring"))
    records, (sharp_lo, sharp_hi, blur_floor) = rank(records, cfg)

    tally = apply_actions(records, out_dir, action=args.action, dry_run=args.dry_run)

    summary = (f"{len(records)} frames - "
               + ", ".join(f"{k}: {v}" for k, v in sorted(tally.items()))
               + f" | sharpness band p20-p85: {sharp_lo:.1f}-{sharp_hi:.1f}"
               + f" | blur floor: {blur_floor:.1f}"
               + f" | action: {'dry-run' if args.dry_run else args.action}")
    print(summary)

    rp.write_csv(records, os.path.join(report_dir, "cull.csv"))
    rp.write_html(records, os.path.join(report_dir, "index.html"),
                  title=f"photoflow - {os.path.basename(os.path.abspath(args.source))}",
                  summary=summary)
    print(f"report: {os.path.join(report_dir, 'index.html')}")
    if args.calibrate:
        print("\n" + rp.print_calibration([r for r in records if not r.get('error')]))
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    paths = _collect(args.source)
    if not paths:
        return 1
    if args.limit and len(paths) > args.limit:
        step = len(paths) / args.limit
        paths = [paths[int(i * step)] for i in range(args.limit)]
    cfg = CullConfig()
    print(f"measuring {len(paths)} sample frame(s)...")
    records = analyze_folder(paths, cfg, jobs=args.jobs,
                             progress=_progress("  measuring"))
    records, (lo, hi, blur_floor) = rank(records, cfg)
    good = [r for r in records if not r.get("error")]
    print(rp.print_calibration(good))
    print(f"\nsharpness band for this gear/style: {lo:.1f} - {hi:.1f}")
    print(f"auto blur floor on this sample: {blur_floor:.1f} "
          f"(pass --blur-floor to pin it)")
    verdicts: dict[str, int] = {}
    for r in records:
        verdicts[r.get("verdict", "error")] = verdicts.get(r.get("verdict", "error"), 0) + 1
    print("verdicts with current defaults: "
          + ", ".join(f"{k}={v}" for k, v in sorted(verdicts.items())))
    return 0


def cmd_retouch(args: argparse.Namespace) -> int:
    paths = _collect(args.source)
    if not paths:
        return 1
    cfg = RetouchConfig(
        strength=args.strength, texture_keep=args.texture,
        blotch_keep=args.blotch, wrinkle_keep=args.wrinkle,
        smooth_radius=args.smooth_radius, smooth_threshold=args.smooth_threshold,
        under_eye=args.under_eye, quality=args.quality, tiles=args.tiles,
    )
    out_dir = args.output or os.path.join(args.source, "_retouched")
    os.makedirs(out_dir, exist_ok=True)
    work = [(p, out_dir, cfg.__dict__, args.suffix, args.preview, args.max_side)
            for p in paths]

    print(f"retouching {len(paths)} image(s) -> {out_dir}")
    show = _progress("  retouching")
    done = 0
    results = []
    if args.jobs <= 1:
        for item in work:
            results.append(retouch_one(item))
            done += 1
            show(done, len(work))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for rec in pool.map(retouch_one, work, chunksize=1):
                results.append(rec)
                done += 1
                show(done, len(work))

    touched = sum(1 for r in results if r["faces"] > 0)
    errors = [r for r in results if r["error"]]
    skipped = len(results) - touched - len(errors)
    print(f"done: {touched} retouched, {skipped} left unchanged (no face found), "
          f"{len(errors)} error(s)")
    for r in errors[:5]:
        print(f"  ! {r['filename']}: {r['error']}", file=sys.stderr)
    return 0


def cmd_style_inspect(args: argparse.Namespace) -> int:
    from .style import inspect as style_inspect
    from .style import xmp as style_xmp

    if not args.catalog and not args.folder:
        print("pass --catalog /path/x.lrcat or a folder of XMP sidecars",
              file=sys.stderr)
        return 1
    try:
        gathered = style_inspect.collect(catalog=args.catalog, folder=args.folder,
                                         limit=args.limit)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        if args.catalog:
            try:
                tables = style_xmp.catalog_tables(args.catalog)
                print("\ntables present in this catalog:", file=sys.stderr)
                print("  " + ", ".join(tables), file=sys.stderr)
            except Exception:
                pass
        return 1

    summary = style_inspect.summarise(gathered["rows"])
    print(style_inspect.render(summary, gathered["source"]))
    return 0


def cmd_style_learn(args: argparse.Namespace) -> int:
    from .style import inspect as style_inspect
    from .style import model as style_model
    from .style import pipeline

    try:
        gathered = style_inspect.collect(catalog=args.catalog, folder=args.folder,
                                         limit=args.limit)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    records, missing = pipeline.resolve_paths(gathered["rows"], args.images_root)
    print(f"{len(gathered['rows'])} edited frames from {gathered['source']}; "
          f"{len(records)} originals located"
          + (f", {missing} missing" if missing else ""))
    if not records:
        print("none of the originals could be found - pass --images-root "
              "pointing at the folder that holds your NEF files", file=sys.stderr)
        return 1

    shoots = len({pipeline.shoot_key(r) for r in records})
    if shoots < 3:
        print(f"only {shoots} shoot(s) - validation needs held-out weddings; "
              f"results below will be unreliable", file=sys.stderr)

    print(f"extracting features from {len(records)} frames "
          f"({shoots} shoots, {args.jobs} workers)...")
    data = pipeline.build_dataset(records, jobs=args.jobs,
                                  progress=_progress("  reading"))
    if data["n_dropped"]:
        print(f"  {data['n_dropped']} frame(s) unreadable, skipped")

    print("training...")
    style = style_model.train(
        data["matrix"], data["targets"], data["present"], data["groups"],
        data["names"])
    print()
    print(style_model.render_report(style))

    style_model.save(style, args.output)
    print(f"\nmodel saved: {args.output}")
    return 0


def cmd_style_apply(args: argparse.Namespace) -> int:
    from .style import model as style_model
    from .style import pipeline
    from .style import xmp as style_xmp

    style = style_model.load(args.model)
    print(f"predicting for {args.source} ...")
    results = pipeline.predict_folder(args.source, style, jobs=args.jobs,
                                      progress=_progress("  reading"))
    if not results:
        print("no readable images found", file=sys.stderr)
        return 1

    if args.dry_run:
        shown = [p for p in style.models] + list(style.constants)[:3]
        head = shown[:6]
        print("  " + "file".ljust(28) + "".join(f"{n[:12]:>14}" for n in head))
        for item in results[:20]:
            row = "  " + os.path.basename(item["path"])[:28].ljust(28)
            row += "".join(f"{item['settings'].get(n, 0):>14.2f}" for n in head)
            print(row)
        if len(results) > 20:
            print(f"  ... and {len(results) - 20} more")
        print(f"\n{len(results)} frame(s), nothing written (--dry-run)")
        return 0

    written, skipped = 0, 0
    for item in results:
        target = (os.path.join(args.output, os.path.splitext(
                      os.path.basename(item["path"]))[0] + ".xmp")
                  if args.output else style_xmp.sidecar_path(item["path"]))
        try:
            style_xmp.write_sidecar(target, item["settings"],
                                    overwrite=args.overwrite)
            written += 1
        except FileExistsError:
            skipped += 1

    print(f"wrote {written} sidecar(s)"
          + (f", skipped {skipped} that already had edits" if skipped else ""))
    print("in Lightroom: import these files, or select them and use "
          "Metadata > Read Metadata from File")
    return 0


def cmd_style_check_raw(args: argparse.Namespace) -> int:
    """Prove a NEF can be decoded here before trusting a whole training run."""
    from .style import features as style_features

    image, meta = style_features.render_raw(args.file)
    if image is None:
        print(f"could not decode {args.file}: {meta.get('error')}", file=sys.stderr)
        return 1
    print(f"decoded {os.path.basename(args.file)}: "
          f"{image.shape[1]}x{image.shape[0]} (half-size render)")
    print(f"as-shot white balance  r/g={meta.get('wb_r', 0):.3f} "
          f"b/g={meta.get('wb_b', 0):.3f}")
    stats = style_features.image_stats(image)
    print(f"median luma {stats['luma_p50']:.0f}, colour cast "
          f"r/g={stats['cast_rg']:.3f} b/g={stats['cast_bg']:.3f}")
    exif = style_features.exif_stats(args.file)
    print(f"exif: ISO {exif['iso']:.0f}, f/{exif['aperture']:.1f}, "
          f"{exif['focal']:.0f}mm, hour {exif['hour']:.1f}")
    if exif["iso"] == 0:
        print("  ! no EXIF read - check exifread is installed", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="photoflow",
        description="Batch culling and skin retouching for event photography.")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- cull ----
    c = sub.add_parser("cull", help="sort a shoot into keep / maybe / reject")
    c.add_argument("source", help="folder of JPEGs")
    c.add_argument("-o", "--output", help="output folder (default: <source>/_cull)")
    c.add_argument("--action", choices=["copy", "move", "link", "none"],
                   default="copy", help="how to place files (default: copy)")
    c.add_argument("--dry-run", action="store_true",
                   help="write the report but do not touch any file")
    c.add_argument("--keep-score", type=float, default=62.0)
    c.add_argument("--reject-score", type=float, default=38.0)
    c.add_argument("--blur-floor", type=float, default=0.0,
                   help="acutance below which a frame is rejected outright; "
                        "0 = auto, 35%% of this shoot's median")
    c.add_argument("--ear-closed", type=float, default=0.18,
                   help="eye aspect ratio below this counts as closed")
    c.add_argument("--ear-open", type=float, default=0.22)
    c.add_argument("--min-face-frac", type=float, default=0.004,
                   help="ignore faces smaller than this fraction of the frame")
    c.add_argument("--burst-gap", type=float, default=3.0,
                   help="seconds between frames to still count as one burst")
    c.add_argument("--burst-distance", type=int, default=12,
                   help="perceptual hash distance for near-duplicates (0-64)")
    c.add_argument("--duplicates", choices=["keep", "maybe", "reject"],
                   default="maybe", help="verdict for non-best frames in a burst")
    c.add_argument("--no-thumbs", action="store_true",
                   help="skip contact-sheet thumbnails (faster)")
    c.add_argument("--calibrate", action="store_true",
                   help="also print the metric percentile table")
    c.add_argument("--tiles", type=int, default=2,
                   help="tiled sweep grid that recovers small faces in group "
                        "and full-length shots; 0 disables it, 3 digs harder")
    c.add_argument("-j", "--jobs", type=int, default=_default_jobs())
    c.set_defaults(func=cmd_cull)

    # ---- calibrate ----
    cal = sub.add_parser("calibrate",
                         help="measure a sample and suggest thresholds")
    cal.add_argument("source")
    cal.add_argument("--limit", type=int, default=200,
                     help="sample at most N frames spread across the folder")
    cal.add_argument("-j", "--jobs", type=int, default=_default_jobs())
    cal.set_defaults(func=cmd_calibrate)

    # ---- retouch ----
    r = sub.add_parser("retouch", help="batch skin smoothing")
    r.add_argument("source", help="folder of JPEGs (or a single file)")
    r.add_argument("-o", "--output",
                   help="output folder (default: <source>/_retouched)")
    r.add_argument("-s", "--strength", type=float, default=0.6,
                   help="0 = off, 1 = full effect (default 0.6)")
    r.add_argument("--texture", type=float, default=0.90,
                   help="how much pore detail to keep (default 0.90)")
    r.add_argument("--blotch", type=float, default=0.55,
                   help="bright mid-frequency retention (default 0.55)")
    r.add_argument("--wrinkle", type=float, default=0.35,
                   help="dark mid-frequency retention - lower softens lines more")
    r.add_argument("--smooth-radius", type=float, default=60.0,
                   help="reach of the base layer (default 60)")
    r.add_argument("--smooth-threshold", type=float, default=0.45,
                   help="contrast the smoothing will not cross, 0.2-0.6; "
                        "raise for stronger wrinkle removal (default 0.45)")
    r.add_argument("--under-eye", type=float, default=0.0,
                   help="extra lift under the eyes, 0..1 (default off)")
    r.add_argument("--preview", action="store_true",
                   help="also write before/after strips to _preview/")
    r.add_argument("--suffix", default="_retouched")
    r.add_argument("--quality", type=int, default=95)
    r.add_argument("--max-side", type=int, default=0,
                   help="downscale output to this long side (0 = full size)")
    r.add_argument("--tiles", type=int, default=2,
                   help="tiled sweep grid that recovers small faces in group "
                        "and full-length shots; 0 disables it, 3 digs harder")
    r.add_argument("-j", "--jobs", type=int, default=_default_jobs())
    r.set_defaults(func=cmd_retouch)

    # ---- style ----
    st = sub.add_parser("style", help="learn and apply your own Lightroom edits")
    st_sub = st.add_subparsers(dest="style_command", required=True)

    si = st_sub.add_parser(
        "inspect",
        help="survey an archive of edits before training anything on it")
    si.add_argument("folder", nargs="?",
                    help="folder of images with XMP sidecars")
    si.add_argument("--catalog", help="path to a Lightroom .lrcat (read-only)")
    si.add_argument("--limit", type=int,
                    help="stop after this many records")
    si.set_defaults(func=cmd_style_inspect)

    sl = st_sub.add_parser("learn", help="train a model on your past edits")
    sl.add_argument("folder", nargs="?", help="folder of images with XMP sidecars")
    sl.add_argument("--catalog", help="path to a Lightroom .lrcat (read-only)")
    sl.add_argument("--images-root",
                    help="where the original files live, if the catalog paths "
                         "no longer resolve (moved drive, other machine)")
    sl.add_argument("-o", "--output", default="style.joblib")
    sl.add_argument("--limit", type=int, help="cap the number of frames used")
    sl.add_argument("-j", "--jobs", type=int, default=_default_jobs())
    sl.set_defaults(func=cmd_style_learn)

    sa = st_sub.add_parser("apply", help="predict settings for a new shoot")
    sa.add_argument("source", help="folder of the new shoot")
    sa.add_argument("--model", default="style.joblib")
    sa.add_argument("-o", "--output",
                    help="write sidecars here instead of next to the originals")
    sa.add_argument("--dry-run", action="store_true",
                    help="print predictions without writing anything")
    sa.add_argument("--overwrite", action="store_true",
                    help="replace existing sidecars (they hold your real edits)")
    sa.add_argument("-j", "--jobs", type=int, default=_default_jobs())
    sa.set_defaults(func=cmd_style_apply)

    sc = st_sub.add_parser("check-raw",
                           help="verify a single NEF decodes on this machine")
    sc.add_argument("file")
    sc.set_defaults(func=cmd_style_check_raw)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "max_side", None) == 0:
        args.max_side = None
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
