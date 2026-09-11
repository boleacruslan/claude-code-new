"""CSV export and a browsable HTML contact sheet."""

from __future__ import annotations

import csv
import html
import os
from typing import Sequence

CSV_FIELDS = [
    "filename", "verdict", "score", "reasons", "group", "group_size", "is_pick",
    "faces_total", "faces_primary", "eyes_open", "eyes_closed", "eyes_squint",
    "eyes_profile", "min_ear", "face_sharp", "global_sharp", "face_luma",
    "mean_luma", "clipped_high", "clipped_low", "contrast", "width", "height",
    "capture_time", "path", "error",
]

VERDICT_ORDER = ["keep", "maybe", "reject", "error"]
VERDICT_COLOR = {"keep": "#1f8f4e", "maybe": "#b8860b", "reject": "#b03030",
                 "error": "#666"}


def write_csv(records: Sequence[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row = dict(rec)
            row["reasons"] = ";".join(rec.get("reasons", []))
            ct = rec.get("capture_time")
            row["capture_time"] = ct.isoformat(sep=" ") if ct else ""
            writer.writerow(row)


def _card(rec: dict) -> str:
    verdict = rec.get("verdict", "error")
    thumb = rec.get("thumb")
    img = (f'<img loading="lazy" src="thumbs/{html.escape(thumb)}" alt="">'
           if thumb else '<div class="noimg">no preview</div>')
    reasons = ", ".join(rec.get("reasons", [])) or "-"
    eyes = (f'{rec.get("eyes_open", 0)} open / {rec.get("eyes_closed", 0)} closed'
            if rec.get("faces_primary") else "no face")
    pick = ' <span class="pick">PICK</span>' if rec.get("is_pick") and rec.get("group_size", 1) > 1 else ""
    return f"""
    <figure class="card" data-verdict="{verdict}">
      {img}
      <figcaption>
        <div class="row"><b>{html.escape(rec.get('filename', ''))}</b></div>
        <div class="row"><span class="score" style="background:{VERDICT_COLOR.get(verdict, '#666')}">{rec.get('score', 0)}</span>
          <span class="v">{verdict}</span>{pick}</div>
        <div class="meta">eyes: {eyes}</div>
        <div class="meta">face sharp: {rec.get('face_sharp', 0)} &middot; skin: {rec.get('face_luma', -1)}</div>
        <div class="meta">clip hi: {rec.get('clipped_high', 0):.3f}</div>
        <div class="meta reasons">{html.escape(reasons)}</div>
      </figcaption>
    </figure>"""


def write_html(records: Sequence[dict], path: str, title: str = "photoflow cull",
               summary: str = "") -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    buckets: dict[str, list[dict]] = {v: [] for v in VERDICT_ORDER}
    for rec in records:
        buckets.setdefault(rec.get("verdict", "error"), []).append(rec)

    sections = []
    for verdict in VERDICT_ORDER:
        items = sorted(buckets.get(verdict, []),
                       key=lambda r: r.get("score", 0), reverse=True)
        if not items:
            continue
        cards = "".join(_card(r) for r in items)
        sections.append(
            f'<h2 id="{verdict}"><span class="dot" style="background:{VERDICT_COLOR[verdict]}"></span>'
            f'{verdict} <small>({len(items)})</small></h2><div class="grid">{cards}</div>'
        )

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{ color-scheme: light dark; --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#e3e3e3; --card:#fafafa; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#16181c; --fg:#e8e8e8; --muted:#9aa0a6; --line:#2c2f36; --card:#1d2026; }}
  }}
  body {{ margin:0; padding:24px; background:var(--bg); color:var(--fg);
         font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  h2 {{ font-size:16px; margin:32px 0 12px; display:flex; align-items:center; gap:8px;
        border-bottom:1px solid var(--line); padding-bottom:8px; text-transform:capitalize; }}
  .dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
  .summary {{ color:var(--muted); margin-bottom:8px; }}
  .grid {{ display:grid; gap:14px; grid-template-columns:repeat(auto-fill,minmax(210px,1fr)); }}
  .card {{ margin:0; background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
  .card img {{ width:100%; display:block; aspect-ratio:3/2; object-fit:cover; background:#000; }}
  .noimg {{ aspect-ratio:3/2; display:grid; place-items:center; color:var(--muted); background:var(--line); }}
  figcaption {{ padding:8px 10px 10px; }}
  .row {{ display:flex; align-items:center; gap:6px; margin-bottom:4px; }}
  .row b {{ font-size:12px; word-break:break-all; }}
  .score {{ color:#fff; border-radius:4px; padding:1px 6px; font-weight:600; font-size:12px; }}
  .v {{ color:var(--muted); font-size:12px; }}
  .pick {{ background:#2b6cb0; color:#fff; border-radius:4px; padding:1px 5px; font-size:10px; }}
  .meta {{ color:var(--muted); font-size:11px; }}
  .reasons {{ margin-top:4px; color:#b03030; }}
  nav a {{ margin-right:12px; color:inherit; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="summary">{html.escape(summary)}</div>
<nav>{"".join(f'<a href="#{v}">{v}</a>' for v in VERDICT_ORDER if buckets.get(v))}</nav>
{"".join(sections)}
</body></html>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)


def print_calibration(records: Sequence[dict]) -> str:
    """Percentile table, so thresholds can be tuned against a real shoot."""
    import numpy as np

    fields = ["face_sharp", "global_sharp", "face_luma", "mean_luma",
              "clipped_high", "clipped_low", "contrast", "min_ear", "score"]
    lines = [f"{'metric':<14}{'p5':>10}{'p25':>10}{'p50':>10}{'p75':>10}{'p95':>10}"]
    for field in fields:
        values = [r[field] for r in records
                  if isinstance(r.get(field), (int, float)) and r[field] >= 0]
        if not values:
            continue
        p = np.percentile(values, [5, 25, 50, 75, 95])
        lines.append(f"{field:<14}" + "".join(f"{v:>10.2f}" for v in p))
    return "\n".join(lines)
