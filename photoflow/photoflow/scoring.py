"""Turn raw measurements into a 0-100 score and a keep/maybe/reject verdict."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class CullConfig:
    """Everything a photographer might want to tune, in one place."""

    # Hard floors - these reject a frame outright, whatever else is good.
    # blur_floor 0 means "derive it from this shoot" - an absolute acutance
    # number is meaningless across bodies, lenses and subject distance.
    blur_floor: float = 0.0
    face_luma_low: float = 45.0      # face this dark is unrecoverable in 8-bit
    face_luma_high: float = 228.0    # face this bright has no detail left
    clip_high_max: float = 0.30      # fraction of frame blown to pure white
    clip_low_max: float = 0.55

    # Eye detection
    ear_closed: float = 0.18
    ear_open: float = 0.22
    min_face_frac: float = 0.004     # ignore faces smaller than 0.4% of frame
    face_rel_to_largest: float = 0.30
    tiles: int = 2                   # tiled sweep grid for small faces, 0 = off

    # Verdict thresholds
    keep_score: float = 62.0
    reject_score: float = 38.0

    # Burst handling
    burst_gap_seconds: float = 3.0
    burst_hash_distance: int = 12
    duplicate_verdict: str = "maybe"  # what to do with non-picks in a burst

    def as_dict(self) -> dict:
        return asdict(self)


def _ramp(value: float, lo: float, hi: float) -> float:
    """Linear 0..1 ramp, clamped."""
    if hi <= lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def _window(value: float, lo: float, ideal_lo: float, ideal_hi: float, hi: float) -> float:
    """1.0 inside the ideal band, falling to 0 at the outer bounds."""
    if value < ideal_lo:
        return _ramp(value, lo, ideal_lo)
    if value > ideal_hi:
        return 1.0 - _ramp(value, ideal_hi, hi)
    return 1.0


def sharpness_reference(records: list[dict]) -> tuple[float, float, float]:
    """Normalisation band and median taken from the shoot itself.

    Absolute sharpness numbers depend on body, lens, aperture and subject
    distance, so a fixed threshold either nukes a soft-focus set or passes
    everything. Taking p20..p85 of the shoot makes the score relative to the
    session's own standard, with sane fallbacks for tiny batches.
    """
    values = [r["face_sharp"] if r["faces_primary"] else r["global_sharp"]
              for r in records]
    values = [v for v in values if v > 0]
    if not values:
        return 1.0, 10.0, 3.0
    median = float(np.median(values))
    if len(values) < 8:
        # Too few frames for stable percentiles. Anchoring the band on the
        # median keeps it scale-free, which invented constants are not: an
        # absolute band tuned for a 45MP body would call every frame from a
        # smaller file soft.
        return max(0.1, median * 0.45), median * 1.8, median
    lo = float(np.percentile(values, 20))
    hi = float(np.percentile(values, 85))
    if hi - lo < 1e-3:
        hi = lo + 1.0
    return lo, hi, median


def auto_blur_floor(cfg: CullConfig, median: float) -> float:
    """Acutance below which a frame is called unusable.

    Relative by default: a frame carrying barely a third of the detail this
    shoot normally resolves is a focus miss, whatever the absolute number.
    """
    if cfg.blur_floor > 0:
        return cfg.blur_floor
    return max(0.5, median * 0.35)


def score_record(rec: dict, cfg: CullConfig, sharp_lo: float, sharp_hi: float,
                 blur_floor: float) -> None:
    """Fill in ``score``, ``verdict`` and ``reasons`` on a metrics record."""
    reasons: list[str] = []
    has_face = rec["faces_primary"] > 0

    sharp_value = rec["face_sharp"] if has_face else rec["global_sharp"]
    sharp_norm = _ramp(sharp_value, sharp_lo, sharp_hi)

    if has_face and rec["face_luma"] >= 0:
        expo_norm = _window(rec["face_luma"], 25.0, 95.0, 200.0, 245.0)
    else:
        expo_norm = _window(rec["mean_luma"], 15.0, 70.0, 185.0, 240.0)
    # Blown highlights are judged separately: a white dress clips a little and
    # that is normal, a blown-out frame is not.
    expo_norm *= 1.0 - _ramp(rec["clipped_high"], 0.08, cfg.clip_high_max)
    expo_norm *= 1.0 - _ramp(rec["clipped_low"], 0.25, cfg.clip_low_max)

    if has_face:
        judged = rec["eyes_open"] + rec["eyes_closed"] + rec["eyes_squint"]
        if judged == 0:
            eyes_norm = 0.6           # every face in profile - not a defect
        else:
            eyes_norm = (rec["eyes_open"] + 0.45 * rec["eyes_squint"]) / judged
    else:
        eyes_norm = 0.0

    contrast_norm = _window(rec["contrast"], 20.0, 90.0, 250.0, 256.0)

    if has_face:
        weights = (("sharp", sharp_norm, 0.34), ("expo", expo_norm, 0.26),
                   ("eyes", eyes_norm, 0.28), ("contrast", contrast_norm, 0.12))
    else:
        weights = (("sharp", sharp_norm, 0.50), ("expo", expo_norm, 0.36),
                   ("contrast", contrast_norm, 0.14))

    # Sharpness also gates the total rather than only contributing to it.
    # Great expression, perfect light and open eyes cannot rescue a frame the
    # lens missed, so a fully soft image keeps barely half of whatever else
    # it earned.
    gate = 0.55 + 0.45 * sharp_norm
    score = 100.0 * sum(value * weight for _, value, weight in weights) * gate

    # --- hard failures -------------------------------------------------
    if sharp_value < blur_floor:
        reasons.append("blurred")
    if has_face and rec["face_luma"] >= 0:
        if rec["face_luma"] < cfg.face_luma_low:
            reasons.append("face_too_dark")
        elif rec["face_luma"] > cfg.face_luma_high:
            reasons.append("face_blown")
    if rec["clipped_high"] > cfg.clip_high_max:
        reasons.append("overexposed")
    if rec["clipped_low"] > cfg.clip_low_max:
        reasons.append("underexposed")
    if has_face and rec["eyes_open"] == 0 and rec["eyes_closed"] > 0:
        reasons.append("eyes_closed")

    rec["score"] = round(score, 1)
    rec["hard_fail"] = bool(reasons)
    if not reasons:
        if sharp_norm < 0.25:
            reasons.append("soft")
        if has_face and rec["eyes_closed"] > 0:
            reasons.append("some_eyes_closed")
        if expo_norm < 0.5:
            reasons.append("exposure_off")
    rec["reasons"] = reasons
    rec["_parts"] = {name: round(value, 3) for name, value, _ in weights}


def decide(rec: dict, cfg: CullConfig) -> None:
    """Final verdict, after scores and burst picks are known."""
    if rec["hard_fail"]:
        rec["verdict"] = "reject"
        return
    if rec["score"] < cfg.reject_score:
        rec["verdict"] = "reject"
        rec["reasons"] = rec["reasons"] or ["low_score"]
        return
    if rec.get("group_size", 1) > 1 and not rec.get("is_pick", True):
        if cfg.duplicate_verdict != "keep":
            rec["verdict"] = cfg.duplicate_verdict
            rec["reasons"] = rec["reasons"] + ["duplicate"]
            return
    rec["verdict"] = "keep" if rec["score"] >= cfg.keep_score else "maybe"
