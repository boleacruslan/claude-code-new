"""Train per-parameter regressors on a photographer's own edits.

Two rules shape this file, and both exist to stop it from lying:

1. **Validation splits by shoot, never by frame.** Frames inside one wedding
   are near-identical; a random split puts a burst's siblings on both sides of
   the fence, the score comes out beautiful, and the model then falls over on
   a wedding it has never seen. Every number reported here comes from
   weddings that were held out whole.

2. **Every parameter must beat the median baseline.** Predicting "what this
   photographer usually does" is free and surprisingly strong. A parameter
   where the model cannot beat that is not knowledge, it is noise wearing a
   lab coat - it gets demoted to a constant and reported as not learned.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold

from . import xmp

MIN_COVERAGE = 0.5      # a parameter must be present on this share of frames
MIN_SAMPLES = 60
MIN_GAIN = 0.05         # must beat the baseline by at least 5%


@dataclass
class ParamReport:
    name: str
    samples: int
    mae_model: float
    mae_baseline: float
    learned: bool
    constant: float

    @property
    def gain(self) -> float:
        if self.mae_baseline <= 0:
            return 0.0
        return 1.0 - self.mae_model / self.mae_baseline


@dataclass
class StyleModel:
    feature_names: list[str]
    models: dict = field(default_factory=dict)
    constants: dict[str, float] = field(default_factory=dict)
    camera_vocab: dict[str, int] = field(default_factory=dict)
    lens_vocab: dict[str, int] = field(default_factory=dict)
    reports: list[ParamReport] = field(default_factory=list)

    def predict(self, matrix: np.ndarray) -> dict[str, np.ndarray]:
        """Predicted settings for each row, clamped to Lightroom's ranges."""
        out: dict[str, np.ndarray] = {}
        for name, model in self.models.items():
            values = model.predict(matrix)
            lo, hi = xmp.PARAM_RANGE.get(name, (-1e6, 1e6))
            out[name] = np.clip(values, lo, hi)
        for name, value in self.constants.items():
            out[name] = np.full(matrix.shape[0], value, dtype=float)
        return out


def _make_regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.06,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=0,
    )


def _cross_val_mae(matrix: np.ndarray, target: np.ndarray, groups: np.ndarray,
                   by_group: bool = True) -> tuple[float, float]:
    """Mean absolute error of the model and of the median baseline.

    Both are measured on exactly the same folds, so the comparison is fair.
    """
    unique = np.unique(groups)
    if by_group:
        splits = min(5, len(unique))
        if splits < 2:
            return float("nan"), float("nan")
        splitter = GroupKFold(n_splits=splits).split(matrix, target, groups)
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=0).split(matrix)

    model_errors, base_errors = [], []
    for train_idx, test_idx in splitter:
        if len(train_idx) < 20 or len(test_idx) < 5:
            continue
        model = _make_regressor()
        model.fit(matrix[train_idx], target[train_idx])
        model_errors.append(
            np.abs(model.predict(matrix[test_idx]) - target[test_idx]).mean())
        baseline = np.median(target[train_idx])
        base_errors.append(np.abs(baseline - target[test_idx]).mean())

    if not model_errors:
        return float("nan"), float("nan")
    return float(np.mean(model_errors)), float(np.mean(base_errors))


def train(matrix: np.ndarray, targets: dict[str, np.ndarray],
          present: dict[str, np.ndarray], groups: np.ndarray,
          feature_names: list[str],
          camera_vocab: Optional[dict] = None,
          lens_vocab: Optional[dict] = None) -> StyleModel:
    """Fit one regressor per develop parameter that earns its place.

    ``targets[name]`` holds values and ``present[name]`` the boolean mask of
    rows where the photographer actually set that parameter.
    """
    style = StyleModel(feature_names=list(feature_names),
                       camera_vocab=dict(camera_vocab or {}),
                       lens_vocab=dict(lens_vocab or {}))

    for name in xmp.ALL_PARAMS:
        if name not in targets:
            continue
        mask = present[name]
        count = int(mask.sum())
        if count < MIN_SAMPLES or count / len(mask) < MIN_COVERAGE:
            continue

        sub_x, sub_y, sub_g = matrix[mask], targets[name][mask], groups[mask]
        median = float(np.median(sub_y))

        # A parameter the photographer always leaves at one value carries no
        # signal to learn; pass it straight through.
        if float(np.percentile(sub_y, 90) - np.percentile(sub_y, 10)) < 1e-6:
            style.constants[name] = median
            style.reports.append(ParamReport(name, count, 0.0, 0.0, False, median))
            continue

        mae_model, mae_base = _cross_val_mae(sub_x, sub_y, sub_g)
        if np.isnan(mae_model):
            style.constants[name] = median
            style.reports.append(ParamReport(name, count, float("nan"),
                                             float("nan"), False, median))
            continue

        beat_baseline = mae_model < mae_base * (1.0 - MIN_GAIN)
        report = ParamReport(name, count, mae_model, mae_base, beat_baseline, median)
        style.reports.append(report)

        if beat_baseline:
            final = _make_regressor()
            final.fit(sub_x, sub_y)
            style.models[name] = final
        else:
            style.constants[name] = median

    return style


def leakage_check(matrix: np.ndarray, target: np.ndarray,
                  groups: np.ndarray) -> tuple[float, float]:
    """Same data, split two ways - the gap is the leak a random split hides."""
    grouped, _ = _cross_val_mae(matrix, target, groups, by_group=True)
    random_, _ = _cross_val_mae(matrix, target, groups, by_group=False)
    return grouped, random_


def render_report(style: StyleModel) -> str:
    """Per-parameter accuracy, in slider units a photographer can judge."""
    learned = [r for r in style.reports if r.learned]
    skipped = [r for r in style.reports if not r.learned]
    lines = [
        f"learned {len(learned)} of {len(style.reports)} parameters "
        f"(validated on held-out shoots)",
        "",
        f"   {'parameter':<28}{'n':>7}{'error':>10}{'baseline':>10}{'gain':>8}",
    ]
    for report in sorted(learned, key=lambda r: -r.gain):
        lines.append(f"   {report.name:<28}{report.samples:>7}"
                     f"{report.mae_model:>10.2f}{report.mae_baseline:>10.2f}"
                     f"{report.gain * 100:>7.0f}%")
    if skipped:
        lines += ["", "not learned - emitted as your usual value instead:"]
        for report in skipped:
            reason = ("no variation" if report.mae_baseline == 0
                      else f"no gain over baseline ({report.mae_model:.2f} "
                           f"vs {report.mae_baseline:.2f})")
            lines.append(f"   {report.name:<28} {reason}")
    lines += ["", "'error' is mean absolute error on shoots the model never saw,",
              "in the slider's own units. 'baseline' is what you get by always",
              "applying your own median value."]
    return "\n".join(lines)


def save(style: StyleModel, path: str) -> None:
    import joblib

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    joblib.dump(style, path)


def load(path: str) -> StyleModel:
    import joblib

    return joblib.load(path)
