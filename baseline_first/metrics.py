"""Scores for predicted perturbation profiles, and confidence intervals over groups.

All per-perturbation scores take predicted and observed profiles with one row
per perturbation. Correlation scores compare *deltas*: each profile minus the
control mean, so a model gets no credit for reproducing baseline expression.

Confidence intervals are bootstrapped over groups (perturbations), never over
cells. Cells within a perturbation are near-replicates, so resampling them
treats correlated measurements as independent and fakes precision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Mean absolute error per row (per perturbation)."""
    return np.mean(np.abs(np.asarray(pred) - np.asarray(true)), axis=1)


def rmse(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Root mean squared error per row (per perturbation)."""
    return np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2, axis=1))


def pearson_delta(pred: np.ndarray, true: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Pearson correlation per row between predicted and observed change from control.

    Returns NaN for a row where either delta is constant across genes.
    """
    a = np.asarray(pred) - control
    b = np.asarray(true) - control
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (a * b).sum(axis=1) / denom, np.nan)


def scaled_score_ci(
    model,
    floor,
    ceiling,
    higher_is_better: bool,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (score, low, high) on a scale where `floor` scores 0 and `ceiling` scores 1.

    The three arguments are per-perturbation values of one metric, aligned by
    perturbation. The score compares means over perturbations:
    ``(mean(model) - mean(floor)) / (mean(ceiling) - mean(floor))``, with the
    sign handled so that higher is always better. The CI resamples
    perturbations, using the same draw for all three, so their pairing is kept.
    """
    values = np.column_stack([model, floor, ceiling]).astype(float)
    values = values[~np.isnan(values).any(axis=1)]
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)

    def score(means: np.ndarray) -> np.ndarray:
        m, f, c = means[..., 0], means[..., 1], means[..., 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            return (m - f) / (c - f) if higher_is_better else (f - m) / (f - c)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = score(values[draws].mean(axis=1))
    tail = (1 - level) / 2
    low, high = np.nanquantile(boot, [tail, 1 - tail])
    return float(score(values.mean(axis=0))), float(low), float(high)


def bootstrap_ci(
    values, groups, n_boot: int = 2000, level: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """Return (mean, low, high): the mean over groups and a percentile bootstrap CI.

    `values` may have several entries per group (e.g. one per cell); they are
    averaged within each group first, and whole groups are resampled. The mean
    is the unweighted mean of group means. NaN values are dropped.
    """
    frame = pd.DataFrame({"value": np.asarray(values, dtype=float), "group": np.asarray(groups)})
    per_group = frame.dropna().groupby("group")["value"].mean().to_numpy()
    if len(per_group) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(per_group), size=(n_boot, len(per_group)))
    boot = per_group[draws].mean(axis=1)
    tail = (1 - level) / 2
    low, high = np.quantile(boot, [tail, 1 - tail])
    return float(per_group.mean()), float(low), float(high)
