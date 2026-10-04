"""The verdict card: what a trivial baseline scores, how much data there really is,
and the bar a bigger model has to clear.

`make_card` takes the outputs of `evaluate.summarise` and `effective_n.effective_n`
and writes a one-page Markdown card plus a figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from baseline_first.effective_n import EffectiveN  # noqa: E402

FLOOR, CEILING = "GlobalMean", "Replicate"
METRIC_NAMES = {"mae": "MAE", "rmse": "RMSE", "pearson_delta": "Pearson delta"}


def the_bar(summary: pd.DataFrame, subset: str = "all") -> pd.DataFrame:
    """Per metric: the best baseline (excluding the replicate ceiling) and the upper
    end of its scaled-score 95% CI, which a new model must exceed."""
    rows = summary[(summary["subset"] == subset) & (summary["baseline"] != CEILING)]
    best = rows.loc[rows.groupby("metric")["scaled"].idxmax()]
    return best[["metric", "baseline", "scaled", "scaled_ci_low", "scaled_ci_high"]]


def _figure(summary: pd.DataFrame, n: EffectiveN, path: Path, title: str) -> None:
    rows = summary[(summary["subset"] == "all") & (summary["metric"] == "mae")]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
    names = list(rows["baseline"])
    y = rows["scaled"].to_numpy()
    err = [y - rows["scaled_ci_low"].to_numpy(), rows["scaled_ci_high"].to_numpy() - y]
    left.barh(names, y, xerr=err, color="#4C72B0", capsize=3)
    left.axvline(0, color="grey", lw=0.8)
    left.axvline(1, color="grey", lw=0.8, ls="--")
    left.set_xlabel("scaled MAE score (0 = GlobalMean, 1 = replicate)")
    left.set_title("Baselines, 95% CI over perturbations")
    counts = {"cells": n.n_cells, "design-effect n_eff": n.n_eff, "conditions": n.n_conditions}
    right.barh(list(counts), list(counts.values()), color="#DD8452")
    right.set_xscale("log")
    right.set_xlabel("count (log scale)")
    right.set_title("How much data is there?")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_card(
    dataset: str,
    summary: pd.DataFrame,
    n: EffectiveN,
    out_dir: Path,
    command: str,
    split: str,
) -> Path:
    """Write verdict.md and verdict.png into `out_dir`; return the card's path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _figure(summary, n, out_dir / "verdict.png", f"baseline-first verdict: {dataset}")
    rows = summary[summary["subset"] == "all"]
    table = ["| method | MAE | Pearson delta | scaled MAE [95% CI] |", "|---|---|---|---|"]
    for name, f in rows.groupby("baseline", sort=False):
        mae = f[f["metric"] == "mae"].iloc[0]
        pear = f[f["metric"] == "pearson_delta"].iloc[0]
        table.append(
            f"| {name} | {mae['mean']:.4f} | {pear['mean']:.3f} "
            f"| {mae['scaled']:.2f} [{mae['scaled_ci_low']:.2f}, {mae['scaled_ci_high']:.2f}] |"
        )
    bar = the_bar(summary)
    bar_lines = [
        f"- **{METRIC_NAMES[r.metric]}:** beat {r.baseline}'s scaled score of {r.scaled:.2f} "
        f"by more than its 95% CI, i.e. score above **{r.scaled_ci_high:.2f}**."
        for r in bar.itertuples()
    ]
    conditions_x = n.inflation_conditions
    text = f"""# Verdict card: {dataset}

Produced by `{command}`. Split: {split}.

## How much data is there?

| | |
|---|---|
| cells | {n.n_cells:,} |
| distinct conditions | {n.n_conditions:,} (cells overstate this **{conditions_x:.0f}x**) |
| design-effect n_eff | {n.n_eff:,.0f} (median ICC {n.icc_median:.3f}; {n.inflation:.1f}x) |

A model of perturbation effects learns from {n.n_conditions:,} examples, not {n.n_cells:,}.

## What do trivial baselines score?

{chr(10).join(table)}

Scaled scores: 0 = GlobalMean (one average profile for every perturbation),
1 = a split-half replicate of the experiment. CIs are bootstrapped over
perturbations, never cells.

## The bar a bigger model must clear (on this split and these metrics)

{chr(10).join(bar_lines)}

A model that does not clear these bars has not shown it beats a ridge
regression on this data. Clearing them on this dataset says nothing about
other datasets, splits or metrics.

![verdict](verdict.png)
"""
    path = out_dir / "verdict.md"
    path.write_text(text, encoding="utf-8")
    return path
