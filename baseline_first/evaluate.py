"""Cross-validated evaluation of baselines on held-out perturbations.

Each fold holds out whole perturbations. `assert_no_leakage` runs inside the
loop, on every fold, on the exact train and test sets the baselines see, so a
custom or buggy splitter cannot slip a leaky split past it.

A replicate ceiling is measured alongside the baselines. Each held-out
perturbation's cells are split at random into two halves: half B is the
observed profile that everything is scored against, and half A's mean is the
"Replicate" prediction, i.e. what an independent repeat of the experiment
would give. Scaled scores put GlobalMean at 0 and Replicate at 1, the same
anchors the Virtual Cell Challenge 2026 uses for its reference-scaled metrics.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from baseline_first import metrics
from baseline_first.data.schema import CONTROL
from baseline_first.pseudobulk import group_means
from baseline_first.splits import GroupSplitter, assert_no_leakage

KEY = "perturbation"
FLOOR = "GlobalMean"
CEILING = "Replicate"
METRICS = {"mae": False, "rmse": False, "pearson_delta": True}  # name -> higher is better


def _halves(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Assign each cell to half 0 (A) or 1 (B), balanced within each group."""
    half = np.empty(len(labels), dtype=np.int8)
    for group in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == group))
        half[idx[: len(idx) // 2]] = 0
        half[idx[len(idx) // 2 :]] = 1
    return half


def _score(predicted, observed, control) -> dict[str, np.ndarray]:
    return {
        "mae": metrics.mae(predicted, observed),
        "rmse": metrics.rmse(predicted, observed),
        "pearson_delta": metrics.pearson_delta(predicted, observed, control),
    }


def evaluate(
    adata,
    baselines,
    n_folds: int = 5,
    seed: int = 0,
    splitter=None,
    gene_key: str = "gene_name",
) -> pd.DataFrame:
    """Score each baseline, and the replicate ceiling, on every held-out perturbation.

    Returns one row per (baseline, perturbation) with MAE, RMSE and Pearson
    correlation of the change from control, all measured against half B of the
    perturbation's cells. The control profile comes from training cells only.
    `n_constituents_in_train` counts, for a pair A+B, how many of the single
    perturbations A and B were in that fold's training set.
    """
    splitter = splitter or GroupSplitter(KEY, seed=seed, always_train=[CONTROL])
    genes = adata.var[gene_key].astype(str).to_numpy() if gene_key in adata.var else adata.var_names
    rows = []
    for fold, split in enumerate(splitter.folds(adata, n_folds=n_folds)):
        train, test = adata[split.train], adata[split.test]
        assert_no_leakage(train, test, keys=KEY)
        if (test.obs[KEY] == CONTROL).any():
            raise ValueError("control cells must stay in training; they define the reference")

        train_labels = train.obs[KEY].astype(str).to_numpy()
        test_labels = test.obs[KEY].astype(str).to_numpy()
        half = _halves(test_labels, np.random.default_rng([seed, fold]))
        test_groups, observed = group_means(test.X[half == 1], test_labels[half == 1])
        replicate_groups, replicate = group_means(test.X[half == 0], test_labels[half == 0])
        if not np.array_equal(test_groups, replicate_groups):
            raise ValueError("every held-out perturbation needs at least 2 cells")
        train_groups, train_means = group_means(train.X, train_labels)
        control = train_means[train_groups == CONTROL][0]
        n_cells = pd.Series(test_labels).value_counts()
        info = pd.DataFrame(
            {
                "fold": fold,
                KEY: test_groups,
                "n_targets": [g.count("+") + 1 for g in test_groups],
                "n_constituents_in_train": [
                    sum(p in set(train_groups) for p in g.split("+")) if "+" in g else 0
                    for g in test_groups
                ],
                "n_cells": n_cells[test_groups].to_numpy(),
            }
        )

        rows.append(info.assign(baseline=CEILING, **_score(replicate, observed, control)))
        for baseline in baselines:
            start = time.perf_counter()
            baseline.fit(train.X, train_labels, genes)
            predicted = baseline.predict(test_groups)
            seconds = time.perf_counter() - start
            rows.append(
                info.assign(
                    baseline=baseline.name,
                    **_score(predicted, observed, control),
                    fit_predict_seconds=seconds,
                )
            )
    results = pd.concat(rows, ignore_index=True)
    return results[["baseline", *results.columns.drop("baseline")]]


def summarise(results: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per baseline, metric and subset: the raw mean and, when the floor and ceiling were
    run, the scaled score (GlobalMean = 0, Replicate = 1). 95% CIs over perturbations."""
    subsets = {
        "all": results,
        "single": results[results["n_targets"] == 1],
        "pair": results[results["n_targets"] == 2],
    }
    rows = []
    for subset, frame in subsets.items():
        wide = {m: frame.pivot(index=KEY, columns="baseline", values=m) for m in METRICS}
        names = list(dict.fromkeys(frame["baseline"]))
        for name in names:
            by_baseline = frame[frame["baseline"] == name]
            for metric, higher_is_better in METRICS.items():
                mean, low, high = metrics.bootstrap_ci(
                    by_baseline[metric], by_baseline[KEY], n_boot=n_boot, seed=seed
                )
                row = {
                    "subset": subset,
                    "baseline": name,
                    "metric": metric,
                    "n_perturbations": by_baseline[KEY].nunique(),
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
                if {FLOOR, CEILING} <= set(names):
                    table = wide[metric]
                    scaled = metrics.scaled_score_ci(
                        table[name],
                        table[FLOOR],
                        table[CEILING],
                        higher_is_better,
                        n_boot=n_boot,
                        seed=seed,
                    )
                    row.update(
                        zip(("scaled", "scaled_ci_low", "scaled_ci_high"), scaled, strict=True)
                    )
                rows.append(row)
    return pd.DataFrame(rows)
