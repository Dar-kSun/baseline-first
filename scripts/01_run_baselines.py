"""Run the baselines on a public dataset with perturbation-held-out cross-validation.

    python scripts/01_run_baselines.py --dataset norman2019

Writes to results/<dataset>/:
  per_perturbation.csv  one row per (baseline, perturbation): MAE, RMSE, Pearson delta
  summary.csv           mean and 95% CI bootstrapped over perturbations, and the
                        scaled score (GlobalMean = 0, split-half Replicate = 1)
  summary.md            the same summary as a table
  run.json              command, seed, folds, git commit and package versions
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from baseline_first.baselines import GlobalMean, HVGRidge
from baseline_first.data import load_norman2019
from baseline_first.evaluate import evaluate, summarise

LOADERS = {"norman2019": load_norman2019}
PACKAGES = ["baseline-first", "numpy", "scipy", "scikit-learn", "anndata", "pandas"]


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        return out + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def to_markdown(summary) -> str:
    """Raw means with 95% CIs, then scaled scores (GlobalMean = 0, Replicate = 1)."""
    lines = [
        "| subset | baseline | n perts | MAE | RMSE | Pearson delta "
        "| scaled MAE | scaled RMSE | scaled Pearson delta |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (subset, baseline), frame in summary.groupby(["subset", "baseline"], sort=False):
        raw, scaled = {}, {}
        for row in frame.itertuples():
            raw[row.metric] = f"{row.mean:.4f} [{row.ci_low:.4f}, {row.ci_high:.4f}]"
            scaled[row.metric] = (
                f"{row.scaled:.2f} [{row.scaled_ci_low:.2f}, {row.scaled_ci_high:.2f}]"
            )
        n = frame["n_perturbations"].iloc[0]
        metrics = ("mae", "rmse", "pearson_delta")
        lines.append(
            f"| {subset} | {baseline} | {n} | "
            + " | ".join(raw[m] for m in metrics)
            + " | "
            + " | ".join(scaled[m] for m in metrics)
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", choices=sorted(LOADERS), default="norman2019")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=4096, help="HVGRidge: number of HVGs")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    adata = LOADERS[args.dataset]()
    for layer in [name for name in adata.layers if name is not None]:
        del adata.layers[layer]  # only X is used; frees memory (newer anndata lists X as None)
    baselines = [GlobalMean(), HVGRidge(k=args.k)]

    results = evaluate(adata, baselines, n_folds=args.folds, seed=args.seed)
    summary = summarise(results, seed=args.seed)

    out = args.out / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "per_perturbation.csv", index=False, float_format="%.6g")
    summary.to_csv(out / "summary.csv", index=False, float_format="%.6g")
    table = to_markdown(summary)
    (out / "summary.md").write_text(table)
    run = {
        "command": " ".join(["python", *sys.argv]),
        "dataset": args.dataset,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "folds": args.folds,
        "seed": args.seed,
        "hvg_k": args.k,
        "split": "GroupSplitter(perturbation), control always in train",
        "ci": "95% percentile bootstrap over perturbations, 2000 resamples",
        "observed_profile": "half B of each held-out perturbation's cells (random split)",
        "replicate_ceiling": "mean of half A, scored against half B",
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "packages": {p: version(p) for p in PACKAGES},
        "source": adata.uns["baseline_first"],
    }
    (out / "run.json").write_text(json.dumps(run, indent=2, default=str) + "\n")
    print(table)


if __name__ == "__main__":
    main()
