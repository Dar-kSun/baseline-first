"""Unseen genes in an unseen cell type, with shrinkage and a fifth cell line (H1).

    python scripts/14_transfer_with_shrinkage.py

Extends scripts/13 in two ways:

1. Cell lines: K562, RPE1, HepG2, Jurkat (Replogle 2022 + Nadig 2025, from
   scripts/11) plus H1 (the VCC 2025 CRISPRi data, 10x Flex, from scripts/10).
   All are restricted to the genes the two sources share.
2. Shrinkage. Scripts/13 found transferred effects too large: on two lines the
   average response did worse than predicting no change. Here each method also
   gets a version scaled by one factor s, fitted without the held-out line or
   the held-out genes: for each training line M, the method is refitted on the
   other training lines, excluding one further gene fold, and predicts M on
   that fold; s is the least-squares scale over those predictions.

As in scripts/13: genes are split into 5 folds; testing fold F in line L trains
only on the other lines and the other folds (`assert_no_leakage` on both keys);
the held-out line's control cells provide its co-expression features.
ContextMean* (the VCC 0 point) reads the test data and is a reference only.

Writes results/transfer_with_shrinkage/{per_perturbation.csv, summary.csv,
summary.md, run.json}.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_first import metrics
from baseline_first.splits import assert_no_leakage
from baseline_first.vcc.transfer import (
    CoexpressionRidge,
    correlation_features,
    load_lines,
    optimal_scale,
    variances,
)

SOURCES = [
    Path("data/processed/replogle_nadig_pseudobulk.h5ad"),
    Path("data/processed/vcc2025/competition_train_pseudobulk.h5ad"),
]
OUT = Path("results/transfer_with_shrinkage")
N_FOLDS = 5
N_ANCHORS = 1000
SEED = 0


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        return sha.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except OSError:
        return "unknown"


def frame(pairs) -> pd.DataFrame:
    return pd.DataFrame(pairs, columns=["cell_line", "perturbation"])


def tile(row: pd.Series, index) -> pd.DataFrame:
    return pd.DataFrame(np.tile(row.to_numpy(), (len(index), 1)), index=index, columns=row.index)


def score(method, line, fold, pred, true, panel):
    keep = [g for g in true.columns if g not in panel]
    p, t = pred[keep].to_numpy(), true[keep].to_numpy()
    return pd.DataFrame(
        {
            "method": method,
            "held_out": line,
            "fold": fold,
            "perturbation": true.index,
            "pds": metrics.pds_cosine(p, t),
            "sq_error": ((p - t) ** 2).sum(axis=1),
            "sq_effect": (t**2).sum(axis=1),
            "pearson": metrics.pearson_delta(p, t, np.zeros(t.shape[1])),
        }
    )


def subset(eff, lines, keep_gene):
    return {m: eff[m].loc[[p for p in eff[m].index if keep_gene(p)]] for m in lines}


def main() -> None:
    commit = git_commit()
    genes, eff, moments = load_lines(SOURCES)
    lines = sorted(eff)
    all_perts = sorted(set().union(*(e.index for e in eff.values())))
    rng = np.random.default_rng(SEED)
    fold_of = dict(
        zip(rng.permutation(all_perts), np.arange(len(all_perts)) % N_FOLDS, strict=True)
    )
    print(f"{len(lines)} cell lines, {len(genes)} shared genes, {len(all_perts)} perturbed genes")

    rows, record = [], []
    for line in lines:
        train_lines = [m for m in lines if m != line]
        pooled = sum(variances(moments[m]) for m in train_lines)
        anchors = np.sort(np.argsort(-pooled, kind="stable")[:N_ANCHORS])
        feats = {m: correlation_features(moments[m], genes, anchors) for m in lines}
        context_mean = eff[line].mean(axis=0)

        for fold in range(N_FOLDS):
            test_perts = [p for p in eff[line].index if fold_of[p] == fold]
            train = subset(eff, train_lines, lambda p, f=fold: fold_of[p] != f)
            assert_no_leakage(
                frame([(m, p) for m, e in train.items() for p in e.index]),
                frame([(line, p) for p in test_perts]),
                keys=["cell_line", "perturbation"],
            )
            true = eff[line].loc[test_perts]
            mean_response = pd.concat(train.values()).mean(axis=0)
            ridge = CoexpressionRidge(seed=SEED).fit([(feats[m], train[m]) for m in train_lines])

            # Shrinkage: refit without each training line and one more gene fold.
            inner_fold = (fold + 1) % N_FOLDS
            ridge_pairs, mean_pairs = [], []
            for held in train_lines:
                rest = [m for m in train_lines if m != held]
                inner = subset(eff, rest, lambda p, f=fold, g=inner_fold: fold_of[p] not in (f, g))
                target = eff[held].loc[[p for p in eff[held].index if fold_of[p] == inner_fold]]
                inner_ridge = CoexpressionRidge(alphas=(ridge.alpha_,), n_inner_folds=2).fit(
                    [(feats[m], inner[m]) for m in rest]
                )
                ridge_pairs.append(
                    (inner_ridge.predict(feats[held], target.index).to_numpy(), target.to_numpy())
                )
                inner_mean = pd.concat(inner.values()).mean(axis=0)
                mean_pairs.append((tile(inner_mean, target.index).to_numpy(), target.to_numpy()))
            s_ridge, s_mean = optimal_scale(ridge_pairs), optimal_scale(mean_pairs)
            record.append(
                {
                    "held_out": line,
                    "fold": fold,
                    "alpha": ridge.alpha_,
                    "s_ridge": s_ridge,
                    "s_mean": s_mean,
                }
            )

            ridge_pred = ridge.predict(feats[line], test_perts)
            mean_pred = tile(mean_response, test_perts)
            preds = {
                "NoChange": true * 0,
                "MeanResponse": mean_pred,
                "MeanResponse+shrink": mean_pred * s_mean,
                "CoexpressionRidge": ridge_pred,
                "CoexpressionRidge+shrink": ridge_pred * s_ridge,
                "ContextMean*": tile(context_mean, test_perts),
            }
            for method, pred in preds.items():
                rows.append(score(method, line, fold, pred, true, set(test_perts)))
            print(
                f"{line} fold {fold}: {len(test_perts)} perts, alpha {ridge.alpha_:g}, "
                f"s_ridge {s_ridge:.2f}, s_mean {s_mean:.2f}",
                flush=True,
            )
    results = pd.concat(rows, ignore_index=True)

    summary = []
    for (method, line), f in results.groupby(["method", "held_out"], sort=False):
        pds, low, high = metrics.bootstrap_ci(f["pds"], f["perturbation"])
        summary.append(
            {
                "method": method,
                "held_out": line,
                "n_perturbations": len(f),
                "pds": pds,
                "pds_ci_low": low,
                "pds_ci_high": high,
                "expr_error_ratio": f["sq_error"].sum() / f["sq_effect"].sum(),
                "pearson": f["pearson"].mean(),
            }
        )
    summary = pd.DataFrame(summary)

    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / "per_perturbation.csv", index=False, float_format="%.6g")
    summary.to_csv(OUT / "summary.csv", index=False, float_format="%.6g")
    table = [
        "| held out | method | n | PDS [95% CI] | expression error ratio | Pearson |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary.sort_values(["held_out", "method"]).itertuples():
        table.append(
            f"| {r.held_out} | {r.method} | {r.n_perturbations} "
            f"| {r.pds:.3f} [{r.pds_ci_low:.3f}, {r.pds_ci_high:.3f}] "
            f"| {r.expr_error_ratio:.3f} | {r.pearson:.3f} |"
        )
    text = "\n".join(table) + "\n"
    (OUT / "summary.md").write_text(text)
    (OUT / "run.json").write_text(
        json.dumps(
            {
                "command": " ".join(["python", *sys.argv]),
                "git_commit": commit,
                "folds": N_FOLDS,
                "anchors": N_ANCHORS,
                "seed": SEED,
                "n_shared_genes": len(genes),
                "fits": record,
                "sources": [str(s) for s in SOURCES],
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print(text)


if __name__ == "__main__":
    main()
