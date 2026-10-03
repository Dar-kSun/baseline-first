"""Unseen genes in an unseen cell type: a local stand-in for the VCC 2026 task.

    python scripts/13_unseen_gene_unseen_context.py

Data: pseudobulks and control-cell moments of K562, RPE1, HepG2 and Jurkat
from scripts/11_pseudobulk_replogle_nadig.py (Arc's harmonised Replogle 2022
+ Nadig 2025 CRISPRi screens, 6,546 measured genes).

The VCC holds out both the cell context and the target genes, so this does
too. Perturbed genes are split into 5 folds. To test fold F in cell line L,
methods train only on the other cell lines AND the other gene folds;
`assert_no_leakage` checks both keys on every split. The held-out line's
control cells are available, as in the challenge.

Methods (all predict effects: profile minus the held-out line's control profile):
  NoChange           zero effect
  MeanResponse       average effect over the training lines and genes
  CoexpressionRidge  baseline_first.vcc.transfer: target gene's correlation with
                     anchor genes in the held-out line's controls -> ridge
  ContextMean*       the held-out line's own average effect over all its
                     perturbations: the challenge's 0 point. It reads the test
                     data, so it is a reference, not a method.

Metrics, per held-out line, with every test-panel target gene removed from
every vector: pds_cosine (as specified by VCC 2026, ranked within the fold's
panel), the expression-error ratio sum||pred - true||^2 / sum||true||^2 (the
spec's ratio without its finite-cell noise corrections) and Pearson delta.
95% CIs: bootstrap over perturbations.

Writes results/unseen_gene_unseen_context/{per_perturbation.csv, summary.csv,
summary.md, run.json}.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from baseline_first import metrics
from baseline_first.splits import assert_no_leakage
from baseline_first.vcc.transfer import (
    CoexpressionRidge,
    correlation_features,
    effects,
    variances,
)

SRC = Path("data/processed/replogle_nadig_pseudobulk.h5ad")
OUT = Path("results/unseen_gene_unseen_context")
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


def frame(line_perts: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(line_perts, columns=["cell_line", "perturbation"])


def score(method, line, fold, pred, true, panel_targets):
    keep = [g for g in true.columns if g not in panel_targets]
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


def main() -> None:
    commit = git_commit()
    bulk = ad.read_h5ad(SRC)
    lines = sorted(bulk.uns["moments"])
    genes = list(bulk.var_names)
    eff = {line: effects(bulk, line) for line in lines}
    moments = {line: bulk.uns["moments"][line] for line in lines}

    all_perts = sorted(set().union(*(e.index for e in eff.values())))
    rng = np.random.default_rng(SEED)
    fold_of = dict(
        zip(rng.permutation(all_perts), np.arange(len(all_perts)) % N_FOLDS, strict=True)
    )

    rows, alphas = [], []
    for line in lines:
        train_lines = [m for m in lines if m != line]
        # Anchors: the most variable genes in the training lines' control cells.
        pooled = sum(variances(moments[m]) for m in train_lines)
        anchors = np.sort(np.argsort(-pooled, kind="stable")[:N_ANCHORS])
        feats = {m: correlation_features(moments[m], genes, anchors) for m in lines}
        context_mean = eff[line].mean(axis=0)

        for fold in range(N_FOLDS):
            test_perts = [p for p in eff[line].index if fold_of[p] == fold]
            train = {
                m: eff[m].loc[[p for p in eff[m].index if fold_of[p] != fold]] for m in train_lines
            }
            assert_no_leakage(
                frame([(m, p) for m, e in train.items() for p in e.index]),
                frame([(line, p) for p in test_perts]),
                keys=["cell_line", "perturbation"],
            )
            true = eff[line].loc[test_perts]
            panel = set(test_perts)
            mean_response = pd.concat(train.values()).mean(axis=0)

            ridge = CoexpressionRidge(seed=SEED).fit([(feats[m], train[m]) for m in train_lines])
            alphas.append({"held_out": line, "fold": fold, "alpha": ridge.alpha_})
            preds = {
                "NoChange": true * 0,
                "MeanResponse": pd.DataFrame(
                    np.tile(mean_response.to_numpy(), (len(test_perts), 1)),
                    index=test_perts,
                    columns=genes,
                ),
                "CoexpressionRidge": ridge.predict(feats[line], test_perts),
                "ContextMean*": pd.DataFrame(
                    np.tile(context_mean.to_numpy(), (len(test_perts), 1)),
                    index=test_perts,
                    columns=genes,
                ),
            }
            for method, pred in preds.items():
                rows.append(score(method, line, fold, pred, true, panel))
            print(
                f"{line} fold {fold}: {len(test_perts)} perts, alpha {ridge.alpha_:g}", flush=True
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
                "ridge_alphas": alphas,
                "source": dict(bulk.uns.get("source", {})),
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print(text)


if __name__ == "__main__":
    main()
