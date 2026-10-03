"""How well do perturbation effects transfer to a cell line the model has not seen?

    python scripts/12_cross_context_transfer.py

A small, local version of the VCC 2026 task, on public data. Arc's VCC 2025
support set has 53 CRISPRi perturbations measured in all four of K562, RPE1,
Jurkat and HepG2 (pseudobulks from scripts/10_pseudobulk_support_set.py).
Each cell line is held out in turn and its 53 effects are predicted from the
other three. As in the challenge, the held-out line's control cells are known.

Methods, all predicting the change from the held-out line's control profile:
  NoChange            zero effect (the control profile itself)
  MeanResponse        the average effect over all perturbations in the other lines
  SameGene            this gene's own effect, averaged over the other lines
  ContextMean*        the held-out line's own average effect: the challenge's 0
                      point. It reads the test data, so it is a reference, not
                      a method.

Profiles follow the VCC 2026 spec: counts summed per group, normalised to
5e4, log1p. Every one of the 53 target genes is removed from every vector.
Metrics: pds_cosine (as specified) and the expression-error ratio
sum ||pred - true||^2 / sum ||true||^2 over effects. The latter omits the
spec's finite-cell sampling corrections, which need single cells, so it
overstates the error of every method by the same noise term.

Writes results/cross_context/{per_perturbation.csv, summary.csv, summary.md}.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from baseline_first import metrics

LINES = ["k562", "rpe1", "jurkat", "hepg2"]
CONTROL = "non-targeting"
BULK_TARGET_SUM = 5e4
SRC = Path("data/processed/vcc2025")
OUT = Path("results/cross_context")


def effects(adata: ad.AnnData, perts: list[str]) -> pd.DataFrame:
    """log1p-normalised pseudobulk of each perturbation minus that of control."""
    obs = adata.obs.set_index("perturbation")
    counts = pd.DataFrame(adata.layers["counts_sum"], index=obs.index, columns=adata.var_names)
    profile = np.log1p(BULK_TARGET_SUM * counts.div(counts.sum(axis=1), axis=0))
    return profile.loc[perts] - profile.loc[CONTROL]


def score(name: str, line: str, pred: pd.DataFrame, true: pd.DataFrame) -> pd.DataFrame:
    p, t = pred.to_numpy(), true.to_numpy()
    return pd.DataFrame(
        {
            "method": name,
            "held_out": line,
            "perturbation": true.index,
            "pds": metrics.pds_cosine(p, t),
            "sq_error": ((p - t) ** 2).sum(axis=1),
            "sq_effect": (t**2).sum(axis=1),
            "pearson": metrics.pearson_delta(p, t, np.zeros(t.shape[1])),
        }
    )


def main() -> None:
    data = {line: ad.read_h5ad(SRC / f"{line}_pseudobulk.h5ad") for line in LINES}
    perts = sorted(
        set.intersection(*(set(a.obs["perturbation"]) for a in data.values())) - {CONTROL}
    )
    genes = [g for g in data[LINES[0]].var_names if g not in set(perts)]
    eff = {line: effects(a, perts)[genes] for line, a in data.items()}

    rows = []
    for line in LINES:
        others = [m for m in LINES if m != line]
        true = eff[line]
        same_gene = sum(eff[m] for m in others) / len(others)
        mean_response = pd.DataFrame(
            np.tile(same_gene.mean(axis=0).to_numpy(), (len(perts), 1)),
            index=perts,
            columns=genes,
        )
        context_mean = pd.DataFrame(
            np.tile(true.mean(axis=0).to_numpy(), (len(perts), 1)), index=perts, columns=genes
        )
        for name, pred in {
            "NoChange": true * 0,
            "MeanResponse": mean_response,
            "SameGene": same_gene,
            "ContextMean*": context_mean,
        }.items():
            rows.append(score(name, line, pred, true))
    results = pd.concat(rows, ignore_index=True)

    summary = []
    for (method, line), frame in results.groupby(["method", "held_out"], sort=False):
        pds, low, high = metrics.bootstrap_ci(frame["pds"], frame["perturbation"])
        summary.append(
            {
                "method": method,
                "held_out": line,
                "n_perturbations": len(frame),
                "pds": pds,
                "pds_ci_low": low,
                "pds_ci_high": high,
                "expr_error_ratio": frame["sq_error"].sum() / frame["sq_effect"].sum(),
                "pearson": frame["pearson"].mean(),
            }
        )
    summary = pd.DataFrame(summary)

    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / "per_perturbation.csv", index=False, float_format="%.6g")
    summary.to_csv(OUT / "summary.csv", index=False, float_format="%.6g")
    lines = [
        "| held out | method | PDS [95% CI] | expression error ratio | Pearson |",
        "|---|---|---|---|---|",
    ]
    for r in summary.sort_values(["held_out", "method"]).itertuples():
        lines.append(
            f"| {r.held_out} | {r.method} | {r.pds:.3f} [{r.pds_ci_low:.3f}, {r.pds_ci_high:.3f}] "
            f"| {r.expr_error_ratio:.3f} | {r.pearson:.3f} |"
        )
    table = "\n".join(lines) + "\n"
    (OUT / "summary.md").write_text(table)
    print(f"{len(perts)} shared perturbations, {len(genes)} genes after removing targets\n")
    print(table)


if __name__ == "__main__":
    main()
