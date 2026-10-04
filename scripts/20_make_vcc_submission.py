"""Build a VCC 2026 validation-round submission file. It does NOT submit.

    python scripts/20_make_vcc_submission.py --method ridge

Methods (see scripts/14 for how each scores on held-out public cell lines):
  ridge  CoexpressionRidge+shrink: each target gene is described by its
         correlation with anchor genes in the context's own control cells; a
         ridge learned on five public CRISPRi cell lines maps that to an
         effect, scaled by a factor fitted on held-out public lines.
  mean   MeanResponse+shrink: the same scaled average effect for every target.

Training data: K562, RPE1, HepG2, Jurkat (Replogle 2022 + Nadig 2025) and H1
(VCC 2025), on the genes they share with each other and the VCC gene axis.
None of the 300 VCC targets is perturbed in these data except 13 in H1 (see
data/MANIFEST.md); targets are never looked up, only described by co-expression.

Each predicted effect (a change in log1p(5e4-normalised) profile) becomes a
per-gene fold change on the context's control profile; genes outside the
shared set keep fold change 1. The target gene itself is set to a 0.25 fold
change as a nominal CRISPRi knockdown; every VCC metric excludes target genes,
so this value does not affect the score. Cells are then simulated from the
context's control cells (baseline_first.vcc.submission).

Outputs data/submissions/arV1-<Method>.h5ad and, after `vcc prep`, .vcc,
plus results/vcc_submission/<method>.json describing the build.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from baseline_first.vcc.submission import write_submission
from baseline_first.vcc.transfer import (
    BULK_TARGET_SUM,
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
VCC = Path("data/raw/vcc2026")
OUT = Path("data/submissions")
RECORD = Path("results/vcc_submission")
CONTEXTS = ["A", "B", "C"]
N_ANCHORS = 1000
N_FOLDS = 5
TARGET_FOLD_CHANGE = 0.25
MAX_FOLD_CHANGE = 10.0
SEED = 0
NAMES = {"ridge": "arV1-CoexprRidge", "mean": "arV1-MeanResponse"}


def git_commit() -> str:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return sha.strip() + ("-dirty" if dirty.stdout.strip() else "")


def context_moments(cells: sp.csr_matrix, columns: np.ndarray) -> dict:
    """Moments of log1p(1e4 * counts / total) on `columns`, total over all genes."""
    totals = np.asarray(cells.sum(axis=1)).ravel()
    scaled = sp.diags(1e4 / np.where(totals > 0, totals, 1)) @ cells
    v = np.log1p(scaled[:, columns].toarray())
    return {"n": float(len(v)), "sum": v.sum(axis=0), "outer": v.T @ v}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--method", choices=sorted(NAMES), default="ridge")
    parser.add_argument("--no-prep", action="store_true", help="skip vcc prep")
    args = parser.parse_args()
    commit = git_commit()
    name = NAMES[args.method]

    vcc_genes = pd.read_csv(VCC / "gene_names.csv")["gene_name"].astype(str).tolist()
    perts = pd.read_csv(VCC / "pert_counts.csv")["target_gene"].astype(str).tolist()
    genes, eff, moments = load_lines(SOURCES)
    in_vcc = set(vcc_genes)
    keep = [i for i, g in enumerate(genes) if g in in_vcc]
    genes = [genes[i] for i in keep]
    eff = {m: e[genes] for m, e in eff.items()}
    moments = {
        m: {"n": v["n"], "sum": v["sum"][keep], "outer": v["outer"][np.ix_(keep, keep)]}
        for m, v in moments.items()
    }
    lines = sorted(eff)
    print(f"training on {lines}, {len(genes)} genes shared with the VCC axis")

    pooled = sum(variances(moments[m]) for m in lines)
    anchors = np.sort(np.argsort(-pooled, kind="stable")[:N_ANCHORS])
    feats = {m: correlation_features(moments[m], genes, anchors) for m in lines}
    model = CoexpressionRidge(seed=SEED).fit([(feats[m], eff[m]) for m in lines])
    mean_response = pd.concat(eff.values()).mean(axis=0)

    # Shrinkage factor, fitted as in scripts/14: each line held out in turn,
    # predicted from the others on a gene fold they did not train on.
    all_perts = sorted(set().union(*(e.index for e in eff.values())))
    rng = np.random.default_rng(SEED)
    fold_of = dict(
        zip(rng.permutation(all_perts), np.arange(len(all_perts)) % N_FOLDS, strict=True)
    )
    pairs = {"ridge": [], "mean": []}
    for held in lines:
        rest = [m for m in lines if m != held]
        inner = {m: eff[m].loc[[p for p in eff[m].index if fold_of[p] != 0]] for m in rest}
        target = eff[held].loc[[p for p in eff[held].index if fold_of[p] == 0]]
        inner_model = CoexpressionRidge(alphas=(model.alpha_,), n_inner_folds=2).fit(
            [(feats[m], inner[m]) for m in rest]
        )
        pairs["ridge"].append(
            (inner_model.predict(feats[held], target.index).to_numpy(), target.to_numpy())
        )
        inner_mean = pd.concat(inner.values()).mean(axis=0).to_numpy()
        pairs["mean"].append((np.tile(inner_mean, (len(target), 1)), target.to_numpy()))
    scale = optimal_scale(pairs[args.method])
    print(f"ridge alpha {model.alpha_:g}, shrinkage factor {scale:.3f}")

    gene_pos = pd.Index(vcc_genes).get_indexer(genes)
    contexts, record = {}, {}
    for c in CONTEXTS:
        cells = ad.read_h5ad(VCC / f"context_{c}.h5ad")
        if list(cells.var_names.astype(str)) != vcc_genes:
            raise ValueError(f"context {c} is not on the official gene axis")
        controls = sp.csr_matrix(cells.X)
        c_feats = correlation_features(context_moments(controls, gene_pos), genes, anchors)
        if args.method == "ridge":
            effect = model.predict(c_feats, perts) * scale
        else:
            effect = pd.DataFrame(
                np.tile(mean_response.to_numpy() * scale, (len(perts), 1)),
                index=perts,
                columns=genes,
            )
        summed = np.asarray(controls.sum(axis=0)).ravel()
        ctrl_profile = np.log1p(BULK_TARGET_SUM * summed / summed.sum())[gene_pos]
        with np.errstate(invalid="ignore", divide="ignore"):
            fc_shared = np.expm1(ctrl_profile + effect.to_numpy()) / np.expm1(ctrl_profile)
        fc_shared = np.where(ctrl_profile > 0, fc_shared, 1.0)
        fc_shared = np.clip(np.nan_to_num(fc_shared, nan=1.0), 0.0, MAX_FOLD_CHANGE)
        fold_change = np.ones((len(perts), len(vcc_genes)))
        fold_change[:, gene_pos] = fc_shared
        target_pos = pd.Index(vcc_genes).get_indexer(perts)
        fold_change[np.arange(len(perts)), target_pos] = TARGET_FOLD_CHANGE
        contexts[c] = (controls, pd.DataFrame(fold_change, index=perts, columns=vcc_genes))
        record[c] = {
            "median_abs_log2_fc": float(np.median(np.abs(np.log2(fc_shared[fc_shared > 0])))),
            "targets_with_features": int(sum(p in set(genes) for p in perts)),
        }
        print(f"context {c}: fold changes ready", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    h5ad_path = OUT / f"{name}.h5ad"
    stats = write_submission(h5ad_path, vcc_genes, contexts, seed=SEED)
    print(f"wrote {h5ad_path}: {stats}", flush=True)

    prep = None
    if not args.no_prep:
        vcc_path = OUT / f"{name}.vcc"
        result = subprocess.run(
            [
                "vcc",
                "prep",
                str(h5ad_path),
                "-g",
                str(VCC / "gene_names.csv"),
                "--perts",
                str(VCC / "pert_counts.csv"),
                "-o",
                str(vcc_path),
                "-f",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        prep = {"exit_code": result.returncode, "output": (result.stdout + result.stderr)[-3000:]}
        print(prep["output"])

    RECORD.mkdir(parents=True, exist_ok=True)
    (RECORD / f"{args.method}.json").write_text(
        json.dumps(
            {
                "model_name": name,
                "method": args.method,
                "git_commit": commit,
                "training_lines": lines,
                "n_shared_genes": len(genes),
                "ridge_alpha": model.alpha_,
                "shrinkage_factor": scale,
                "anchors": N_ANCHORS,
                "seed": SEED,
                "contexts": record,
                "submission": stats,
                "vcc_prep": prep,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
