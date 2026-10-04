"""Build a VCC 2026 validation-round submission file. It does NOT submit.

    python scripts/20_make_vcc_submission.py --method ridge [--depth 0.3]

Methods (scripts/14 and scripts/15 show how each scores on held-out public data):
  ridge  co-expression ridge + shrinkage: each target gene is described by its
         correlation with anchor genes in the context's own control cells; a
         ridge learned on public CRISPRi cell lines maps that to an effect,
         scaled by a factor fitted on held-out public lines.
  mean   the same scaled average effect for every target.
  none   no change: resampled control cells (a reference point).

Training data: K562, RPE1, HepG2, Jurkat (Replogle 2022 + Nadig 2025) and H1
(VCC 2025), on the genes they share with each other and the VCC gene axis.
Targets are never looked up, only described by co-expression.

`--depth` keeps each read with that probability. A full-depth file has about
2.1 billion stored values, which `vcc prep` cannot load in under ~25 GB of RAM.

Outputs data/submissions/arV1-<Method>[-d<depth>].h5ad and, after `vcc prep`,
.vcc, plus results/vcc_submission/<name>.json. Exits non-zero if prep fails.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from baseline_first.vcc.submission import write_submission
from baseline_first.vcc.transfer import context_moments, fit_transfer, fold_changes, load_lines

SOURCES = [
    Path("data/processed/replogle_nadig_pseudobulk.h5ad"),
    Path("data/processed/vcc2025/competition_train_pseudobulk.h5ad"),
]
VCC = Path("data/raw/vcc2026")
OUT = Path("data/submissions")
RECORD = Path("results/vcc_submission")
CONTEXTS = ["A", "B", "C"]
SEED = 0
NAMES = {"ridge": "arV1-CoexprRidge", "mean": "arV1-MeanResponse", "none": "arV1-NoChange"}


def git_commit() -> str:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return sha.strip() + ("-dirty" if dirty.stdout.strip() else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=sorted(NAMES), default="ridge")
    parser.add_argument("--depth", type=float, default=1.0, help="read-keeping probability")
    parser.add_argument("--no-prep", action="store_true", help="skip vcc prep")
    args = parser.parse_args()
    commit = git_commit()
    name = NAMES[args.method] + (f"-d{args.depth:g}" if args.depth < 1 else "")

    vcc_genes = pd.read_csv(VCC / "gene_names.csv")["gene_name"].astype(str).tolist()
    perts = pd.read_csv(VCC / "pert_counts.csv")["target_gene"].astype(str).tolist()
    genes, eff, moments = load_lines(SOURCES)
    keep = [i for i, g in enumerate(genes) if g in set(vcc_genes)]
    genes = [genes[i] for i in keep]
    eff = {m: e[genes] for m, e in eff.items()}
    moments = {
        m: {"n": v["n"], "sum": v["sum"][keep], "outer": v["outer"][np.ix_(keep, keep)]}
        for m, v in moments.items()
    }
    model = fit_transfer(genes, eff, moments, seed=SEED)
    print(
        f"trained on {sorted(eff)}; {len(genes)} genes; ridge alpha {model.ridge.alpha_:g}; "
        f"shrinkage ridge {model.scale_ridge:.3f}, mean {model.scale_mean:.3f}",
        flush=True,
    )

    gene_pos = pd.Index(vcc_genes).get_indexer(genes)
    contexts, record = {}, {}
    for c in CONTEXTS:
        cells = ad.read_h5ad(VCC / f"context_{c}.h5ad")
        if list(cells.var_names.astype(str)) != vcc_genes:
            raise ValueError(f"context {c} is not on the official gene axis")
        controls = sp.csr_matrix(cells.X)
        effect = model.predict(args.method, context_moments(controls, gene_pos), perts)
        fc = fold_changes(effect, controls, vcc_genes)
        contexts[c] = (controls, fc)
        shared = fc.to_numpy()[:, gene_pos]
        record[c] = {"median_abs_log2_fc": float(np.median(np.abs(np.log2(shared[shared > 0]))))}
        print(f"context {c}: fold changes ready", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    h5ad_path = OUT / f"{name}.h5ad"
    stats = write_submission(h5ad_path, vcc_genes, contexts, seed=SEED, depth=args.depth)
    print(f"wrote {h5ad_path}: {stats}", flush=True)

    prep = None
    if not args.no_prep:
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
                str(OUT / f"{name}.vcc"),
                "-f",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        prep = {"exit_code": result.returncode, "output": (result.stdout + result.stderr)[-3000:]}
        print(prep["output"])

    RECORD.mkdir(parents=True, exist_ok=True)
    (RECORD / f"{name}.json").write_text(
        json.dumps(
            {
                "model_name": name,
                "method": args.method,
                "depth": args.depth,
                "git_commit": commit,
                "training_lines": sorted(eff),
                "n_shared_genes": len(genes),
                "ridge_alpha": model.ridge.alpha_,
                "shrinkage_ridge": model.scale_ridge,
                "shrinkage_mean": model.scale_mean,
                "seed": SEED,
                "contexts": record,
                "submission": stats,
                "vcc_prep": prep,
            },
            indent=2,
        )
        + "\n"
    )
    if prep is not None and prep["exit_code"] != 0:
        sys.exit(f"vcc prep failed (exit {prep['exit_code']}); see {RECORD / (name + '.json')}")


if __name__ == "__main__":
    main()
