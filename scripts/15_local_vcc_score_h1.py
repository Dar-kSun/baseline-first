"""Score candidate VCC 2026 methods locally, with the official metric suite, on H1.

    python scripts/15_local_vcc_score_h1.py

H1 (the VCC 2025 CRISPRi data, 10x Flex) stands in for an unseen challenge
context. The local reference mirrors how the 2026 challenge builds its own:
a fixed random set of 60 perturbations with 400 cells each, plus 4,000 control cells,
downsampled to a median of 20,000 UMI per cell. Predictions are built from
those same control cells, as the challenge contexts provide them.

Methods train on K562, RPE1, HepG2 and Jurkat only, with every H1 panel gene
removed from their perturbations, so no target was ever seen perturbed (as in
the challenge); `assert_no_leakage` checks this.

Each prediction is scored with cell-eval2's `vcc2026` preset. The scaling
anchors are computed on the same reference with cell-eval2's own builders:
0 = `build_generic_baseline` (the context mean response, an oracle) and
1 = `compute_replicate_anchor` (split-half replicate, 5 splits). Scaled score
= (u - b) / (r - b), with the spec's clamps (expression error to [0, 1],
fold-change NMAE floored at -6); the overall score is the mean of the six.

Writes results/local_vcc_h1/{summary.md, scores.csv, run.json}; caches the
reference and per-method raw metrics under data/processed/local_vcc_h1/.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp

from baseline_first.data.bulk import read_column
from baseline_first.splits import assert_no_leakage
from baseline_first.vcc.submission import write_submission
from baseline_first.vcc.transfer import context_moments, fit_transfer, fold_changes, load_lines

H5 = Path("data/raw/vcc2025/competition_train.h5")
TRAIN = Path("data/processed/replogle_nadig_pseudobulk.h5ad")
CACHE = Path("data/processed/local_vcc_h1")
OUT = Path("results/local_vcc_h1")
CONTROL = "non-targeting"
# Sized so the 5-split replicate anchor fits in ~10 GB of free RAM (100 perturbations
# with 8,000 controls needed ~12 GB).
N_PERTS, CELLS_PER_PERT, N_CONTROLS, MEDIAN_UMI = 60, 400, 4000, 20_000
SEED = 0
RUNS = [("none", 1.0), ("mean", 1.0), ("ridge", 1.0), ("ridge", 0.3)]
MEMBERS = [
    "pds_cosine",
    "expr_mse_unbiased_capped_norm",
    "de_wilcoxon_direction_fidelity_yield_raw",
    "de_wilcoxon_direction_reach_raw",
    "de_wilcoxon_sig_jaccard",
    "de_wilcoxon_lfc_nmae",
]
CLAMPS = {"expr_mse_unbiased_capped_norm": (0.0, 1.0), "de_wilcoxon_lfc_nmae": (-6.0, np.inf)}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git_commit() -> str:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return sha.strip() + ("-dirty" if dirty.stdout.strip() else "")


def sparse_counts(block: sp.csr_matrix) -> sp.csr_matrix:
    """Raw counts from a CSR block of counts or of log1p(counts x per-cell factor)."""
    if np.all(np.mod(block.data, 1) == 0):
        return block
    scaled = block.copy()
    scaled.data = np.expm1(scaled.data)
    unit = np.array(
        [
            scaled.data[a:b].min() if b > a else 1.0
            for a, b in zip(scaled.indptr[:-1], scaled.indptr[1:], strict=True)
        ]
    )
    scaled.data = scaled.data / np.repeat(unit, np.diff(scaled.indptr))
    rounded = np.rint(scaled.data)
    if np.abs(scaled.data - rounded).max() > 1e-3:
        raise ValueError("H1 values are neither counts nor log1p(scaled counts)")
    scaled.data = rounded
    return scaled


def build_reference() -> ad.AnnData:
    path = CACHE / "reference.h5ad"
    if path.exists():
        return ad.read_h5ad(path)
    rng = np.random.default_rng(SEED)
    with h5py.File(H5, "r") as h5:
        obs, X = h5["obs"], h5["X"]
        perts = read_column(obs, "target_gene").astype(str)
        genes = read_column(h5["var"], h5["var"].attrs.get("_index", "_index")).astype(str)
        counts = pd.Series(perts).value_counts()
        eligible = sorted(p for p in counts.index if p != CONTROL and counts[p] >= CELLS_PER_PERT)
        panel = sorted(rng.choice(eligible, size=min(N_PERTS, len(eligible)), replace=False))
        chosen = []
        for p in panel:
            chosen.extend(rng.choice(np.flatnonzero(perts == p), CELLS_PER_PERT, replace=False))
        ctrl = np.flatnonzero(perts == CONTROL)
        chosen.extend(rng.choice(ctrl, min(N_CONTROLS, len(ctrl)), replace=False))
        rows = np.sort(np.asarray(chosen))

        indptr = X["indptr"][:]
        blocks, step = [], 16384
        for start in range(0, len(perts), step):
            stop = min(start + step, len(perts))
            want = rows[(rows >= start) & (rows < stop)]
            if not len(want):
                continue
            lo, hi = indptr[start], indptr[stop]
            block = sp.csr_matrix(
                (X["data"][lo:hi], X["indices"][lo:hi], indptr[start : stop + 1] - lo),
                shape=(stop - start, len(genes)),
            )
            blocks.append(sparse_counts(block[want - start]))
            log(f"read rows up to {stop:,}")
    counts_mat = sp.vstack(blocks).tocsr()
    totals = np.asarray(counts_mat.sum(axis=1)).ravel()
    keep_p = min(1.0, MEDIAN_UMI / np.median(totals))
    counts_mat.data = rng.binomial(counts_mat.data.astype(np.int64), keep_p).astype(np.float32)
    counts_mat.eliminate_zeros()
    ref = ad.AnnData(
        X=counts_mat,
        obs=pd.DataFrame({"target_gene": perts[rows]}, index=[f"c{i}" for i in rows]),
        var=pd.DataFrame(index=genes),
    )
    ref.uns["build"] = {"panel": panel, "thinning": keep_p, "seed": SEED}
    CACHE.mkdir(parents=True, exist_ok=True)
    ref.write_h5ad(path)
    return ref


def wide_mean(results: pl.DataFrame, cfg) -> dict:
    from cell_eval2.run import aggregate_metrics_wide, metric_output_names

    wide = aggregate_metrics_wide(results, metrics=metric_output_names(cfg))
    row = wide.filter(pl.col("statistic") == "mean").to_dicts()[0]
    return {m: float(row[m]) for m in MEMBERS}


def main() -> None:
    from cell_eval2 import EvalConfig, compute_metrics
    from cell_eval2.anchor import compute_replicate_anchor
    from cell_eval2.baseline import build_generic_baseline

    commit = git_commit()
    CACHE.mkdir(parents=True, exist_ok=True)
    cfg = replace(EvalConfig.from_preset("vcc2026"), pert_col="target_gene")

    log("building the local H1 reference")
    ref = build_reference()
    panel = list(ref.uns["build"]["panel"])
    genes_all = list(ref.var_names)
    controls = sp.csr_matrix(ref[ref.obs["target_gene"] == CONTROL].X)
    log(f"reference: {ref.n_obs:,} cells, {len(panel)} perturbations, {controls.shape[0]} controls")

    # Anchors, each cached as soon as it is computed.
    b_path, r_path = CACHE / "anchor_baseline.json", CACHE / "anchor_replicate.csv"
    if not b_path.exists():
        log("baseline anchor (context mean)")
        b_path.write_text(
            json.dumps(wide_mean(build_generic_baseline(ref, config=cfg).results, cfg))
        )
    if not r_path.exists():
        log("replicate anchor (5 splits)")
        compute_replicate_anchor(ref, config=cfg)[1].write_csv(r_path)
    replicate = pl.read_csv(r_path)
    r_rows = {d["metric"]: d for d in replicate.to_dicts()}
    anchors = {
        "baseline": json.loads(b_path.read_text()),
        "replicate": {m: float(r_rows[m]["replicate"]) for m in MEMBERS},
        "replicate_sd": {m: float(r_rows[m]["replicate_sd"]) for m in MEMBERS},
    }
    log(f"anchors: {anchors}")

    # Training data: the four public lines, with every H1 panel gene removed.
    genes, eff, moments = load_lines([TRAIN])
    keep = [i for i, g in enumerate(genes) if g in set(genes_all)]
    genes = [genes[i] for i in keep]
    excluded = {}
    for m in list(eff):
        hit = [p for p in eff[m].index if p in set(panel)]
        excluded[m] = len(hit)
        eff[m] = eff[m].drop(index=hit)[genes]
    moments = {
        m: {"n": v["n"], "sum": v["sum"][keep], "outer": v["outer"][np.ix_(keep, keep)]}
        for m, v in moments.items()
    }
    assert_no_leakage(
        pd.DataFrame({"perturbation": sorted(set().union(*(e.index for e in eff.values())))}),
        pd.DataFrame({"perturbation": panel}),
        keys="perturbation",
    )
    model = fit_transfer(genes, eff, moments, seed=SEED)
    log(
        f"model: alpha {model.ridge.alpha_:g}, "
        f"shrink ridge {model.scale_ridge:.3f}, mean {model.scale_mean:.3f}"
    )
    gene_pos = pd.Index(genes_all).get_indexer(genes)
    ctx_moments = context_moments(controls, gene_pos)

    raw = {}
    for method, depth in RUNS:
        tag = f"{method}-d{depth:g}"
        path = CACHE / f"{tag}.json"
        if path.exists():
            raw[tag] = json.loads(path.read_text())
            continue
        log(f"{tag}: building prediction")
        fc = fold_changes(model.predict(method, ctx_moments, panel), controls, genes_all)
        pred_path = CACHE / f"{tag}.h5ad"
        write_submission(pred_path, genes_all, {"H1": (controls, fc)}, seed=SEED, depth=depth)
        # The scorer requires the same labels on both sides, control included. Under the
        # vcc2026 preset (control_source=real) the control cells are taken from the real
        # side regardless, so the reference's own controls are appended unchanged.
        perturbed = ad.read_h5ad(pred_path)
        ctrl = ref[ref.obs["target_gene"] == CONTROL]
        pred = ad.AnnData(
            X=sp.vstack([perturbed.X.astype(np.float32), ctrl.X.astype(np.float32)]).tocsr(),
            obs=pd.DataFrame(
                {"target_gene": list(perturbed.obs["target_gene"]) + [CONTROL] * ctrl.n_obs},
                index=[f"p{i}" for i in range(perturbed.n_obs + ctrl.n_obs)],
            ),
            var=pd.DataFrame(index=genes_all),
        )
        log(f"{tag}: scoring")
        raw[tag] = wide_mean(compute_metrics(pred, ref, config=cfg), cfg)
        path.write_text(json.dumps(raw[tag], indent=2))
        pred_path.unlink()
        log(f"{tag}: {raw[tag]}")

    rows = []
    b, r = anchors["baseline"], anchors["replicate"]
    for tag, u in raw.items():
        row = {"run": tag}
        scaled = []
        for m in MEMBERS:
            s = (u[m] - b[m]) / (r[m] - b[m])
            lo, hi = CLAMPS.get(m, (-np.inf, np.inf))
            s = float(np.clip(s, lo, hi))
            row[f"{m}_raw"], row[f"{m}_scaled"] = u[m], s
            scaled.append(s)
        row["overall"] = float(np.mean(scaled))
        rows.append(row)
    scores = pd.DataFrame(rows)

    OUT.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUT / "scores.csv", index=False, float_format="%.6g")
    short = {
        "pds_cosine": "PDS",
        "expr_mse_unbiased_capped_norm": "MSE",
        "de_wilcoxon_direction_fidelity_yield_raw": "FID",
        "de_wilcoxon_direction_reach_raw": "Reach",
        "de_wilcoxon_sig_jaccard": "JAC",
        "de_wilcoxon_lfc_nmae": "NMAE",
    }
    head = "| run | " + " | ".join(short[m] for m in MEMBERS) + " | overall |"
    lines = [head, "|" + "---|" * (len(MEMBERS) + 2)]
    lines.append("| anchor b (0) raw | " + " | ".join(f"{b[m]:.3f}" for m in MEMBERS) + " | 0 |")
    lines.append("| anchor r (1) raw | " + " | ".join(f"{r[m]:.3f}" for m in MEMBERS) + " | 1 |")
    for row in rows:
        cells = " | ".join(f"{row[f'{m}_scaled']:.2f} ({row[f'{m}_raw']:.3f})" for m in MEMBERS)
        lines.append(f"| {row['run']} | {cells} | **{row['overall']:.2f}** |")
    text = "\n".join(lines) + "\n\nCells: scaled score (raw value).\n"
    (OUT / "summary.md").write_text(text)
    (OUT / "run.json").write_text(
        json.dumps(
            {
                "command": " ".join(["python", *sys.argv]),
                "git_commit": commit,
                "reference": {
                    "cells": int(ref.n_obs),
                    "perturbations": len(panel),
                    "controls": int(controls.shape[0]),
                    "thinning": ref.uns["build"]["thinning"],
                },
                "training_perts_excluded_as_panel_genes": excluded,
                "ridge_alpha": model.ridge.alpha_,
                "shrinkage": {"ridge": model.scale_ridge, "mean": model.scale_mean},
                "anchors": anchors,
                "seed": SEED,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print(text)


if __name__ == "__main__":
    main()
