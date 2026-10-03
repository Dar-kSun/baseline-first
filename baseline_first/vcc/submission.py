"""Turn predicted fold changes into a Virtual Cell Challenge 2026 submission file.

A submission holds single cells as raw integer counts: for each context and
each of its perturbations, a fixed number of cells (400) on the official
18,533-gene axis, labelled with `target_gene` and `context`, and no control
cells. Four of the six metrics run Wilcoxon tests on these cells, so they
must look like real cells, not copies of a mean profile.

`simulate_cells` builds them from the context's own control cells: each
predicted cell is a resampled control cell whose counts are scaled gene by
gene with exact integer arithmetic, so the expected mean moves by exactly the
predicted fold change:

- fold change f <= 1: binomial thinning, count ~ Binomial(c, f)
- fold change f > 1:  count = c + Poisson(c * (f - 1))

A gene with zero counts in the sampled control cell stays zero, which keeps
the expected mean exact (zeros contribute zero either way).

`write_submission` streams one perturbation at a time into a gzip-compressed
h5ad, so memory stays at one perturbation's cells regardless of panel size.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata.io import write_elem

CELLS_PER_PERT = 400
PERT_COL = "target_gene"
CONTEXT_COL = "context"


def simulate_cells(
    controls: sp.csr_matrix,
    fold_change: np.ndarray,
    n_cells: int,
    rng: np.random.Generator,
) -> sp.csr_matrix:
    """Resample `n_cells` control cells and scale each gene by `fold_change`."""
    if np.any(fold_change < 0) or not np.all(np.isfinite(fold_change)):
        raise ValueError("fold changes must be finite and non-negative")
    picked = controls[rng.integers(0, controls.shape[0], size=n_cells)].tocsr()
    picked.sort_indices()
    counts = picked.data.astype(np.int64)
    f = fold_change[picked.indices]
    down = f <= 1
    new = np.empty_like(counts)
    new[down] = rng.binomial(counts[down], f[down])
    new[~down] = counts[~down] + rng.poisson(counts[~down] * (f[~down] - 1))
    out = sp.csr_matrix((new.astype(np.int32), picked.indices, picked.indptr), shape=picked.shape)
    out.eliminate_zeros()
    return out


def _append(ds: h5py.Dataset, values: np.ndarray) -> None:
    start = ds.shape[0]
    ds.resize((start + len(values),))
    ds[start:] = values


def write_submission(
    path: Path,
    genes: Iterable[str],
    contexts: Mapping[str, tuple[sp.csr_matrix, pd.DataFrame]],
    n_cells: int = CELLS_PER_PERT,
    seed: int = 0,
) -> dict[str, int]:
    """Write a submission h5ad.

    `contexts` maps a context label ("A", "B", ...) to (control cells as a CSR
    count matrix on `genes`, fold changes as a perturbations x genes DataFrame).
    Returns cell and nonzero counts for the record.
    """
    genes = pd.Index(list(genes), name=None)
    n_genes = len(genes)
    labels: list[str] = []
    context_of: list[str] = []
    n_nonzero = 0
    with h5py.File(path, "w") as h5:
        X = h5.create_group("X")
        X.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0"})
        opts = {"maxshape": (None,), "chunks": (1 << 20,), "compression": "gzip"}
        data = X.create_dataset("data", shape=(0,), dtype=np.int32, **opts)
        indices = X.create_dataset("indices", shape=(0,), dtype=np.int32, **opts)
        indptr = [0]
        for c_index, (context, (controls, fold_changes)) in enumerate(contexts.items()):
            if list(fold_changes.columns) != list(genes):
                raise ValueError(f"context {context}: fold-change columns must match the gene axis")
            if controls.shape[1] != n_genes:
                raise ValueError(f"context {context}: control cells must be on the gene axis")
            for p_index, (pert, row) in enumerate(fold_changes.iterrows()):
                rng = np.random.default_rng([seed, c_index, p_index])
                cells = simulate_cells(controls, row.to_numpy(dtype=float), n_cells, rng)
                _append(data, cells.data)
                _append(indices, cells.indices.astype(np.int32))
                indptr.extend(indptr[-1] + cells.indptr[1:])
                labels.extend([pert] * n_cells)
                context_of.extend([context] * n_cells)
                n_nonzero += cells.nnz
        X.create_dataset("indptr", data=np.asarray(indptr, dtype=np.int64))
        X.attrs["shape"] = (len(labels), n_genes)
        obs = pd.DataFrame(
            {
                PERT_COL: pd.Categorical(labels),
                CONTEXT_COL: pd.Categorical(context_of),
            },
            index=pd.Index([f"cell_{i}" for i in range(len(labels))]),
        )
        write_elem(h5, "obs", obs)
        write_elem(h5, "var", pd.DataFrame(index=genes))
        h5.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
        for group in ("layers", "obsm", "obsp", "varm", "varp", "uns"):
            write_elem(h5, group, {})
    return {"n_cells": len(labels), "n_nonzero": n_nonzero}


def read_submission(path: Path) -> ad.AnnData:
    """Read a submission back (for checks and local scoring)."""
    return ad.read_h5ad(path)
