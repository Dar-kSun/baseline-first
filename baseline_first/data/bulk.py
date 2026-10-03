"""Reduce single-cell h5ad/h5 files to per-group pseudobulks, chunk by chunk.

Works on any `h5py.File`, including one opened over HTTP with fsspec, so a
multi-GB file can be reduced without ever being stored. Only row chunks are
held in memory.

Each output row is one (cell type, perturbation) group:
  layers["counts_sum"]  summed raw counts
  layers["cpm_sum"]     summed per-cell counts-per-million (target 1e6, as the
                        VCC 2026 DE metrics normalise), so cpm_sum / n_cells is
                        the mean CPM that fold changes are defined on
  obs["n_cells"]
"""

from __future__ import annotations

import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd

CHUNK = 4096


def _decode(values) -> np.ndarray:
    return np.array([v.decode() if isinstance(v, bytes) else v for v in values], dtype=object)


def read_column(obs: h5py.Group, name: str) -> np.ndarray:
    """Read one obs column, decoding categoricals and byte strings."""
    node = obs[name]
    if isinstance(node, h5py.Group) and "categories" in node:  # categorical
        return _decode(node["categories"][:])[node["codes"][:]]
    if isinstance(node, h5py.Group):  # nullable string array (newer anndata)
        values = _decode(node["values"][:])
        if "mask" in node and node["mask"][:].any():
            raise ValueError(f"obs column {name!r} has missing values")
        return values
    return _decode(node[:])


def read_rows(X, start: int, stop: int) -> np.ndarray:
    """Rows start:stop of a dense or CSR-encoded X, as a dense float64 array."""
    if isinstance(X, h5py.Dataset):
        return X[start:stop].astype(np.float64)
    indptr = X["indptr"][start : stop + 1]
    data = X["data"][indptr[0] : indptr[-1]]
    indices = X["indices"][indptr[0] : indptr[-1]]
    dense = np.zeros((stop - start, X.attrs["shape"][1]), dtype=np.float64)
    dense[np.repeat(np.arange(stop - start), np.diff(indptr)), indices] = data
    return dense


def to_counts(block: np.ndarray) -> np.ndarray:
    """Return raw counts from a block of counts or of log1p(counts x per-cell factor).

    Several public files store log1p of counts scaled per cell. Undoing the log
    and dividing each cell by its smallest non-zero value (a count of 1) gives
    the counts back exactly. Anything that does not come back integral within
    1e-3 is refused rather than guessed at.
    """
    if np.all(np.mod(block, 1) == 0):
        return block
    scaled = np.expm1(block)
    unit = np.where(scaled > 0, scaled, np.inf).min(axis=1, keepdims=True)
    unit[~np.isfinite(unit)] = 1.0
    counts = scaled / unit
    rounded = np.rint(counts)
    if np.abs(counts - rounded).max() > 1e-3:
        raise ValueError("values are neither counts nor log1p(scaled counts)")
    return rounded


def pseudobulk(
    h5: h5py.File,
    pert_col: str,
    cell_type_col: str | None = None,
    checkpoint: Path | None = None,
    chunk: int = CHUNK,
    log_every: int = 10,
) -> ad.AnnData:
    """Sum cells per (cell type, perturbation) group.

    With `checkpoint`, partial sums are saved every `log_every` chunks and a
    rerun resumes from the last saved row, so a long network read survives
    interruption.
    """
    obs, X = h5["obs"], h5["X"]
    perts = read_column(obs, pert_col)
    cell_types = (
        read_column(obs, cell_type_col) if cell_type_col else np.full(len(perts), "", dtype=object)
    )
    var = h5["var"]
    genes = read_column(var, var.attrs.get("_index", "_index"))
    keys = pd.Series(cell_types).astype(str) + "|" + pd.Series(perts).astype(str)
    groups, inverse = np.unique(keys.to_numpy(), return_inverse=True)

    counts = np.zeros((len(groups), len(genes)))
    cpm = np.zeros_like(counts)
    start = 0
    if checkpoint is not None and checkpoint.exists():
        saved = np.load(checkpoint)
        counts, cpm, start = saved["counts"], saved["cpm"], int(saved["next_row"])
        print(f"resuming at row {start:,}")

    n, began = len(perts), time.time()
    for i, row in enumerate(range(start, n, chunk)):
        stop = min(row + chunk, n)
        block = to_counts(read_rows(X, row, stop))
        totals = block.sum(axis=1, keepdims=True)
        idx = inverse[row:stop]
        np.add.at(counts, idx, block)
        np.add.at(cpm, idx, block / np.where(totals > 0, totals, 1) * 1e6)
        if (i + 1) % log_every == 0 or stop == n:
            if checkpoint is not None:
                np.savez(checkpoint, counts=counts, cpm=cpm, next_row=stop)
            rate = (stop - start) / (time.time() - began)
            print(f"  {stop:,}/{n:,} cells, {(n - stop) / rate / 60:.0f} min left", flush=True)

    labels = [g.split("|", 1) for g in groups]
    out = ad.AnnData(
        obs=pd.DataFrame(
            {
                "cell_type": [c for c, _ in labels],
                "perturbation": [p for _, p in labels],
                "n_cells": np.bincount(inverse, minlength=len(groups)),
            },
            index=list(groups),
        ),
        var=pd.DataFrame(index=pd.Index(genes, name="gene_name")),
    )
    out.layers["counts_sum"] = counts.astype(np.float32)
    out.layers["cpm_sum"] = cpm.astype(np.float32)
    return out
