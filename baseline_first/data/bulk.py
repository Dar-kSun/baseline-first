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
    counts, ok = recover_counts(block)
    if not ok.all():
        raise ValueError("values are neither counts nor log1p(scaled counts)")
    return counts


MAX_SMALLEST_COUNT = 8


def recover_counts(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Like `to_counts`, but per cell: return (counts, ok), with ok False where a
    cell cannot be recovered exactly.

    A file that keeps only a subset of genes can leave a cell with no count of 1
    among them, so its smallest stored value stands for a count of 2 or more.
    Each such cell is retried assuming its smallest value is a count of k =
    2..MAX_SMALLEST_COUNT, accepting the smallest k that makes every value an
    integer within 1e-3. Rows that still fail come back as zeros with ok False.
    """
    if np.all(np.mod(block, 1) == 0):
        return block, np.ones(len(block), dtype=bool)
    scaled = np.expm1(block)
    unit = np.where(scaled > 0, scaled, np.inf).min(axis=1, keepdims=True)
    unit[~np.isfinite(unit)] = 1.0
    out = np.zeros_like(scaled)
    ok = np.zeros(len(block), dtype=bool)
    for k in range(1, MAX_SMALLEST_COUNT + 1):
        todo = ~ok
        if not todo.any():
            break
        counts = scaled[todo] / unit[todo] * k
        rounded = np.rint(counts)
        exact = np.abs(counts - rounded).max(axis=1) <= 1e-3
        rows = np.flatnonzero(todo)[exact]
        out[rows] = rounded[exact]
        ok[rows] = True
    return out, ok


def pseudobulk(
    h5: h5py.File,
    pert_col: str,
    cell_type_col: str | None = None,
    checkpoint: Path | None = None,
    chunk: int = CHUNK,
    log_every: int = 10,
    checkpoint_every: int = 10,
    moments_of: str | None = None,
    max_skip_fraction: float = 0.01,
) -> ad.AnnData:
    """Sum cells per (cell type, perturbation) group.

    With `checkpoint`, partial sums are saved every `checkpoint_every` chunks
    and a rerun resumes from the last saved row, so a long network read
    survives interruption.

    With `moments_of` (a perturbation label, normally the control), the cells
    carrying that label are also summarised per cell type as first and second
    moments of v = log1p(1e4 * counts / total), from which gene-gene
    covariances follow. They are stored in ``uns["moments"][cell_type]`` as
    ``n`` (cells), ``sum`` (genes) and ``outer`` (genes x genes, sum of v v^T).

    Cells whose counts cannot be recovered exactly (see `recover_counts`) are
    left out of every sum and counted in ``obs["n_skipped"]``; more than
    `max_skip_fraction` of cells skipped stops the run.
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
    types = sorted(set(map(str, cell_types)))
    type_index = pd.Index(types).get_indexer(pd.Series(cell_types).astype(str))
    in_moments = (perts == moments_of) if moments_of is not None else np.zeros(len(perts), bool)

    state = {
        "counts": np.zeros((len(groups), len(genes))),
        "cpm": np.zeros((len(groups), len(genes))),
        "skipped": np.zeros(len(groups)),
    }
    if moments_of is not None:
        state["m_n"] = np.zeros(len(types))
        state["m_sum"] = np.zeros((len(types), len(genes)))
        state["m_outer"] = np.zeros((len(types), len(genes), len(genes)))
    start = 0
    if checkpoint is not None and checkpoint.exists():
        saved = np.load(checkpoint)
        state = {k: saved[k] if k in saved.files else v for k, v in state.items()}
        start = int(saved["next_row"])
        print(f"resuming at row {start:,}")

    n, began = len(perts), time.time()
    for i, row in enumerate(range(start, n, chunk)):
        stop = min(row + chunk, n)
        block, ok = recover_counts(read_rows(X, row, stop))
        totals = np.where((t := block.sum(axis=1, keepdims=True)) > 0, t, 1)
        idx = inverse[row:stop]
        np.add.at(state["counts"], idx[ok], block[ok])
        np.add.at(state["cpm"], idx[ok], block[ok] / totals[ok] * 1e6)
        np.add.at(state["skipped"], idx[~ok], 1)
        if state["skipped"].sum() > max_skip_fraction * stop:
            raise ValueError(
                f"{int(state['skipped'].sum())} of the first {stop:,} cells could not be "
                "recovered as integer counts; refusing to continue"
            )
        if moments_of is not None and (in_moments[row:stop] & ok).any():
            sel = in_moments[row:stop] & ok
            v = np.log1p(block[sel] / totals[sel] * 1e4)
            for t_i in np.unique(type_index[row:stop][sel]):
                rows_t = type_index[row:stop][sel] == t_i
                state["m_n"][t_i] += rows_t.sum()
                state["m_sum"][t_i] += v[rows_t].sum(axis=0)
                state["m_outer"][t_i] += v[rows_t].T @ v[rows_t]
        last = stop == n
        if checkpoint is not None and ((i + 1) % checkpoint_every == 0 or last):
            np.savez(checkpoint, next_row=stop, **state)
        if (i + 1) % log_every == 0 or last:
            rate = (stop - start) / (time.time() - began)
            print(
                f"  {stop:,}/{n:,} cells, {(n - stop) / rate / 60:.0f} min left, "
                f"{int(state['skipped'].sum())} skipped",
                flush=True,
            )

    labels = [g.split("|", 1) for g in groups]
    out = ad.AnnData(
        obs=pd.DataFrame(
            {
                "cell_type": [c for c, _ in labels],
                "perturbation": [p for _, p in labels],
                "n_cells": np.bincount(inverse, minlength=len(groups)) - state["skipped"],
                "n_skipped": state["skipped"],
            },
            index=list(groups),
        ),
        var=pd.DataFrame(index=pd.Index(genes, name="gene_name")),
    )
    out.layers["counts_sum"] = state["counts"].astype(np.float32)
    out.layers["cpm_sum"] = state["cpm"].astype(np.float32)
    if moments_of is not None:
        out.uns["moments_of"] = moments_of
        out.uns["moments"] = {
            t: {
                "n": float(state["m_n"][k]),
                "sum": state["m_sum"][k],
                "outer": state["m_outer"][k],
            }
            for k, t in enumerate(types)
        }
    return out
