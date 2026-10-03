"""Reduce files from Arc's VCC 2025 support set to per-perturbation pseudobulks.

    python scripts/10_pseudobulk_support_set.py k562_gwps [--keep]

The support set (competition_support_set.zip, Arc's public bucket) holds
several multi-GB single-cell files. This extracts one member at a time
(zipfile verifies its CRC while reading), sums cells per (cell line,
perturbation) in row chunks, writes a small .h5ad to data/processed/vcc2025/,
and deletes the extracted file unless --keep is given. Disk use peaks at one
unpacked member.

Each output row is one (cell_type, perturbation) group:
  layers["counts_sum"]  summed raw counts (recovered exactly from log1p(scaled
                        counts) where a file stores those; see to_counts)
  layers["cpm_sum"]     summed per-cell counts-per-million (1e6 target, as the
                        VCC 2026 DE metrics normalise), so cpm_sum / n_cells
                        is the mean CPM the fold changes are defined on
  obs["n_cells"]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

import anndata as ad
import fsspec
import h5py
import numpy as np
import pandas as pd

URL = "https://storage.googleapis.com/vcc_data_prod/datasets/state/competition_support_set.zip"
MEMBERS = ["k562_gwps", "k562", "rpe1", "jurkat", "hepg2", "competition_train"]
RAW = Path("data/raw/vcc2025")
OUT = Path("data/processed/vcc2025")
CHUNK = 4096


def extract(member: str) -> Path:
    dest = RAW / f"{member}.h5"
    if dest.exists():
        return dest
    RAW.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(fsspec.open(URL, block_size=8 * 2**20, cache_type="readahead").open())
    part = dest.with_suffix(".h5.part")
    with z.open(f"competition_support_set/{member}.h5") as src, open(part, "wb") as dst:
        shutil.copyfileobj(src, dst, length=16 * 2**20)
    part.replace(dest)
    return dest


def _decode(values) -> np.ndarray:
    return np.array([v.decode() if isinstance(v, bytes) else v for v in values], dtype=object)


def _column(obs: h5py.Group, name: str) -> np.ndarray:
    node = obs[name]
    if isinstance(node, h5py.Group):  # categorical
        return _decode(node["categories"][:])[node["codes"][:]]
    return _decode(node[:])


def _rows(X, start: int, stop: int) -> np.ndarray:
    if isinstance(X, h5py.Dataset):
        return X[start:stop]
    indptr = X["indptr"][start : stop + 1]
    data = X["data"][indptr[0] : indptr[-1]]
    indices = X["indices"][indptr[0] : indptr[-1]]
    n_genes = X.attrs["shape"][1]
    dense = np.zeros((stop - start, n_genes), dtype=np.float64)
    rows = np.repeat(np.arange(stop - start), np.diff(indptr))
    dense[rows, indices] = data
    return dense


def to_counts(block: np.ndarray) -> np.ndarray:
    """Return raw counts from a block that holds counts, or log1p(counts * size factor).

    The support-set files store log1p of counts scaled per cell. Undoing the log
    and dividing each cell by its smallest non-zero value (a count of 1) gives
    the counts back exactly; anything that does not come back integral within
    1e-3 is refused rather than guessed at.
    """
    if np.all(np.mod(block, 1) == 0):
        return block
    scaled = np.expm1(block)
    positive = np.where(scaled > 0, scaled, np.inf)
    unit = positive.min(axis=1, keepdims=True)
    unit[~np.isfinite(unit)] = 1.0
    counts = scaled / unit
    rounded = np.rint(counts)
    if np.abs(counts - rounded).max() > 1e-3:
        raise ValueError("values are neither counts nor log1p(scaled counts)")
    return rounded


def pseudobulk(path: Path, pert_col: str = "target_gene") -> ad.AnnData:
    with h5py.File(path, "r") as h:
        obs, X = h["obs"], h["X"]
        perts = _column(obs, pert_col)
        cell_type = _column(obs, "cell_type") if "cell_type" in obs else np.full(len(perts), "")
        genes = _decode(h["var"][h["var"].attrs.get("_index", "_index")][:])
        keys = pd.Index(pd.Series(cell_type).astype(str) + "|" + pd.Series(perts).astype(str))
        groups, inverse = np.unique(keys, return_inverse=True)
        counts = np.zeros((len(groups), len(genes)))
        cpm = np.zeros_like(counts)
        for start in range(0, len(perts), CHUNK):
            stop = min(start + CHUNK, len(perts))
            block = to_counts(_rows(X, start, stop).astype(np.float64))
            totals = block.sum(axis=1, keepdims=True)
            idx = inverse[start:stop]
            np.add.at(counts, idx, block)
            np.add.at(cpm, idx, block / np.where(totals > 0, totals, 1) * 1e6)
    cell_types, labels = zip(*(g.split("|", 1) for g in groups), strict=True)
    out = ad.AnnData(
        X=None,
        obs=pd.DataFrame(
            {
                "cell_type": list(cell_types),
                "perturbation": list(labels),
                "n_cells": np.bincount(inverse, minlength=len(groups)),
            },
            index=list(groups),
        ),
        var=pd.DataFrame(index=pd.Index(genes, name="gene_name")),
    )
    out.layers["counts_sum"] = counts.astype(np.float32)
    out.layers["cpm_sum"] = cpm.astype(np.float32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("member", choices=MEMBERS)
    parser.add_argument("--pert-col", default="target_gene")
    parser.add_argument("--keep", action="store_true", help="keep the extracted file")
    args = parser.parse_args()

    path = extract(args.member)
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            md5.update(chunk)
    bulk = pseudobulk(path, args.pert_col)
    bulk.uns["source"] = {
        "url": URL,
        "member": f"competition_support_set/{args.member}.h5",
        "member_md5": md5.hexdigest(),
        "script": "scripts/10_pseudobulk_support_set.py",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{args.member}_pseudobulk.h5ad"
    bulk.write_h5ad(out, compression="gzip")
    print(f"wrote {out}: {bulk.n_obs} groups x {bulk.n_vars} genes, md5 {md5.hexdigest()}")
    if not args.keep:
        path.unlink()


if __name__ == "__main__":
    main()
