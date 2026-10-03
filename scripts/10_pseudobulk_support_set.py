"""Reduce files from Arc's VCC 2025 support set to per-perturbation pseudobulks.

    python scripts/10_pseudobulk_support_set.py k562_gwps [--keep]

The support set (competition_support_set.zip, Arc's public bucket) holds
several multi-GB single-cell files. This extracts one member at a time
(zipfile verifies its CRC while reading), sums cells per (cell line,
perturbation) in row chunks, writes a small .h5ad to data/processed/vcc2025/,
and deletes the extracted file unless --keep is given. Disk use peaks at one
unpacked member.

The output layout is described in `baseline_first.data.bulk`. These files
store log1p(counts x size factor); `bulk.to_counts` recovers the counts exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

import fsspec
import h5py

from baseline_first.data.bulk import pseudobulk

URL = "https://storage.googleapis.com/vcc_data_prod/datasets/state/competition_support_set.zip"
MEMBERS = ["k562_gwps", "k562", "rpe1", "jurkat", "hepg2", "competition_train"]
RAW = Path("data/raw/vcc2025")
OUT = Path("data/processed/vcc2025")


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
    with h5py.File(path, "r") as h5:
        cell_type_col = "cell_type" if "cell_type" in h5["obs"] else None
        bulk = pseudobulk(h5, args.pert_col, cell_type_col)
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
