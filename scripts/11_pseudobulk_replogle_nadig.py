"""Stream Arc's harmonised Replogle 2022 + Nadig 2025 CRISPRi screens into pseudobulks.

    python scripts/11_pseudobulk_replogle_nadig.py

Source: Hugging Face dataset arcinstitute/State-Replogle-Filtered, file
replogle_concat.h5ad (about 30 GB; 643,413 cells from K562, RPE1, HepG2 and
Jurkat; 2,024 perturbed genes; 6,546 measured genes). The file is read over
HTTP in row chunks and never stored. It takes about 100 minutes; partial sums
are checkpointed, so an interrupted run resumes where it stopped.

Output: data/processed/replogle_nadig_pseudobulk.h5ad, laid out as described
in `baseline_first.data.bulk`.
"""

from __future__ import annotations

from pathlib import Path

import fsspec
import h5py

from baseline_first.data.bulk import pseudobulk

REPO = "arcinstitute/State-Replogle-Filtered"
FILE = "replogle_concat.h5ad"
# The dataset commit inspected on 2026-10-04, pinned so reruns read the same bytes.
REVISION = "d790193bb2c93726541a75ca3fa873a92ed44da5"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{FILE}"
OUT = Path("data/processed/replogle_nadig_pseudobulk.h5ad")
CHECKPOINT = Path("data/processed/replogle_nadig_pseudobulk.partial.npz")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    remote = fsspec.open(URL, block_size=32 * 2**20, cache_type="readahead").open()
    with h5py.File(remote, "r") as h5:
        bulk = pseudobulk(h5, pert_col="gene", cell_type_col="cell_line", checkpoint=CHECKPOINT)
    bulk.uns["source"] = {"url": URL, "repo": REPO, "file": FILE, "revision": REVISION}
    bulk.write_h5ad(OUT, compression="gzip")
    CHECKPOINT.unlink(missing_ok=True)
    print(f"wrote {OUT}: {bulk.n_obs} groups x {bulk.n_vars} genes")


if __name__ == "__main__":
    main()
