"""Stream Arc's harmonised Replogle 2022 + Nadig 2025 CRISPRi screens into pseudobulks.

    python scripts/11_pseudobulk_replogle_nadig.py

Source: Hugging Face dataset arcinstitute/State-Replogle-Filtered, file
replogle_concat.h5ad (about 30 GB; 643,413 cells from K562, RPE1, HepG2 and
Jurkat; 2,024 perturbed genes; 6,546 measured genes). The file is read over
HTTP in row chunks and never stored. It takes about 100 minutes; partial sums
are checkpointed, so an interrupted run resumes where it stopped.

Output: data/processed/replogle_nadig_pseudobulk.h5ad, laid out as described
in `baseline_first.data.bulk`, including gene-gene moments of each cell
line's control cells (uns["moments"]) for co-expression features.
"""

from __future__ import annotations

from pathlib import Path

import aiohttp
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
# A whole local copy, if one has been downloaded. Reading over HTTP needs random
# access, so a connection that drops mid-chunk loses the chunk and re-reads `obs`
# on the next try; on a flaky link that can outpace the progress. Fetching the
# file once with a resuming download and reading it from disk is slower to start
# and far more predictable. 30,010,775,846 bytes.
LOCAL = Path("data/raw/state_replogle/replogle_concat.h5ad")


def open_source():
    """The local copy if it is complete, else the file over HTTP."""
    if LOCAL.exists() and LOCAL.stat().st_size == 30_010_775_846:
        print(f"reading the local copy at {LOCAL}", flush=True)
        return LOCAL
    print(f"streaming {URL}", flush=True)
    # Without a socket timeout a dropped connection leaves the read blocked for
    # ever rather than raising, so a rerun cannot resume: the whole point of the
    # checkpoint. `total=None` because the read as a whole takes hours; it is the
    # gap between bytes that signals a dead connection.
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=120)
    return fsspec.open(
        URL,
        block_size=32 * 2**20,
        cache_type="readahead",
        client_kwargs={"timeout": timeout},
    ).open()


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(open_source(), "r") as h5:
        bulk = pseudobulk(
            h5,
            pert_col="gene",
            cell_type_col="cell_line",
            checkpoint=CHECKPOINT,
            checkpoint_every=10,
            moments_of="non-targeting",
        )
    bulk.uns["source"] = {"url": URL, "repo": REPO, "file": FILE, "revision": REVISION}
    bulk.write_h5ad(OUT, compression="gzip")
    CHECKPOINT.unlink(missing_ok=True)
    print(f"wrote {OUT}: {bulk.n_obs} groups x {bulk.n_vars} genes")


if __name__ == "__main__":
    main()
