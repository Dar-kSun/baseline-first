"""Norman et al. 2019: CRISPRa Perturb-seq of single and paired gene activations in K562.

Source: the GEARS preprocessed copy on Harvard Dataverse (CC0), see
``data/MANIFEST.md`` for the exact file, checksum and what was done upstream.
"""

from __future__ import annotations

import zipfile
from collections import Counter
from pathlib import Path

import anndata as ad
import numpy as np

from baseline_first.data._download import fetch
from baseline_first.data.schema import CONTROL

URL = "https://dataverse.harvard.edu/api/access/datafile/6154020"
MD5 = "cdc41d6050e619c37fd9dd44d440e2b4"
ARCHIVE = "norman.zip"
MEMBER = "norman/perturb_processed.h5ad"
CELL_LINE = "K562"

_CONTROL_TOKEN = "ctrl"


def canonical_perturbation(condition: str) -> str:
    """Map a GEARS condition label to the canonical perturbation label.

    GEARS writes single perturbations as either ``"KLF1+ctrl"`` or
    ``"ctrl+KLF1"`` (both appear in this dataset), pairs as ``"A+B"`` and
    control as ``"ctrl"``. Both single forms target the same gene, so they map
    to one label; otherwise a group split could put one in train and the other
    in test.
    """
    genes = sorted({g for g in condition.split("+") if g != _CONTROL_TOKEN})
    return "+".join(genes) if genes else CONTROL


def standardise(adata: ad.AnnData, source: dict) -> ad.AnnData:
    """Add the standard columns (see `baseline_first.data`) to a GEARS Norman AnnData."""
    obs = adata.obs
    conditions = obs["condition"].astype(str)
    mapping = {c: canonical_perturbation(c) for c in conditions.unique()}
    obs["perturbation"] = conditions.map(mapping).astype("category")
    obs["is_control"] = (obs["perturbation"] == CONTROL).to_numpy()
    obs["n_targets"] = np.array(
        [0 if p == CONTROL else p.count("+") + 1 for p in obs["perturbation"]], dtype=np.int8
    )
    # GEARS labels every cell "A549", but Norman et al. 2019 profiled K562 cells.
    obs["cell_line"] = CELL_LINE
    obs["cell_line"] = obs["cell_line"].astype("category")

    labels_per_perturbation = Counter(mapping.values())
    n_merged = sum(n > 1 for n in labels_per_perturbation.values())
    adata.uns["baseline_first"] = {
        **source,
        "dataset": "norman2019",
        "cell_line_corrected_from": "A549",
        "n_perturbations_merged": n_merged,
        "notes": (
            "X is log1p of library-size-normalised expression on 5045 genes, as provided "
            "by GEARS. 'perturbation' merges 'GENE+ctrl' and 'ctrl+GENE' into 'GENE'."
        ),
    }
    return adata


def load_norman2019(data_dir: str | Path = "data/raw", download: bool = True) -> ad.AnnData:
    """Load Norman 2019 with the standard columns, downloading (~170 MB zip) if needed.

    The extracted file is about 2.2 GB and is read fully into memory.
    """
    data_dir = Path(data_dir)
    h5ad = data_dir / MEMBER
    if not h5ad.exists():
        archive = data_dir / ARCHIVE
        if not archive.exists() and not download:
            raise FileNotFoundError(f"{archive} not found and download=False")
        fetch(URL, archive, MD5)
        with zipfile.ZipFile(archive) as zf:
            zf.extract(MEMBER, data_dir)
    adata = ad.read_h5ad(h5ad)
    return standardise(adata, {"url": URL, "md5": MD5, "file": MEMBER})
