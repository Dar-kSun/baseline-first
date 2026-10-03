import os
import shutil
import subprocess

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from baseline_first.vcc.submission import simulate_cells, write_submission


def control_counts(n_cells=300, n_genes=12, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.uniform(0.5, 20, size=n_genes)
    return sp.csr_matrix(rng.poisson(means, size=(n_cells, n_genes)).astype(np.float32))


def test_simulated_cells_shift_the_mean_by_the_fold_change():
    controls = control_counts()
    fold = np.linspace(0.1, 3.0, controls.shape[1])
    cells = simulate_cells(controls, fold, n_cells=40_000, rng=np.random.default_rng(1))
    observed = np.asarray(cells.mean(axis=0)).ravel() / np.asarray(controls.mean(axis=0)).ravel()
    assert np.allclose(observed, fold, rtol=0.03)
    assert cells.dtype == np.int32
    assert (cells.data > 0).all()


def test_fold_change_of_one_keeps_cells_unchanged():
    controls = control_counts()
    rng = np.random.default_rng(3)
    cells = simulate_cells(controls, np.ones(controls.shape[1]), n_cells=50, rng=rng)
    picks = np.random.default_rng(3).integers(0, controls.shape[0], size=50)
    assert np.array_equal(cells.toarray(), controls[picks].toarray().astype(np.int32))


def test_invalid_fold_changes_are_refused():
    with pytest.raises(ValueError):
        simulate_cells(control_counts(), -np.ones(12), 5, np.random.default_rng(0))


def small_submission(tmp_path, n_cells=6):
    genes = [f"G{i}" for i in range(12)]
    contexts = {}
    for c in "AB":
        perts = ["G1", "G2", "G3"]
        fc = pd.DataFrame(np.ones((3, 12)), index=perts, columns=genes)
        fc.loc["G1", "G1"] = 0.2
        contexts[c] = (control_counts(seed=ord(c)), fc)
    path = tmp_path / "pred.h5ad"
    stats = write_submission(path, genes, contexts, n_cells=n_cells, seed=0)
    return path, genes, stats


def test_submission_reads_back_as_valid_anndata(tmp_path):
    path, genes, stats = small_submission(tmp_path)
    adata = ad.read_h5ad(path)
    assert adata.shape == (2 * 3 * 6, 12) == (stats["n_cells"], 12)
    assert list(adata.var_names) == genes
    assert adata.obs.groupby(["context", "target_gene"], observed=True).size().eq(6).all()
    assert np.issubdtype(adata.X.dtype, np.integer)
    assert adata.X.nnz == stats["n_nonzero"]


def test_submission_is_reproducible(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = ad.read_h5ad(small_submission(tmp_path / "a")[0])
    b = ad.read_h5ad(small_submission(tmp_path / "b")[0])
    assert (a.X != b.X).nnz == 0


@pytest.mark.skipif(shutil.which("vcc") is None, reason="vcc CLI not installed")
def test_official_validator_accepts_the_format(tmp_path):
    """`vcc prep --dry-run` is the challenge's own local validator."""
    path, genes, _ = small_submission(tmp_path, n_cells=6)
    genes_csv = tmp_path / "genes.csv"
    pd.Series(genes, name="gene_name").to_csv(genes_csv, index=False)
    result = subprocess.run(
        [
            "vcc",
            "prep",
            str(path),
            "-g",
            str(genes_csv),
            "--no-verify-targets",
            "--no-check-cell-counts",
            "--contexts",
            "A,B",
            "--expected-gene-dim",
            "-1",
            "-o",
            str(tmp_path / "pred.vcc"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
