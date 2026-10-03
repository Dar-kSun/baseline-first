import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from baseline_first.data.bulk import pseudobulk, to_counts


def counts_matrix(seed=0, n_cells=60, n_genes=8):
    rng = np.random.default_rng(seed)
    return rng.poisson(3.0, size=(n_cells, n_genes)).astype(np.float64)


def test_to_counts_recovers_log1p_of_scaled_counts_exactly():
    counts = counts_matrix()
    counts[:, 0] = 1  # every cell has a count of exactly 1 somewhere
    factors = np.random.default_rng(1).uniform(0.5, 4.0, size=(len(counts), 1))
    stored = np.log1p(counts * factors).astype(np.float32).astype(np.float64)
    assert np.array_equal(to_counts(stored), counts)


def test_to_counts_passes_counts_through():
    counts = counts_matrix()
    assert np.array_equal(to_counts(counts), counts)


def test_to_counts_refuses_other_transforms():
    z_scores = np.random.default_rng(0).normal(size=(10, 8))
    with pytest.raises(ValueError):
        to_counts(np.abs(z_scores))


@pytest.mark.parametrize("sparse", [False, True])
def test_pseudobulk_sums_counts_and_cpm_per_group(tmp_path, sparse):
    counts = counts_matrix()
    obs = pd.DataFrame(
        {
            "gene": np.repeat(["A", "B", "non-targeting"], 20),
            "line": np.tile(["x", "y"], 30),
        },
        index=[f"c{i}" for i in range(60)],
    )
    path = tmp_path / "cells.h5ad"
    X = sp.csr_matrix(counts) if sparse else counts.astype(np.float32)
    ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(8)])).write_h5ad(path)

    with h5py.File(path, "r") as h5:
        bulk = pseudobulk(h5, pert_col="gene", cell_type_col="line", chunk=7)

    assert bulk.n_obs == 6 and list(bulk.var_names) == [f"g{i}" for i in range(8)]
    row = bulk.obs.index.get_loc("x|A")
    mask = ((obs["gene"] == "A") & (obs["line"] == "x")).to_numpy()
    assert bulk.obs["n_cells"].iloc[row] == mask.sum()
    assert np.allclose(bulk.layers["counts_sum"][row], counts[mask].sum(0))
    cpm = counts / counts.sum(1, keepdims=True) * 1e6
    assert np.allclose(bulk.layers["cpm_sum"][row], cpm[mask].sum(0), rtol=1e-5)


def test_pseudobulk_resumes_from_checkpoint(tmp_path):
    counts = counts_matrix(n_cells=50)
    obs = pd.DataFrame({"gene": np.repeat(["A", "B"], 25)}, index=[f"c{i}" for i in range(50)])
    path = tmp_path / "cells.h5ad"
    ad.AnnData(X=counts.astype(np.float32), obs=obs).write_h5ad(path)
    checkpoint = tmp_path / "partial.npz"

    with h5py.File(path, "r") as h5:
        full = pseudobulk(h5, pert_col="gene", chunk=10)
        # Simulate an interrupted run: sums for the first 20 rows only.
        first = counts[:20]
        idx = np.array([0] * 20)
        partial = np.zeros((2, counts.shape[1]))
        np.add.at(partial, idx, first)
        partial_cpm = np.zeros_like(partial)
        np.add.at(partial_cpm, idx, first / first.sum(1, keepdims=True) * 1e6)
        np.savez(checkpoint, counts=partial, cpm=partial_cpm, next_row=20)
        resumed = pseudobulk(h5, pert_col="gene", chunk=10, checkpoint=checkpoint)

    assert np.allclose(resumed.layers["counts_sum"], full.layers["counts_sum"])
    assert np.allclose(resumed.layers["cpm_sum"], full.layers["cpm_sum"], rtol=1e-5)


def test_control_moments_match_a_direct_computation(tmp_path):
    counts = counts_matrix(n_cells=60)
    obs = pd.DataFrame(
        {
            "gene": np.repeat(["A", "non-targeting", "non-targeting"], 20),
            "line": np.tile(["x", "y"], 30),
        },
        index=[f"c{i}" for i in range(60)],
    )
    path = tmp_path / "cells.h5ad"
    ad.AnnData(X=counts.astype(np.float32), obs=obs).write_h5ad(path)
    with h5py.File(path, "r") as h5:
        bulk = pseudobulk(h5, "gene", "line", chunk=7, moments_of="non-targeting")

    v = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4)
    for line in ["x", "y"]:
        mask = ((obs["gene"] == "non-targeting") & (obs["line"] == line)).to_numpy()
        m = bulk.uns["moments"][line]
        assert m["n"] == mask.sum()
        assert np.allclose(m["sum"], v[mask].sum(0))
        assert np.allclose(m["outer"], v[mask].T @ v[mask])
