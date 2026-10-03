import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

from baseline_first.baselines import GlobalMean, HVGRidge
from baseline_first.data import CONTROL
from baseline_first.metrics import mae
from baseline_first.pseudobulk import group_means

FIXTURE = Path(__file__).parent / "fixtures" / "norman2019_mini.h5ad"


def modular_data(seed=0, n_modules=3, genes_per_module=10, shift=2.0):
    """Genes in co-varying modules; perturbing a gene raises its whole module.

    A held-out gene's module can be read from its co-expression in training
    cells, so a baseline that uses co-expression should beat the global mean.
    """
    rng = np.random.default_rng(seed)
    n_genes = n_modules * genes_per_module
    module = np.repeat(np.arange(n_modules), genes_per_module)
    genes = [f"g{i}" for i in range(n_genes)]

    def cells(n, raised_module=None):
        factors = rng.normal(size=(n, n_modules))
        x = 3.0 + factors[:, module] + rng.normal(scale=0.5, size=(n, n_genes))
        if raised_module is not None:
            x[:, module == raised_module] += shift
        return x

    blocks, labels = [cells(300)], [CONTROL] * 300
    for g in range(n_genes):
        blocks.append(cells(40, raised_module=module[g]))
        labels += [genes[g]] * 40
    return np.vstack(blocks), np.array(labels), genes


def test_pseudobulk_group_means():
    X = sp.csr_matrix(np.array([[1.0, 0.0], [3.0, 2.0], [10.0, 10.0]]))
    groups, means = group_means(X, ["a", "a", "b"])
    assert list(groups) == ["a", "b"]
    assert np.allclose(means, [[2.0, 1.0], [10.0, 10.0]])


def test_global_mean_weights_perturbations_equally():
    X = np.array([[0.0], [0.0], [0.0], [10.0], [4.0]])
    labels = np.array(["a", "a", "a", "b", CONTROL])
    model = GlobalMean()
    model.fit(X, labels, ["g"])
    assert np.allclose(model.predict(["x", "y"]), [[5.0], [5.0]])


def test_hvg_ridge_beats_global_mean_on_learnable_structure():
    X, labels, genes = modular_data()
    held_out = ["g0", "g10", "g20"]
    train = ~np.isin(labels, held_out)
    groups, observed = group_means(X[~train], labels[~train])

    scores = {}
    for model in (GlobalMean(), HVGRidge(k=30)):
        model.fit(X[train], labels[train], genes)
        scores[model.name] = mae(model.predict(groups), observed).mean()
    assert scores["HVGRidge"] < 0.5 * scores["GlobalMean"]


def test_hvg_ridge_falls_back_to_intercept_for_unmeasured_targets():
    X, labels, genes = modular_data()
    model = HVGRidge(k=30)
    model.fit(X, labels, genes)
    pred = model.predict(["NOT_MEASURED"])
    assert model.n_unmeasured_ == 1
    assert np.allclose(pred[0], model.control_ + model.ridge_.intercept_)


def test_baselines_run_fast_on_fixture():
    adata = ad.read_h5ad(FIXTURE)
    labels = adata.obs["perturbation"].astype(str).to_numpy()
    genes = adata.var["gene_name"].astype(str).to_numpy()
    queries = ["ETS2", "FOXF1+HOXB9"]
    for model in (GlobalMean(), HVGRidge()):
        start = time.perf_counter()
        model.fit(adata.X, labels, genes)
        pred = model.predict(queries)
        assert time.perf_counter() - start < 10
        assert pred.shape == (2, adata.n_vars)
        assert np.isfinite(pred).all()
