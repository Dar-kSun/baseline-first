import numpy as np
import pandas as pd
import pytest

from baseline_first.vcc.transfer import CoexpressionRidge, control_moments, correlation_features

N_MODULES, PER_MODULE = 4, 15
N_GENES = N_MODULES * PER_MODULE
GENES = [f"g{i}" for i in range(N_GENES)]
MODULE = np.repeat(np.arange(N_MODULES), PER_MODULE)


def cell_type(seed):
    """Control cells with module co-variation, and effects: knocking down a gene
    lowers its whole module (with a cell-type-specific strength)."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(800, N_MODULES))
    rates = np.exp(1.5 + 0.6 * factors[:, MODULE])
    controls = rng.poisson(rates)
    strength = rng.uniform(0.5, 1.0)
    eff = np.array([-strength * (MODULE == MODULE[g]) for g in range(N_GENES)], dtype=float)
    eff += rng.normal(scale=0.05, size=eff.shape)
    return controls, pd.DataFrame(eff, index=GENES, columns=GENES)


def test_ridge_predicts_unseen_genes_in_an_unseen_cell_type():
    lines = {name: cell_type(seed) for seed, name in enumerate(["a", "b", "c"])}
    anchors = np.arange(N_GENES)
    feats = {
        n: correlation_features(control_moments(c), GENES, anchors) for n, (c, _) in lines.items()
    }

    held_genes = [g for i, g in enumerate(GENES) if i % 5 == 0]
    train_genes = [g for g in GENES if g not in held_genes]
    training = [(feats[n], lines[n][1].loc[train_genes]) for n in ["a", "b"]]
    model = CoexpressionRidge(alphas=(0.1, 1.0, 10.0)).fit(training)

    true = lines["c"][1].loc[held_genes]
    pred = model.predict(feats["c"], held_genes)
    mean_response = np.tile(
        np.vstack([e.to_numpy() for _, e in training]).mean(axis=0), (len(held_genes), 1)
    )
    ridge_err = ((pred.to_numpy() - true.to_numpy()) ** 2).sum()
    mean_err = ((mean_response - true.to_numpy()) ** 2).sum()
    assert ridge_err < 0.3 * mean_err


def test_unknown_target_falls_back_to_the_intercept():
    controls, eff = cell_type(0)
    feats = correlation_features(control_moments(controls), GENES, np.arange(N_GENES))
    model = CoexpressionRidge(alphas=(1.0,)).fit([(feats, eff)])
    pred = model.predict(feats, ["NOT_A_GENE"])
    assert np.allclose(pred.to_numpy()[0], model.model_.intercept_)


def test_ridge_path_matches_sklearn():
    from sklearn.linear_model import Ridge

    from baseline_first.vcc.transfer import ridge_path

    rng = np.random.default_rng(0)
    X, Y, X_new = rng.normal(size=(40, 7)), rng.normal(size=(40, 3)), rng.normal(size=(5, 7))
    alphas = (0.1, 3.0, 100.0)
    for alpha, pred in zip(alphas, ridge_path(X, Y, X_new, alphas), strict=True):
        assert np.allclose(pred, Ridge(alpha=alpha).fit(X, Y).predict(X_new))


def test_optimal_scale_recovers_a_known_factor():
    from baseline_first.vcc.transfer import optimal_scale

    rng = np.random.default_rng(0)
    true = [rng.normal(size=(5, 4)) for _ in range(3)]
    pred = [t / 0.4 for t in true]  # predictions 2.5x too large
    assert np.isclose(optimal_scale(zip(pred, true, strict=True)), 0.4)


def test_fold_changes_map_effects_onto_the_full_axis():
    import scipy.sparse as sp

    from baseline_first.vcc.transfer import fold_changes

    genes_all = ["A", "B", "C", "D"]
    controls = sp.csr_matrix(np.array([[10.0, 5.0, 0.0, 2.0], [12.0, 3.0, 0.0, 2.0]]))
    effect = pd.DataFrame([[0.0, np.log(2.0), 0.3]], index=["D"], columns=["A", "B", "C"])
    fc = fold_changes(effect, controls, genes_all)
    assert fc.loc["D", "A"] == pytest.approx(1.0)
    assert fc.loc["D", "B"] == pytest.approx(2.0, rel=1e-3)  # log1p ~ log when well expressed
    assert fc.loc["D", "C"] == 1.0  # unexpressed in controls: left unchanged
    assert fc.loc["D", "D"] == 0.25  # the target itself: nominal knockdown
