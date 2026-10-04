import numpy as np
import pytest
import scipy.sparse as sp

from baseline_first.effective_n import anova_icc, effective_n


def random_effects(k=60, m=50, n_genes=40, sigma_b=1.0, sigma_w=1.0, seed=0):
    """k conditions of m cells: x = condition effect (sd sigma_b) + noise (sd sigma_w)."""
    rng = np.random.default_rng(seed)
    effects = rng.normal(scale=sigma_b, size=(k, n_genes))
    groups = np.repeat(np.arange(k), m)
    X = effects[groups] + rng.normal(scale=sigma_w, size=(k * m, n_genes))
    return X, groups


@pytest.mark.parametrize(("sigma_b", "expected"), [(1.0, 0.5), (2.0, 0.8), (0.5, 0.2)])
def test_icc_recovers_the_variance_share(sigma_b, expected):
    X, groups = random_effects(sigma_b=sigma_b)
    icc, m0 = anova_icc(X, groups)
    assert m0 == pytest.approx(50)
    assert np.median(icc) == pytest.approx(expected, abs=0.06)


def test_no_condition_effect_means_every_cell_counts():
    X, groups = random_effects(sigma_b=0.0)
    result = effective_n(X, groups)
    assert result.icc_median < 0.02
    assert result.inflation < 1.5


def test_pure_replicates_collapse_to_the_number_of_conditions():
    X, groups = random_effects(sigma_b=5.0, sigma_w=0.01)
    result = effective_n(X, groups)
    assert result.n_eff == pytest.approx(60, rel=0.01)
    assert result.inflation == pytest.approx(50, rel=0.01)


def test_unequal_group_sizes_use_the_anova_constant():
    groups = np.repeat(np.arange(3), [10, 20, 30])
    X = np.random.default_rng(0).normal(size=(60, 5))
    _, m0 = anova_icc(X, groups)
    n = 60
    assert m0 == pytest.approx((n - (100 + 400 + 900) / n) / 2)


def test_sparse_and_dense_agree():
    X, groups = random_effects()
    X = np.clip(X, 0, None)
    a = effective_n(X, groups)
    b = effective_n(sp.csr_matrix(X), groups)
    assert a.n_eff == pytest.approx(b.n_eff)


def test_needs_replicates_and_two_conditions():
    with pytest.raises(ValueError):
        anova_icc(np.ones((4, 2)), ["a"] * 4)
    with pytest.raises(ValueError):
        anova_icc(np.ones((3, 2)), ["a", "b", "c"])
