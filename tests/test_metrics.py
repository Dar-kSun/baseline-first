import numpy as np
import pytest

from baseline_first import metrics


def test_mae_and_rmse_per_row():
    pred = np.array([[0.0, 0.0], [1.0, 3.0]])
    true = np.array([[1.0, -1.0], [1.0, 1.0]])
    assert np.allclose(metrics.mae(pred, true), [1.0, 1.0])
    assert np.allclose(metrics.rmse(pred, true), [1.0, np.sqrt(2.0)])


def test_pearson_delta_uses_change_from_control():
    control = np.array([5.0, 5.0, 5.0])
    true = np.array([[6.0, 5.0, 4.0]])
    assert np.allclose(metrics.pearson_delta(true, true, control), [1.0])
    assert np.allclose(metrics.pearson_delta(2 * true - control, true, control), [1.0])
    assert np.allclose(metrics.pearson_delta(2 * control - true, true, control), [-1.0])


def test_pearson_delta_is_nan_for_constant_delta():
    control = np.zeros(3)
    assert np.isnan(metrics.pearson_delta(np.ones((1, 3)), np.array([[1.0, 2.0, 3.0]]), control))


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    rng = np.random.default_rng(0)
    values = rng.normal(size=50)
    groups = np.arange(50)
    mean, low, high = metrics.bootstrap_ci(values, groups, seed=1)
    assert low < mean < high
    assert mean == pytest.approx(values.mean())
    assert metrics.bootstrap_ci(values, groups, seed=1) == (mean, low, high)


def test_bootstrap_resamples_groups_not_cells():
    """1,000 near-identical cells in each of 2 groups are 2 data points, not 2,000.

    Resampling cells would give a CI of width ~0.01. Resampling groups gives one
    spanning both group values, which is the honest answer.
    """
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(0, 0.01, 1000), rng.normal(1, 0.01, 1000)])
    groups = np.repeat(["a", "b"], 1000)
    mean, low, high = metrics.bootstrap_ci(values, groups)
    assert mean == pytest.approx(0.5, abs=0.01)
    assert low < 0.05 and high > 0.95


def test_bootstrap_ignores_nan():
    mean, _, _ = metrics.bootstrap_ci([1.0, np.nan, 3.0], ["a", "b", "c"])
    assert mean == pytest.approx(2.0)


def test_scaled_score_maps_floor_to_0_and_ceiling_to_1():
    floor = np.array([4.0, 6.0])
    ceiling = np.array([1.0, 1.0])
    for model, expected in [(floor, 0.0), (ceiling, 1.0), (np.array([2.0, 4.0]), 0.5)]:
        score, low, high = metrics.scaled_score_ci(model, floor, ceiling, higher_is_better=False)
        assert score == pytest.approx(expected)
    score, _, _ = metrics.scaled_score_ci([0.5, 0.5], [0.0, 0.0], [1.0, 1.0], higher_is_better=True)
    assert score == pytest.approx(0.5)


def test_scaled_score_goes_negative_below_the_floor():
    score, _, _ = metrics.scaled_score_ci([7.0], [5.0], [1.0], higher_is_better=False)
    assert score == pytest.approx(-0.5)
