from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from baseline_first.baselines import GlobalMean, HVGRidge
from baseline_first.data import CONTROL
from baseline_first.evaluate import evaluate, summarise
from baseline_first.splits import LeakageError, Split

FIXTURE = Path(__file__).parent / "fixtures" / "norman2019_mini.h5ad"


@pytest.fixture(scope="module")
def mini():
    return ad.read_h5ad(FIXTURE)


@pytest.fixture(scope="module")
def results(mini):
    return evaluate(mini, [GlobalMean(), HVGRidge()], n_folds=3)


def test_every_perturbation_is_scored_once_per_baseline(mini, results):
    expected = set(mini.obs["perturbation"].astype(str)) - {CONTROL}
    for _, frame in results.groupby("baseline"):
        assert sorted(frame["perturbation"]) == sorted(expected)
    assert np.isfinite(results[["mae", "rmse"]].to_numpy()).all()


def test_summary_has_cis_for_each_subset(results):
    summary = summarise(results, n_boot=200)
    assert set(summary["subset"]) == {"all", "single", "pair"}
    assert (summary["ci_low"] <= summary["mean"]).all()
    assert (summary["mean"] <= summary["ci_high"]).all()


class LeakySplitter:
    """A splitter that ignores groups: what a careless resplit would look like."""

    def folds(self, data, n_folds):
        perturbed = np.flatnonzero(~data.obs["is_control"].to_numpy())
        control = np.flatnonzero(data.obs["is_control"].to_numpy())
        yield Split(
            train=np.concatenate([control, perturbed[::2]]),
            test=perturbed[1::2],
            group_key="perturbation",
        )


def test_evaluation_loop_rejects_a_leaky_splitter(mini):
    """The assertion runs inside the loop, so swapping in a bad splitter cannot bypass it."""
    with pytest.raises(LeakageError):
        evaluate(mini, [GlobalMean()], splitter=LeakySplitter())


def test_scaled_scores_anchor_floor_at_0_and_replicate_at_1(results):
    summary = summarise(results, n_boot=200)
    by = summary.set_index(["subset", "baseline", "metric"])["scaled"]
    for metric in ("mae", "rmse", "pearson_delta"):
        assert by[("all", "GlobalMean", metric)] == pytest.approx(0.0)
        assert by[("all", "Replicate", metric)] == pytest.approx(1.0)


def test_pairs_record_whether_their_singles_were_trained_on(results):
    pairs = results[results["n_targets"] == 2]
    singles = results[results["n_targets"] == 1]
    assert pairs["n_constituents_in_train"].between(0, 2).all()
    assert (singles["n_constituents_in_train"] == 0).all()
