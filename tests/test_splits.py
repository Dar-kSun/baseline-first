"""Leakage tests: the most important tests in the repo.

Each `test_leaky_*` builds a split that leaks on purpose and checks that
`assert_no_leakage` raises. If any of them stops raising, every number this
tool reports is suspect.
"""

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from baseline_first.splits import GroupSplitter, LeakageError, Split, assert_no_leakage


def make_obs(n_perts=10, cells_per_pert=20, n_donors=4, seed=0):
    rng = np.random.default_rng(seed)
    perts = np.repeat([f"KO_{i}" for i in range(n_perts)] + ["control"], cells_per_pert)
    return pd.DataFrame(
        {
            "perturbation": perts,
            "donor": rng.choice([f"D{i}" for i in range(n_donors)], size=len(perts)),
        },
        index=[f"cell_{i}" for i in range(len(perts))],
    )


# --- Deliberately leaky splits: these must raise -------------------------------------


def test_leaky_random_cell_split_raises():
    """The classic mistake: shuffle cells, cut 80/20. Perturbations land on both sides."""
    obs = make_obs()
    order = np.random.default_rng(0).permutation(len(obs))
    cut = int(0.8 * len(obs))
    train, test = obs.iloc[order[:cut]], obs.iloc[order[cut:]]
    with pytest.raises(LeakageError, match="perturbation"):
        assert_no_leakage(train, test, keys="perturbation")


def test_leaky_single_shared_group_raises():
    """Even one cell of one shared group is leakage."""
    obs = make_obs()
    train = obs[obs.perturbation != "KO_3"]
    test = pd.concat([obs[obs.perturbation == "KO_3"], train.iloc[[0]].rename(index=lambda s: "x")])
    with pytest.raises(LeakageError, match="1 value"):
        assert_no_leakage(train, test, keys="perturbation")


def test_leaky_on_second_key_raises():
    """Clean by perturbation, but donors are shared: fails when donor is also checked."""
    obs = make_obs()
    split = GroupSplitter("perturbation", seed=0).split(obs)
    train, test = split.apply(obs)
    assert_no_leakage(train, test, keys="perturbation")
    with pytest.raises(LeakageError, match="donor"):
        assert_no_leakage(train, test, keys=["perturbation", "donor"])


def test_leaky_same_cell_on_both_sides_raises():
    obs = make_obs()
    train = obs[obs.perturbation.isin(["KO_0", "KO_1"])]
    test = obs[obs.perturbation == "KO_2"].copy()
    test.loc[train.index[0]] = {"perturbation": "KO_2", "donor": "D0"}
    with pytest.raises(LeakageError, match="cell"):
        assert_no_leakage(train, test, keys="perturbation")


def test_leaky_anndata_split_raises():
    obs = make_obs()
    adata = ad.AnnData(X=np.zeros((len(obs), 3), dtype=np.float32), obs=obs)
    leaky = Split(
        train=np.arange(0, len(obs), 2), test=np.arange(1, len(obs), 2), group_key="perturbation"
    )
    with pytest.raises(LeakageError):
        leaky.apply(adata)


def test_leakage_is_an_exception_not_a_warning():
    """Silencing warnings must not silence the leakage check."""
    assert not issubclass(LeakageError, Warning)
    obs = make_obs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(LeakageError):
            assert_no_leakage(obs, obs, keys="perturbation")


# --- Unverifiable inputs: these must refuse rather than pass -------------------------


def test_missing_key_refuses():
    obs = make_obs()
    with pytest.raises(KeyError, match="cell_line"):
        assert_no_leakage(obs.iloc[:10], obs.iloc[-10:], keys="cell_line")


def test_missing_group_labels_refuse():
    obs = make_obs()
    test = obs.iloc[-10:].copy()
    test.loc[test.index[0], "perturbation"] = np.nan
    with pytest.raises(ValueError, match="no value"):
        assert_no_leakage(obs.iloc[:10], test, keys="perturbation")


def test_no_keys_refuses():
    obs = make_obs()
    with pytest.raises(ValueError):
        assert_no_leakage(obs, obs, keys=[])


# --- GroupSplitter produces clean splits ---------------------------------------------


def test_group_split_is_clean_and_complete():
    obs = make_obs()
    split = GroupSplitter("perturbation", test_size=0.3, seed=1).split(obs)
    train, test = split.apply(obs)
    assert set(train.perturbation).isdisjoint(test.perturbation)
    assert sorted(np.concatenate([split.train, split.test])) == list(range(len(obs)))
    assert test.perturbation.nunique() == 3  # 30% of the 10 testable groups


def test_test_size_counts_groups_not_cells():
    """4 groups, one of them 100x larger: test_size=0.25 must hold out 1 group, whichever."""
    sizes = {"KO_0": 5, "KO_1": 5, "KO_2": 5, "KO_big": 500}
    obs = pd.DataFrame({"perturbation": np.repeat(list(sizes), list(sizes.values()))})
    for seed in range(10):
        split = GroupSplitter("perturbation", test_size=0.25, seed=seed).split(obs)
        assert obs.iloc[split.test].perturbation.nunique() == 1


def test_split_is_reproducible_and_seed_dependent():
    obs = make_obs(n_perts=30)
    a = GroupSplitter("perturbation", seed=7).split(obs)
    b = GroupSplitter("perturbation", seed=7).split(obs)
    c = GroupSplitter("perturbation", seed=8).split(obs)
    assert np.array_equal(a.test, b.test)
    assert not np.array_equal(a.test, c.test)


def test_always_train_groups_are_never_tested():
    obs = make_obs()
    splitter = GroupSplitter("perturbation", seed=0, always_train=["control"])
    for split in [splitter.split(obs), *splitter.folds(obs, n_folds=5)]:
        assert "control" not in set(obs.iloc[split.test].perturbation)
        assert "control" in set(obs.iloc[split.train].perturbation)


def test_unknown_always_train_group_refuses():
    with pytest.raises(ValueError, match="not found"):
        GroupSplitter("perturbation", always_train=["ctrl"]).split(make_obs())


def test_folds_test_every_group_exactly_once():
    obs = make_obs()
    splitter = GroupSplitter("perturbation", seed=0, always_train=["control"])
    tested = []
    for split in splitter.folds(obs, n_folds=4):
        train, test = split.apply(obs)
        tested.extend(test.perturbation.unique())
    assert sorted(tested) == sorted(f"KO_{i}" for i in range(10))


def test_splitter_works_on_anndata():
    obs = make_obs()
    adata = ad.AnnData(X=np.zeros((len(obs), 3), dtype=np.float32), obs=obs)
    train, test = GroupSplitter("perturbation", seed=0).split(adata).apply(adata)
    assert set(train.obs.perturbation).isdisjoint(test.obs.perturbation)


@pytest.mark.parametrize("test_size", [0, 1, -0.1, 1.5])
def test_invalid_test_size_refuses(test_size):
    with pytest.raises(ValueError):
        GroupSplitter("perturbation", test_size=test_size)


def test_too_few_groups_refuses():
    obs = make_obs(n_perts=1)
    with pytest.raises(ValueError, match="at least 2 groups"):
        GroupSplitter("perturbation", always_train=["control"]).split(obs)
