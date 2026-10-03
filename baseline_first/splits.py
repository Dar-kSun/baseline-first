"""Group-aware splitting and leakage detection.

Splitting cells at random puts cells from the same perturbation (or donor, or
cell line) on both sides of the split. Those cells are near-replicates, so the
test set then measures memorisation, not generalisation. This module splits by
group, never by cell, and `assert_no_leakage` raises if any group value ends up
on both sides.

Leakage raises `LeakageError`, an exception. It is never a warning, so it
cannot be silenced by a warnings filter.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

_MAX_EXAMPLES = 5


class LeakageError(Exception):
    """A group value (or a cell) appears in both the train and the test set."""


def _obs(data) -> pd.DataFrame:
    """Return the per-cell metadata table from an AnnData or a DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data
    obs = getattr(data, "obs", None)
    if isinstance(obs, pd.DataFrame):
        return obs
    raise TypeError(f"expected an AnnData or a pandas DataFrame, got {type(data).__name__}")


def _group_values(obs: pd.DataFrame, key: str, side: str) -> pd.Series:
    if key not in obs.columns:
        raise KeyError(f"group key {key!r} is not a column of the {side} metadata")
    values = obs[key]
    if values.isna().any():
        n = int(values.isna().sum())
        raise ValueError(
            f"{n} {side} cells have no value for {key!r}; leakage cannot be checked "
            "for cells without a group label"
        )
    return values


def assert_no_leakage(train, test, keys: str | Sequence[str]) -> None:
    """Raise `LeakageError` if any value of any key appears in both train and test.

    `train` and `test` are AnnData objects or DataFrames of per-cell metadata.
    Each key is checked independently: a split by perturbation that also shares
    a donor across sides fails if `keys` includes the donor column.

    The cell index is checked too, unless both sides carry a default integer
    index (which says nothing about cell identity): the same cell on both sides
    is the most direct form of leakage.

    A key missing from either side raises `KeyError`, and missing group labels
    raise `ValueError`, because in both cases the absence of leakage cannot be
    verified.
    """
    train_obs, test_obs = _obs(train), _obs(test)
    keys = [keys] if isinstance(keys, str) else list(keys)
    if not keys:
        raise ValueError("assert_no_leakage needs at least one group key")

    for key in keys:
        train_values = set(_group_values(train_obs, key, "train"))
        test_values = set(_group_values(test_obs, key, "test"))
        shared = train_values & test_values
        if shared:
            examples = sorted(map(str, shared))[:_MAX_EXAMPLES]
            raise LeakageError(
                f"{len(shared)} value(s) of {key!r} appear in both train and test, "
                f"e.g. {examples}. Split by {key!r} instead of by cell."
            )

    default_index = isinstance(train_obs.index, pd.RangeIndex) and isinstance(
        test_obs.index, pd.RangeIndex
    )
    if not default_index:
        shared_cells = train_obs.index.intersection(test_obs.index)
        if len(shared_cells):
            examples = list(map(str, shared_cells[:_MAX_EXAMPLES]))
            raise LeakageError(
                f"{len(shared_cells)} cell(s) appear in both train and test, e.g. {examples}"
            )


@dataclass(frozen=True)
class Split:
    """Row positions of one train/test split, plus the key it was grouped by."""

    train: np.ndarray
    test: np.ndarray
    group_key: str

    def apply(self, data):
        """Return (train, test) subsets of an AnnData or DataFrame, checked for leakage."""
        if isinstance(data, pd.DataFrame):
            train, test = data.iloc[self.train], data.iloc[self.test]
        else:
            train, test = data[self.train], data[self.test]
        assert_no_leakage(train, test, self.group_key)
        return train, test


class GroupSplitter:
    """Split cells so that every group falls wholly in train or wholly in test.

    Parameters
    ----------
    group_key:
        Column of the cell metadata to group by, e.g. the perturbation, the
        donor or the cell line. Cells are never split individually.
    test_size:
        Fraction of *groups* (not cells) held out for testing, in (0, 1).
    seed:
        Seed for the random assignment of groups, for reproducible splits.
    always_train:
        Group values that must stay in training and never be tested, such as
        the unperturbed control. They are used as reference, not predicted.
    """

    def __init__(
        self,
        group_key: str,
        test_size: float = 0.2,
        seed: int = 0,
        always_train: Iterable[str] = (),
    ) -> None:
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        self.group_key = group_key
        self.test_size = test_size
        self.seed = seed
        self.always_train = frozenset(always_train)

    def _groups(self, data) -> tuple[pd.Series, np.ndarray]:
        """Return each cell's group label and the shuffled groups that may be tested."""
        labels = _group_values(_obs(data), self.group_key, "input")
        unknown = self.always_train - set(labels)
        if unknown:
            raise ValueError(f"always_train groups not found in {self.group_key!r}: {unknown}")
        candidates = np.array(sorted(set(labels) - self.always_train, key=str), dtype=object)
        if len(candidates) < 2:
            raise ValueError(
                f"need at least 2 groups in {self.group_key!r} to split, "
                f"found {len(candidates)} (excluding always_train)"
            )
        rng = np.random.default_rng(self.seed)
        return labels, rng.permutation(candidates)

    def _make_split(self, data, labels: pd.Series, test_groups: Iterable) -> Split:
        in_test = labels.isin(set(test_groups)).to_numpy()
        split = Split(
            train=np.flatnonzero(~in_test),
            test=np.flatnonzero(in_test),
            group_key=self.group_key,
        )
        split.apply(data)  # asserts no leakage before the split is ever handed out
        return split

    def split(self, data) -> Split:
        """Return one train/test split holding out `test_size` of the groups."""
        labels, groups = self._groups(data)
        n_test = min(max(1, round(self.test_size * len(groups))), len(groups) - 1)
        return self._make_split(data, labels, groups[:n_test])

    def folds(self, data, n_folds: int = 5) -> Iterator[Split]:
        """Yield `n_folds` splits in which every testable group is tested exactly once."""
        labels, groups = self._groups(data)
        if not 2 <= n_folds <= len(groups):
            raise ValueError(f"n_folds must be between 2 and the number of groups ({len(groups)})")
        for test_groups in np.array_split(groups, n_folds):
            yield self._make_split(data, labels, test_groups)
