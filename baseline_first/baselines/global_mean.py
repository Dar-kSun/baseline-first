"""GlobalMean: predict the same profile, the average training perturbation, for everything."""

from __future__ import annotations

import numpy as np

from baseline_first.data.schema import CONTROL
from baseline_first.pseudobulk import group_means


class GlobalMean:
    """The floor. Ignores which gene was perturbed.

    The prediction is the unweighted mean of the training perturbations' mean
    profiles (control excluded), so a perturbation with many cells does not
    dominate it.
    """

    name = "GlobalMean"

    def fit(self, X, perturbations, genes) -> None:
        groups, means = group_means(X, perturbations)
        self.mean_ = means[groups != CONTROL].mean(axis=0)

    def predict(self, perturbations) -> np.ndarray:
        return np.tile(self.mean_, (len(perturbations), 1))
