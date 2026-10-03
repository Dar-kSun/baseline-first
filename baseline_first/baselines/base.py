"""The interface every baseline implements.

The task is to predict the mean expression profile of a perturbation that was
never seen in training. The only input at prediction time is the
perturbation's label (its target genes), so `predict` takes labels, not a
feature matrix. Anything a baseline knows about a gene it must learn in `fit`,
from training cells only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Baseline(Protocol):
    name: str

    def fit(self, X, perturbations: Sequence[str], genes: Sequence[str]) -> None:
        """Learn from training cells.

        X: cells x genes log-normalised expression (dense or sparse).
        perturbations: canonical perturbation label per cell; control cells are
            labelled `baseline_first.data.CONTROL`.
        genes: gene name per column of X, used to look up perturbation targets.
        """
        ...

    def predict(self, perturbations: Sequence[str]) -> np.ndarray:
        """Return predicted mean expression, one row per perturbation, one column per gene."""
        ...


def targets(perturbation: str) -> list[str]:
    """Target genes of a canonical perturbation label ("A+B" -> ["A", "B"])."""
    return perturbation.split("+")
