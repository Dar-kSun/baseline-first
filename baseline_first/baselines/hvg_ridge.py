"""HVGRidge: describe each target gene by its co-expression with the top-k variable genes,
then fit a ridge regression from that description to the perturbation's effect.

For each measured gene g, its feature vector is the Pearson correlation between g
and each of the k most variable genes, across training cells. A perturbation's
features are the sum over its target genes. Ridge maps features to the change
from control; the prediction adds the control mean back.

Every quantity is computed from training cells, so a held-out perturbation's
cells never inform its own features. The ridge penalty is chosen by efficient
leave-one-out over training perturbations (each perturbation is one sample).
A target gene that is not measured gets a zero feature vector, which makes the
prediction the ridge intercept; `n_unmeasured_` counts such predictions.

This is in the spirit of the linear gene-embedding baseline of Ahlmann-Eltze,
Huber & Anders (Nat. Methods 2025), with a correlation embedding in place of PCA.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import RidgeCV

from baseline_first.baselines.base import targets
from baseline_first.data.schema import CONTROL
from baseline_first.pseudobulk import group_means

DEFAULT_ALPHAS = tuple(np.logspace(-2, 4, 13))


def _column_moments(X) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(X.mean(axis=0)).ravel()
    sq = X.multiply(X) if sp.issparse(X) else np.asarray(X) ** 2
    var = np.asarray(sq.mean(axis=0)).ravel() - mean**2
    return mean, np.clip(var, 0, None)


class HVGRidge:
    name = "HVGRidge"

    def __init__(self, k: int = 4096, alphas=DEFAULT_ALPHAS) -> None:
        self.k = k
        self.alphas = alphas

    def _features(self, perturbations) -> np.ndarray:
        rows = np.zeros((len(perturbations), self.embedding_.shape[1]))
        self.n_unmeasured_ = 0
        for i, p in enumerate(perturbations):
            idx = [self.gene_index_[g] for g in targets(p) if g in self.gene_index_]
            if len(idx) < len(targets(p)):
                self.n_unmeasured_ += 1
            if idx:
                rows[i] = self.embedding_[idx].sum(axis=0)
        return rows

    def fit(self, X, perturbations, genes) -> None:
        X = sp.csr_matrix(X) if sp.issparse(X) else np.asarray(X)
        n_cells = X.shape[0]
        mean, var = _column_moments(X)
        k = min(self.k, X.shape[1])
        self.hvg_ = np.sort(np.argsort(-var, kind="stable")[:k])

        # Correlation of every gene with each HVG, across training cells.
        cross = X.T @ X[:, self.hvg_]
        cross = cross.toarray() if sp.issparse(cross) else np.asarray(cross)
        cov = cross / n_cells - np.outer(mean, mean[self.hvg_])
        sd = np.sqrt(var)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = cov / np.outer(sd, sd[self.hvg_])
        self.embedding_ = np.nan_to_num(corr)
        self.gene_index_ = {g: i for i, g in enumerate(genes)}

        groups, means = group_means(X, perturbations)
        is_control = groups == CONTROL
        if not is_control.any():
            raise ValueError("HVGRidge needs control cells in training to define the effect")
        self.control_ = means[is_control][0]
        train_groups = groups[~is_control]
        deltas = means[~is_control] - self.control_
        self.ridge_ = RidgeCV(alphas=self.alphas).fit(self._features(train_groups), deltas)
        self.alpha_ = float(self.ridge_.alpha_)

    def predict(self, perturbations) -> np.ndarray:
        return self.control_ + self.ridge_.predict(self._features(perturbations))
