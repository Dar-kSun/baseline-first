"""Predict knockdown effects for genes and cell types absent from training.

The VCC 2026 task holds out both: the 300 target genes were not perturbed in
the public screens, and the cell contexts are new. What the challenge does
give is each context's control cells. So genes are described by how they
co-vary with a fixed set of anchor genes *in the cell type at hand*, and a
ridge regression learned on public screens maps that description to the
knockdown effect.

Profiles follow the VCC 2026 spec: counts summed per perturbation, normalised
to 5e4, log1p; an effect is a profile minus the control profile.
"""

from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

BULK_TARGET_SUM = 5e4
CONTROL = "non-targeting"
DEFAULT_ALPHAS = tuple(np.logspace(0, 5, 11))


def effects(bulk: ad.AnnData, cell_type: str, control: str = CONTROL) -> pd.DataFrame:
    """Effects (perturbations x genes) of one cell type from a pseudobulk AnnData."""
    sub = bulk[bulk.obs["cell_type"] == cell_type]
    counts = pd.DataFrame(
        np.asarray(sub.layers["counts_sum"], dtype=float),
        index=sub.obs["perturbation"].astype(str).to_numpy(),
        columns=sub.var_names,
    )
    profile = np.log1p(BULK_TARGET_SUM * counts.div(counts.sum(axis=1), axis=0))
    return profile.drop(index=control) - profile.loc[control]


def control_moments(cells) -> dict:
    """Moments of v = log1p(1e4 * counts / total) over a context's control cells,
    in the same form `baseline_first.data.bulk.pseudobulk` stores them."""
    counts = cells.toarray() if hasattr(cells, "toarray") else np.asarray(cells)
    totals = counts.sum(axis=1, keepdims=True)
    v = np.log1p(counts / np.where(totals > 0, totals, 1) * 1e4)
    return {"n": float(len(v)), "sum": v.sum(axis=0), "outer": v.T @ v}


def variances(moments: dict) -> np.ndarray:
    mean = moments["sum"] / moments["n"]
    return np.clip(np.diag(moments["outer"]) / moments["n"] - mean**2, 0, None)


def correlation_features(
    moments: dict, genes: Sequence[str], anchors: Sequence[int]
) -> pd.DataFrame:
    """Correlation of every gene with each anchor gene, across control cells."""
    n = moments["n"]
    mean = moments["sum"] / n
    anchors = np.asarray(anchors)
    cov = moments["outer"][:, anchors] / n - np.outer(mean, mean[anchors])
    sd = np.sqrt(variances(moments))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(sd, sd[anchors])
    return pd.DataFrame(np.nan_to_num(corr), index=list(genes))


def ridge_path(X, Y, X_new, alphas):
    """Ridge predictions (with intercept) at X_new for each alpha, from one SVD.

    Equal to sklearn's Ridge(alpha).fit(X, Y).predict(X_new) for every alpha,
    at the cost of a single decomposition.
    """
    x_mean, y_mean = X.mean(axis=0), Y.mean(axis=0)
    U, s, Vt = np.linalg.svd(X - x_mean, full_matrices=False)
    UtY = U.T @ (Y - y_mean)
    Z = (X_new - x_mean) @ Vt.T
    for alpha in alphas:
        yield Z @ ((s / (s**2 + alpha))[:, None] * UtY) + y_mean


class CoexpressionRidge:
    """Ridge from a target gene's co-expression description to its knockdown effect.

    `fit` takes, per training cell type, the gene features (from that cell
    type's controls) and the measured effects. A perturbation's features are
    its target gene's row; a target without a row gets zeros, so its
    prediction is the ridge intercept, i.e. the average response. The penalty
    is chosen by grouped cross-validation over target genes, so the selection
    never sees a gene's own effect through another cell type.
    """

    name = "CoexpressionRidge"

    def __init__(self, alphas=DEFAULT_ALPHAS, n_inner_folds: int = 5, seed: int = 0) -> None:
        self.alphas = alphas
        self.n_inner_folds = n_inner_folds
        self.seed = seed

    @staticmethod
    def _design(features: pd.DataFrame, perts: Sequence[str]) -> np.ndarray:
        rows = features.reindex(list(perts))
        return np.nan_to_num(rows.to_numpy(dtype=float))

    def fit(self, training: Sequence[tuple[pd.DataFrame, pd.DataFrame]]) -> CoexpressionRidge:
        X = np.vstack([self._design(f, e.index) for f, e in training])
        Y = np.vstack([e.to_numpy(dtype=float) for _, e in training])
        genes = np.concatenate([np.asarray(e.index) for _, e in training])
        self.columns_ = training[0][1].columns

        unique = np.unique(genes)
        rng = np.random.default_rng(self.seed)
        fold_of = dict(
            zip(rng.permutation(unique), np.arange(len(unique)) % self.n_inner_folds, strict=True)
        )
        folds = np.array([fold_of[g] for g in genes])
        errors = np.zeros(len(self.alphas))
        for k in range(self.n_inner_folds):
            test = folds == k
            for i, pred in enumerate(ridge_path(X[~test], Y[~test], X[test], self.alphas)):
                errors[i] += ((pred - Y[test]) ** 2).sum()
        self.errors_ = errors
        self.alpha_ = float(self.alphas[int(np.argmin(errors))])
        self.model_ = Ridge(alpha=self.alpha_).fit(X, Y)
        return self

    def predict(self, features: pd.DataFrame, perts: Sequence[str]) -> pd.DataFrame:
        pred = self.model_.predict(self._design(features, perts))
        return pd.DataFrame(pred, index=list(perts), columns=self.columns_)
