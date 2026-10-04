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


def load_lines(paths: Sequence) -> tuple[list[str], dict, dict]:
    """Effects and control moments of every cell type in the given pseudobulk files,
    restricted to the genes they all share (in the first file's order).

    Returns (genes, effects by cell type, moments by cell type).
    """
    eff, moments, gene_sets = {}, {}, []
    for path in paths:
        bulk = ad.read_h5ad(path)
        m_genes = list(bulk.uns.get("moments_genes", bulk.var_names))
        gene_sets += [list(bulk.var_names), m_genes]
        for cell_type, m in bulk.uns["moments"].items():
            eff[cell_type] = effects(bulk, cell_type)
            moments[cell_type] = (m, m_genes)
    shared = set.intersection(*(set(g) for g in gene_sets))
    genes = [g for g in gene_sets[0] if g in shared]
    restricted = {}
    for cell_type, (m, m_genes) in moments.items():
        idx = pd.Index(m_genes).get_indexer(genes)
        restricted[cell_type] = {
            "n": float(m["n"]),
            "sum": np.asarray(m["sum"])[idx],
            "outer": np.asarray(m["outer"])[np.ix_(idx, idx)],
        }
        eff[cell_type] = eff[cell_type][genes]
    return genes, eff, restricted


def optimal_scale(pairs) -> float:
    """The scalar s minimising sum ||s * pred - true||^2 over (pred, true) pairs."""
    pairs = list(pairs)
    num = sum(float((p * t).sum()) for p, t in pairs)
    den = sum(float((p * p).sum()) for p, t in pairs)
    return num / den if den > 0 else 0.0


def fold_changes(
    effect: pd.DataFrame,
    controls,
    genes_all: Sequence[str],
    target_fold_change: float = 0.25,
    max_fold_change: float = 10.0,
) -> pd.DataFrame:
    """Per-gene fold changes (perturbations x genes_all) on a context's control profile.

    `effect` holds predicted changes in log1p(5e4-normalised) profile on a subset of
    genes; genes outside it keep fold change 1. Each perturbation's own target gene is
    set to `target_fold_change`, a nominal CRISPRi knockdown that no VCC metric scores.
    """
    genes_all = list(genes_all)
    pos = pd.Index(genes_all).get_indexer(list(effect.columns))
    if (pos < 0).any():
        raise ValueError("effect genes must all be on the full gene axis")
    summed = np.asarray(controls.sum(axis=0)).ravel()
    ctrl_profile = np.log1p(BULK_TARGET_SUM * summed / summed.sum())[pos]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        fc = np.expm1(ctrl_profile + effect.to_numpy()) / np.expm1(ctrl_profile)
    fc = np.where(ctrl_profile > 0, fc, 1.0)
    fc = np.clip(np.nan_to_num(fc, nan=1.0, posinf=max_fold_change), 0.0, max_fold_change)
    out = np.ones((len(effect), len(genes_all)))
    out[:, pos] = fc
    targets = pd.Index(genes_all).get_indexer(list(effect.index))
    hit = targets >= 0
    out[np.flatnonzero(hit), targets[hit]] = target_fold_change
    return pd.DataFrame(out, index=effect.index, columns=genes_all)


class Transfer:
    """A fitted transfer model: anchors, ridge, average response and shrinkage factors."""

    def __init__(self, genes, anchors, ridge, mean_response, scale_ridge, scale_mean):
        self.genes = list(genes)
        self.anchors = anchors
        self.ridge = ridge
        self.mean_response = mean_response
        self.scale_ridge = scale_ridge
        self.scale_mean = scale_mean

    def predict(self, method: str, moments: dict, perts: Sequence[str]) -> pd.DataFrame:
        """Predicted effects (perts x genes) in a context described by its control moments."""
        if method == "ridge":
            feats = correlation_features(moments, self.genes, self.anchors)
            return self.ridge.predict(feats, perts) * self.scale_ridge
        if method == "mean":
            row = self.mean_response.to_numpy() * self.scale_mean
            return pd.DataFrame(
                np.tile(row, (len(perts), 1)), index=list(perts), columns=self.genes
            )
        if method == "none":
            return pd.DataFrame(0.0, index=list(perts), columns=self.genes)
        raise ValueError(f"unknown method {method!r}")


def fit_transfer(
    genes: Sequence[str],
    eff: dict,
    moments: dict,
    n_anchors: int = 1000,
    n_folds: int = 5,
    seed: int = 0,
) -> Transfer:
    """Fit the co-expression ridge and the average response on the given cell lines,
    and their shrinkage factors: each line held out in turn and predicted, by models
    fitted on the others, on a gene fold they did not train on."""
    lines = sorted(eff)
    pooled = sum(variances(moments[m]) for m in lines)
    anchors = np.sort(np.argsort(-pooled, kind="stable")[:n_anchors])
    feats = {m: correlation_features(moments[m], genes, anchors) for m in lines}
    ridge = CoexpressionRidge(seed=seed).fit([(feats[m], eff[m]) for m in lines])
    mean_response = pd.concat(eff.values()).mean(axis=0)

    all_perts = sorted(set().union(*(e.index for e in eff.values())))
    rng = np.random.default_rng(seed)
    fold_of = dict(
        zip(rng.permutation(all_perts), np.arange(len(all_perts)) % n_folds, strict=True)
    )
    ridge_pairs, mean_pairs = [], []
    for held in lines:
        rest = [m for m in lines if m != held]
        inner = {m: eff[m].loc[[p for p in eff[m].index if fold_of[p] != 0]] for m in rest}
        target = eff[held].loc[[p for p in eff[held].index if fold_of[p] == 0]]
        inner_ridge = CoexpressionRidge(alphas=(ridge.alpha_,), n_inner_folds=2).fit(
            [(feats[m], inner[m]) for m in rest]
        )
        ridge_pairs.append(
            (inner_ridge.predict(feats[held], target.index).to_numpy(), target.to_numpy())
        )
        inner_mean = pd.concat(inner.values()).mean(axis=0).to_numpy()
        mean_pairs.append((np.tile(inner_mean, (len(target), 1)), target.to_numpy()))
    return Transfer(
        genes, anchors, ridge, mean_response, optimal_scale(ridge_pairs), optimal_scale(mean_pairs)
    )


def context_moments(cells, columns) -> dict:
    """Moments of log1p(1e4 * counts / total) over a context's control cells on `columns`,
    with each cell's total taken over all genes."""
    import scipy.sparse as sp

    cells = sp.csr_matrix(cells)
    totals = np.asarray(cells.sum(axis=1)).ravel()
    scaled = sp.diags(1e4 / np.where(totals > 0, totals, 1)) @ cells
    v = np.log1p(scaled[:, columns].toarray())
    return {"n": float(len(v)), "sum": v.sum(axis=0), "outer": v.T @ v}
