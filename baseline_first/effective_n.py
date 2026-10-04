"""Effective sample size of a single-cell perturbation dataset.

Cells from the same condition are near-replicates, so a dataset's raw cell
count overstates how much independent information it holds. Two standard
estimates are reported, because no single definition is canonical:

1. Unique conditions: the number of distinct condition labels. Simple and
   defensible: a model learning condition effects has this many examples.
2. Design effect (Kish 1965): n_eff = n / (1 + (m0 - 1) * ICC), where ICC is
   the intraclass correlation, the share of a gene's variance that lies
   between conditions, estimated by one-way ANOVA, and m0 is the ANOVA
   group-size constant for unequal groups. ICC is computed per gene; the
   median over the most variable genes is used.

Two inflation factors follow: n / n_eff and n / (number of conditions).
See docs/effective-n-method.md for the assumptions and where the estimator
breaks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class EffectiveN:
    n_cells: int
    n_conditions: int
    m0: float
    icc_median: float
    icc_iqr: tuple[float, float]
    n_genes: int
    n_eff: float
    inflation: float
    inflation_conditions: float

    def as_dict(self) -> dict:
        return asdict(self)


def anova_icc(X, groups) -> tuple[np.ndarray, float]:
    """One-way ANOVA intraclass correlation per column of X, and m0.

    ICC = (MSB - MSW) / (MSB + (m0 - 1) MSW), clipped to [0, 1], with
    m0 = (n - sum(n_i^2) / n) / (k - 1).
    """
    groups = np.asarray(groups).astype(str)
    labels, inverse = np.unique(groups, return_inverse=True)
    k, n = len(labels), len(groups)
    if k < 2:
        raise ValueError("need at least two conditions")
    sizes = np.bincount(inverse).astype(float)
    if (sizes < 2).all():
        raise ValueError("need replicate cells within conditions")
    indicator = sp.csr_matrix((np.ones(n), (inverse, np.arange(n))), shape=(k, n))
    X = sp.csr_matrix(X) if sp.issparse(X) else np.asarray(X, dtype=float)
    sums = np.asarray((indicator @ X).todense() if sp.issparse(X) else indicator @ X)
    sq = X.multiply(X) if sp.issparse(X) else X**2
    total_sq = np.asarray(sq.sum(axis=0)).ravel()
    grand = sums.sum(axis=0) / n
    ss_between = (sums**2 / sizes[:, None]).sum(axis=0) - n * grand**2
    ss_within = total_sq - (sums**2 / sizes[:, None]).sum(axis=0)
    msb = ss_between / (k - 1)
    msw = ss_within / (n - k)
    m0 = (n - (sizes**2).sum() / n) / (k - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        icc = (msb - msw) / (msb + (m0 - 1) * msw)
    return np.clip(icc, 0.0, 1.0), float(m0)


def effective_n(X, groups, n_top_genes: int = 2000) -> EffectiveN:
    """Both effective-sample-size estimates for cells X (cells x genes) in `groups`.

    X should be on a variance-stabilised scale (e.g. log-normalised). ICC is
    computed on the `n_top_genes` columns with the highest variance.
    """
    X = sp.csr_matrix(X) if sp.issparse(X) else np.asarray(X, dtype=float)
    n = X.shape[0]
    mean = np.asarray(X.mean(axis=0)).ravel()
    sq = X.multiply(X) if sp.issparse(X) else X**2
    var = np.asarray(sq.mean(axis=0)).ravel() - mean**2
    top = np.sort(np.argsort(-var, kind="stable")[: min(n_top_genes, X.shape[1])])
    icc, m0 = anova_icc(X[:, top], groups)
    icc = icc[np.isfinite(icc)]
    median = float(np.median(icc))
    n_eff = n / (1 + (m0 - 1) * median)
    q1, q3 = np.percentile(icc, [25, 75])
    return EffectiveN(
        n_cells=n,
        n_conditions=len(np.unique(np.asarray(groups).astype(str))),
        m0=m0,
        icc_median=median,
        icc_iqr=(float(q1), float(q3)),
        n_genes=len(icc),
        n_eff=float(n_eff),
        inflation=float(n / n_eff),
        inflation_conditions=float(n / len(np.unique(np.asarray(groups).astype(str)))),
    )
