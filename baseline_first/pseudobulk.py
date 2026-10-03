"""Average cells into one profile per group (a "pseudobulk")."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def group_means(X, labels) -> tuple[np.ndarray, np.ndarray]:
    """Return (groups, means): the sorted unique labels and each group's mean row of X.

    `X` is a cells x genes array or sparse matrix; `labels` has one entry per cell.
    """
    labels = np.asarray(labels).astype(str)
    groups, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(groups)).astype(np.float64)
    indicator = sp.csr_matrix(
        (np.ones(len(labels)), (inverse, np.arange(len(labels)))),
        shape=(len(groups), len(labels)),
    )
    sums = indicator @ X
    sums = sums.toarray() if sp.issparse(sums) else np.asarray(sums)
    return groups, sums / counts[:, None]
