"""Dataset loaders.

Every loader returns an AnnData with the same per-cell columns, so the rest of
the package never needs to know which dataset it is looking at:

- ``perturbation``: canonical perturbation label. Target genes sorted and
  joined with ``+``; unperturbed cells are ``"control"``. This is the column
  to split on: two raw labels for the same perturbation become one value here.
- ``is_control``: True for unperturbed cells.
- ``n_targets``: number of genes perturbed (0 for control).
- ``cell_line``: the cell line the cells come from.

``X`` holds log-normalised expression. ``uns["baseline_first"]`` records where
the data came from and anything that was corrected on load.
"""

from baseline_first.data.norman import load_norman2019
from baseline_first.data.schema import CONTROL, STANDARD_COLUMNS

__all__ = ["CONTROL", "STANDARD_COLUMNS", "load_norman2019"]
