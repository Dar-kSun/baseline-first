# baseline-first

> Run the dumb baseline before you train the big model.

**Status: v0.1, in development.** Nothing here is finished, and this README
contains no results yet. Numbers will appear only once each one can be
reproduced from a command in this repo.

## What it will do

Before you fine-tune a single-cell foundation model, `baseline-first` will tell
you two things about your dataset:

1. **What a trivial baseline scores on it**: predicting the training mean,
   per-condition means, highly-variable-genes + ridge, PCA + linear, and nearest
   neighbour. These are evaluated with group-aware splits, so a perturbation
   seen in training never appears in test.
2. **How much information it actually contains**: an effective sample size
   (`n_eff`), and the factor by which the raw cell count overstates it.

## Development

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"
pytest
ruff check .
```

## Licence

MIT. See [LICENSE](LICENSE).
