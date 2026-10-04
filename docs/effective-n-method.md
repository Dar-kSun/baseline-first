# Effective sample size: method, assumptions, limits

Implementation: `baseline_first/effective_n.py`. Survey:
`python scripts/02_effective_n_survey.py` → `results/effective_n/summary.md`.

## Why

A perturbation dataset reported as "91,205 cells" sounds large. But cells
that share a condition are near-replicates of one another, so they carry far
less independent information than their count suggests. There is no single
canonical definition of how much less, so two standard estimates are
reported, and they answer different questions.

## Estimate 1: distinct conditions

The number of distinct condition labels (after merging labels that denote the
same perturbation; see `data/MANIFEST.md` for the Norman case).

- **Question it answers:** how many independent examples does a model that
  learns *condition effects* get to see? A model predicting the effect of an
  unseen perturbation learns from one example per training perturbation, not
  from one per cell.
- **Inflation factor:** cells / conditions.
- **Assumption:** conditions are themselves independent. Related conditions
  (a pair and its two singles; two guides for one gene) make even this an
  overestimate.

## Estimate 2: design effect (Kish 1965)

```
n_eff = n / (1 + (m0 - 1) * ICC)
```

- `ICC`, the intraclass correlation, is the share of a gene's variance that
  lies between conditions rather than within them. It is estimated per gene by
  one-way ANOVA: `ICC = (MSB - MSW) / (MSB + (m0 - 1) MSW)`, clipped to [0, 1].
- `m0 = (n - sum(n_i^2) / n) / (k - 1)` is the standard ANOVA group-size
  constant; it equals the group size when all groups are the same size.
- ICC is computed on log1p-normalised expression for the 2,000 most variable
  genes, and the **median** over genes is used. The interquartile range is
  reported alongside.
- **Question it answers:** how many independent cells' worth of information
  does the dataset hold for estimating a gene's mean expression, given that
  cells cluster by condition?
- **Inflation factor:** cells / n_eff.

## Where it breaks

- **One number for many genes.** ICC differs widely between genes: most genes
  do not respond to most perturbations, so their ICC is near 0, while a few
  respond strongly. The median hides that spread, which is why the IQR is
  reported. A different summary (a mean, or a PCA-based multivariate ICC)
  would give a different n_eff.
- **Single-cell noise drives ICC down.** Sparse counts add within-condition
  variance, so per-gene ICC is small and n_eff is large. That is a property of
  the measurement, not evidence that conditions are informative.
- **Normal random-effects model.** The ANOVA estimator assumes condition
  effects and residuals that are roughly normal and homoscedastic. Log counts
  only approximate this.
- **Hidden structure.** Batch (gem group), cell cycle or guide efficiency add
  between-condition variance that is not the perturbation's effect, and would
  inflate ICC.
- **It is not a learning-theoretic quantity.** For "how many examples does a
  model of perturbation effects have", use Estimate 1.

## Checks

- Unit tests (`tests/test_effective_n.py`) recover known variance shares under
  a simulated random-effects model, give n_eff ≈ n when conditions have no
  effect, and give n_eff ≈ k when cells within a condition are identical.
- **Negative control on real data:** VCC 2026 control cells grouped by their
  46 non-targeting guides. Guides that target nothing should not differ, and
  the median ICC is indeed about 0 (inflation 1.0–1.1x). In that row the
  "cells per condition" column is not meaningful, because the guides are not
  real conditions.

## Results (from the survey command)

| dataset | cells | conditions | cells / condition | median ICC [IQR] | design-effect n_eff | cells / n_eff |
|---|---|---|---|---|---|---|
| Norman 2019 | 91,205 | 237 | **385x** | 0.015 [0.004, 0.045] | 13,678 | 6.7x |
| VCC 2026 controls, context A (negative control) | 18,400 | 46 guides | – | 0.000 [0.000, 0.001] | 16,603 | 1.1x |

Read the two Norman numbers together. A model of perturbation effects has 237
examples, so the cell count overstates its data by 385x. The design effect
says the same cells are worth about 13,700 independent cells for estimating
gene means. Both are true; they answer different questions.
