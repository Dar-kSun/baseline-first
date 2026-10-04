# Verdict card: norman2019

Produced by `baseline-first card --dataset norman2019 --folds 5 --seed 0`. Split: 5-fold cross-validation over perturbations; control always in training.

## How much data is there?

| | |
|---|---|
| cells | 91,205 |
| distinct conditions | 237 (cells overstate this **385x**) |
| design-effect n_eff | 13,678 (median ICC 0.015; 6.7x) |

A model of perturbation effects learns from 237 examples, not 91,205.

## What do trivial baselines score?

| method | MAE | Pearson delta | scaled MAE [95% CI] |
|---|---|---|---|
| Replicate | 0.0104 | 0.891 | 1.00 [1.00, 1.00] |
| GlobalMean | 0.0187 | 0.505 | 0.00 [0.00, 0.00] |
| HVGRidge | 0.0131 | 0.775 | 0.68 [0.62, 0.73] |

Scaled scores: 0 = GlobalMean (one average profile for every perturbation),
1 = a split-half replicate of the experiment. CIs are bootstrapped over
perturbations, never cells.

## The bar a bigger model must clear (on this split and these metrics)

- **MAE:** beat HVGRidge's scaled score of 0.68 by more than its 95% CI, i.e. score above **0.73**.
- **Pearson delta:** beat HVGRidge's scaled score of 0.70 by more than its 95% CI, i.e. score above **0.76**.
- **RMSE:** beat HVGRidge's scaled score of 0.67 by more than its 95% CI, i.e. score above **0.72**.

A model that does not clear these bars has not shown it beats a ridge
regression on this data. Clearing them on this dataset says nothing about
other datasets, splits or metrics.

![verdict](verdict.png)
