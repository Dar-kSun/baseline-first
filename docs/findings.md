# Findings

Every number here comes from a command in this repo, and each section names it.
Claims are scoped to the dataset, split and metrics stated; nothing here says
how these baselines compare with any foundation model, because no foundation
model has been run yet.

## M3: GlobalMean and HVGRidge on Norman 2019

**Command:** `python scripts/01_run_baselines.py --dataset norman2019`
(about 2 minutes on a laptop). Output: `results/norman2019/`, with the exact
commit, seed and package versions in `run.json`. Two runs from the same commit
gave byte-identical CSVs.

**Setup.**
- Data: Norman et al. 2019, GEARS copy, 91,205 K562 cells × 5,045 genes,
  log-normalised (see `data/MANIFEST.md`).
- Task: predict the mean expression profile of a perturbation never seen in
  training.
- Split: 5-fold cross-validation over the 236 canonical perturbations (105
  single-gene, 131 two-gene), each held out exactly once. Control cells are
  always in training. `assert_no_leakage` runs on every fold.
- Observed profile: each held-out perturbation's cells are split at random into
  halves. Half B is the truth everything is scored against. Half A's mean is
  the **Replicate** ceiling: what repeating the experiment would give.
- Metrics: MAE and RMSE of the predicted profile; Pearson correlation of the
  predicted vs observed change from control ("Pearson delta"). The **scaled**
  score puts GlobalMean at 0 and Replicate at 1, the same anchors as the 2026
  Virtual Cell Challenge's reference-scaled metrics. It is not the VCC score:
  the VCC uses six metrics, which this repo does not implement yet.
- 95% CIs: percentile bootstrap over perturbations (2,000 resamples), never
  over cells.

**Results** (mean [95% CI]):

| subset | n | method | MAE | Pearson delta | scaled MAE | scaled Pearson delta |
|---|---|---|---|---|---|---|
| all | 236 | GlobalMean | 0.0187 [0.0177, 0.0198] | 0.50 [0.47, 0.54] | 0 | 0 |
| all | 236 | HVGRidge | 0.0131 [0.0125, 0.0138] | 0.78 [0.74, 0.80] | 0.68 [0.62, 0.73] | 0.70 [0.64, 0.76] |
| all | 236 | Replicate | 0.0104 [0.0098, 0.0111] | 0.89 [0.88, 0.91] | 1 | 1 |
| single | 105 | HVGRidge | 0.0121 [0.0112, 0.0130] | 0.65 [0.59, 0.70] | 0.46 [0.36, 0.55] | 0.48 [0.38, 0.58] |
| pair | 131 | HVGRidge | 0.0140 [0.0131, 0.0148] | 0.88 [0.86, 0.89] | 0.81 [0.75, 0.87] | 0.90 [0.86, 0.93] |

Full table, including RMSE: `results/norman2019/summary.md`.

**What this shows.**
1. **GlobalMean is a high floor on correlation.** Predicting the same profile
   for every perturbation already gives a Pearson delta of 0.50. A model
   reporting a Pearson delta of 0.6 on this data has not shown much. This is
   why the scaled score matters.
2. **HVGRidge closes about two thirds of the gap** between GlobalMean and a
   replicate experiment, overall. It uses no pretraining, only co-expression
   in the training cells.
3. **Unseen single-gene perturbations are the hard part.** HVGRidge closes
   under half the gap for them (0.46 scaled MAE).
4. **Pair scores are mostly the "seen singles" effect.** When a pair A+B is
   held out, A and B are usually still in training. Broken down:

   | singles of the pair in training | n pairs | scaled MAE | scaled Pearson delta |
   |---|---|---|---|
   | both | 68 | 0.91 [0.83, 0.99] | 0.96 [0.92, 1.01] |
   | one | 52 | 0.72 [0.62, 0.80] | 0.86 [0.82, 0.90] |
   | neither | 11 | 0.74 [0.45, 1.01] | 0.72 [0.48, 0.92] |

   With both singles seen, a linear model is close to replicate quality. The
   "neither" row is only 11 pairs, so its CI is too wide to conclude much.
   Any result on Norman pairs should be reported with this breakdown.

**Limitations.**
- One dataset and one cell line, with CRISPR *activation*; this says nothing
  about knockdown data or about new cell contexts (the VCC 2026 task).
- Half B has half the cells of a full profile, so all errors include more
  measurement noise than a full-profile evaluation would; the Replicate row
  measures exactly that noise level.
- Perturbation-level splits allow a held-out single gene to appear inside a
  training pair. A stricter split by target gene is not implemented yet.
- The 5,045 genes were preselected by GEARS as highly variable, so `k = 4096`
  in HVGRidge selects among already-variable genes.
- The HVGRidge penalty is chosen by leave-one-perturbation-out inside the
  training folds, which still lets a pair and its singles inform each other
  during that inner selection. The outer test sets are unaffected.

**Deviation from the spec.** `CLAUDE.md` §4.1 gives the baseline interface as
`fit(X_train, y_train, groups_train)` / `predict(X_test)`. For unseen
perturbations there is no test feature matrix (the only input is the
perturbation's label), so baselines implement `fit(X, perturbations, genes)` /
`predict(perturbations)`. See `baseline_first/baselines/base.py`.
