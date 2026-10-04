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

## M5: effective sample size

**Command:** `python scripts/02_effective_n_survey.py` → `results/effective_n/`.
Method, assumptions and failure modes: `docs/effective-n-method.md`.

| dataset | cells | conditions | cells / condition | median ICC [IQR] | design-effect n_eff | cells / n_eff |
|---|---|---|---|---|---|---|
| Norman 2019 | 91,205 | 237 | **385x** | 0.015 [0.004, 0.045] | 13,678 | 6.7x |
| VCC 2026 controls A, by guide (negative control) | 18,400 | 46 guides | – | 0.000 [0.000, 0.001] | 16,603 | 1.1x |

A model of perturbation effects trained on Norman has 237 examples, not 91,205.
The design effect answers a different question (independent cells for
estimating gene means) and is far less dramatic, because single-cell noise
keeps per-gene ICC low. The negative control behaves: non-targeting guides do
not differ.

## Virtual Cell Challenge 2026: what public training data covers

Facts from the official challenge files and Arc's public data (sources and
checksums in `data/MANIFEST.md`):

- The task: predict single-cell responses to 300 CRISPRi knockdowns in each
  of three new cell contexts (validation round), given only each context's
  control cells. Scoring: six metrics from `cell-eval2`, each scaled so that
  0 = the context's own mean perturbation response and 1 = a split-half
  replicate of the real experiment.
- **Coverage of the 300 targets by public perturbation data:**

  | source | targets covered |
  |---|---|
  | Replogle 2022 + Nadig 2025 essential screens (K562, RPE1, HepG2, Jurkat; 2,024 genes) | 0 / 300 |
  | Arc VCC 2025 support set, K562 genome-wide subset (184 perturbations) | 17 / 300 |
  | VCC 2025 H1 training data (150 perturbations) | 13 / 300, all within the 17 |

  So "look up the gene's effect elsewhere" is unavailable for 94% of targets.
  Any method has to predict effects for genes that were never perturbed in
  public data, in cell types it has never seen.

## Does a gene's effect transfer to another cell type?

**Command:** `python scripts/12_cross_context_transfer.py` →
`results/cross_context/`. 53 perturbations measured in all of K562, RPE1,
Jurkat and HepG2 (VCC 2025 support set). Each line is held out and predicted
from the other three; the held-out line's controls are known. All 53 target
genes are removed from every vector. The error ratio is
sum‖pred − true‖² / sum‖true‖², without the finite-cell noise correction of
the official metric (it needs single cells), so every method's error is
overstated by the same noise term.

| method | PDS (range over 4 held-out lines) | expression error ratio |
|---|---|---|
| NoChange | 0.50 | 1.00 |
| MeanResponse (average effect in the other lines) | 0.50 | 0.95 – 1.03 |
| SameGene (this gene's effect in the other lines) | **0.82 – 0.89** | 1.10 – 1.45 |
| ContextMean* (VCC 0 point; reads the test line) | 0.50 | 0.85 – 0.95 |

Which genes a knockdown moves is substantially conserved across these cell
types: a gene's own effect elsewhere identifies the right perturbation 82–89%
of the way. Its magnitude is not: unshrunk, it has more error than predicting
no change. But this route exists for only 17 of the 300 VCC targets.

## Unseen genes in an unseen cell type (the VCC setting, on public data)

**Command:** `python scripts/13_unseen_gene_unseen_context.py` →
`results/unseen_gene_unseen_context/`. Run overnight at commit `4166bb6`;
`run.json` says `-dirty` only because an untracked local
`.claude/settings.json` existed (no tracked file differed).

Data: K562, RPE1, HepG2, Jurkat (643,413 cells streamed from Arc's
harmonised file; 2 cells skipped as unrecoverable; 6,546 genes). Perturbed
genes are split into 5 folds; testing fold F in line L trains only on the
other lines **and** the other folds, with `assert_no_leakage` on both keys.

| held out | n perts | CoexpressionRidge PDS [95% CI] | error ratio: ridge / MeanResponse / NoChange / ContextMean* |
|---|---|---|---|
| HepG2 | 1,340 | 0.531 [0.514, 0.546] | 0.901 / 0.902 / 1.000 / 0.853 |
| Jurkat | 1,537 | 0.540 [0.526, 0.555] | 1.100 / 1.077 / 1.000 / 0.933 |
| K562 | 1,383 | 0.538 [0.522, 0.553] | 1.147 / 1.110 / 1.000 / 0.944 |
| RPE1 | 1,499 | 0.520 [0.504, 0.534] | 0.885 / 0.897 / 1.000 / 0.727 |

1. Co-expression in the held-out cell type's own controls carries a weak but
   real gene-specific signal: PDS 0.52–0.54, every CI above 0.5.
2. It does not reduce expression error beyond the plain average response.
3. The average response itself transfers poorly: worse than predicting no
   change for Jurkat and K562, and every transfer method is clearly worse
   than the context's own mean (the VCC 0 point), which no zero-shot method
   can see. A zero-shot submission should therefore be expected to score
   below 0 on expression error and only slightly above 0 on discrimination.
