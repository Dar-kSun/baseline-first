# baseline-first

> Run the dumb baseline before you train the big model.

Before you fine-tune a single-cell foundation model, `baseline-first` tells you
what a ten-line baseline scores on your data, and how much information your
data actually contains. On Norman 2019, **91,205 cells hold 237 distinct
perturbations: the cell count overstates the examples a perturbation model
learns from by 385x**, and a ridge regression already closes two thirds of
the gap between a trivial mean and a repeat of the experiment.

**Status: v0.1, active development.** Every number below comes from a command
in this repo; the command is given next to it.

![Verdict card figure for Norman 2019](results/norman2019/verdict.png)

## Quickstart

```bash
git clone https://github.com/Dar-kSun/baseline-first && cd baseline-first
python -m venv .venv && .venv/Scripts/activate   # .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"
baseline-first card --dataset fixture      # 440-cell bundled fixture, seconds, offline
baseline-first card --dataset norman2019   # downloads ~170 MB (CC0), a few minutes
```

`card` writes `results/<dataset>/verdict.md` and `verdict.png`.
`baseline-first check-leakage data.h5ad --split-col split --key perturbation`
exits with an error if any perturbation appears in both train and test.

## What a verdict card says

1. **How much data there is:** cells, distinct conditions, and a design-effect
   effective sample size, with how much the cell count overstates each.
2. **What trivial baselines score**, under splits that hold out whole
   perturbations, with 95% CIs bootstrapped over perturbations (never cells):
   - `GlobalMean`: the same average profile for every perturbation (score 0).
   - `HVGRidge`: each target gene described by its co-expression with the top
     variable genes, then ridge regression.
   - `Replicate`: half of each perturbation's cells predicting the other half
     (score 1): the noise ceiling.
3. **The bar a bigger model must clear:** the upper end of the best baseline's
   95% CI, per metric.

Example: [`results/norman2019/verdict.md`](results/norman2019/verdict.md).

## Results

### Baselines and data size (Norman 2019, K562 CRISPRa, 5-fold CV over perturbations)

| method | MAE | Pearson delta | scaled MAE [95% CI] | scaled Pearson delta [95% CI] |
|---|---|---|---|---|
| GlobalMean | 0.0187 | 0.50 | 0 | 0 |
| HVGRidge | 0.0131 | 0.78 | **0.68** [0.62, 0.73] | **0.70** [0.64, 0.76] |
| Replicate (split-half) | 0.0104 | 0.89 | 1 | 1 |

| cells | distinct conditions | cells / condition | design-effect n_eff | cells / n_eff |
|---|---|---|---|---|
| 91,205 | 237 | **385x** | 13,678 | 6.7x |

Commands: `python scripts/01_run_baselines.py --dataset norman2019`,
`python scripts/02_effective_n_survey.py`. Details and the single-gene vs
pair breakdown: [`docs/findings.md`](docs/findings.md). Note that predicting one
profile for everything already reaches a Pearson delta of 0.50; a raw
correlation without a baseline means little.

### Predicting unseen genes in unseen cell types (the Virtual Cell Challenge setting)

On public CRISPRi screens (K562, RPE1, HepG2, Jurkat; 643,413 cells), holding
out a cell line **and** the perturbed genes at once
(`python scripts/13_unseen_gene_unseen_context.py`):

| | perturbation discrimination | expression error vs. predicting no change |
|---|---|---|
| a gene's own effect, measured in other cell types* (separate run, 53 genes) | 0.82 – 0.89 | 1.10 – 1.45x |
| co-expression ridge (no data on the gene's perturbation) | 0.52 – 0.54 | 0.89 – 1.15x |
| average response from other cell types | 0.50 | 0.90 – 1.11x |
| the held-out line's own average (the VCC "0", unreachable zero-shot) | 0.50 | 0.73 – 0.94x |

\* Only possible when the gene was perturbed elsewhere: true for 17 of the 300
VCC 2026 targets (`scripts/12_cross_context_transfer.py`; data coverage in
[`data/MANIFEST.md`](data/MANIFEST.md)).

Which genes a knockdown moves is substantially conserved across cell types,
but its size is not, and without data on the gene itself, co-expression gives
only a weak signal.

## Virtual Cell Challenge 2026

**Entered.** `arV1-MeanResponse` was submitted to the **final round** on
1 November 2026 (entry `Hu8TAlvVsGrmdmZt95Mo`, panel `vcc2026-test-1`,
contexts D, E and F; 360,000 cells over 18,533 genes, 120,000 per context).
The method is the scaled average response measured in other cell lines,
trained on four public CRISPRi screens (K562, RPE1, HepG2, Jurkat) over the
6,197 genes they share with the VCC gene axis. Target genes were never looked
up, only described by co-expression. Final-phase scores stay hidden until the
winners are announced, so **no score is recorded here and no placement is
implied**; the round's own submissions close on 5 November 2026. Full build
provenance, including a power loss during the first `vcc prep` run and the
standalone re-run that replaced it, is in
[`results/vcc_submission/arV1-MeanResponse.json`](results/vcc_submission/arV1-MeanResponse.json).

**Local rehearsal with the official metrics.** The challenge's own scorer
(`cell-eval2`, `vcc2026` preset) was run on a challenge-style reference built
from the public VCC 2025 H1 data, with methods trained only on four other cell
lines and never on the target genes
(`python scripts/15_local_vcc_score_h1.py`). Overall scaled score (0 = the
context's own mean response, 1 = a replicate experiment):

| prediction | overall |
|---|---|
| no change | −0.02 |
| average response from other cell lines | 0.06 |
| co-expression ridge | 0.06 |
| no change, but cells thinned to 30% of reads | **0.12** |

Zero-shot transfer from public screens barely moves the score. And thinning
the predicted cells' reads lifts even a prediction with no effect at all,
because thinned cells produce spurious differential-expression calls against
full-depth controls: a weakness of the metrics worth knowing about, not a
better prediction. Details and caveats (the local anchors differ from the
official ones on two metrics) are in [`docs/findings.md`](docs/findings.md).

**The entry was submitted at full depth, deliberately, but the depth result
should not be read as a claim about the official scorer.** On this local
rehearsal a thinned entry scores twice what a full-depth one does, and a
thinned *no-change* entry scores the same as a thinned model. Both of those
gains run through direction fidelity and Jaccard, which are the same two
metrics whose local 0-anchors sit far from the official bundles (0.017 here
against 0.505-0.522 there). So the size of the effect, and possibly the effect
itself, may belong to this reference rather than to the challenge's. It was
treated as a reason to be conservative, not as a result: full depth was
entered. The mean response and the co-expression ridge tie at 0.06 here, and
the simpler of the two was entered.

## Limitations

- **Datasets.** Baseline results are on one CRISPRa dataset (Norman 2019,
  K562). Transfer results are on four CRISPRi cell lines' essential-gene
  screens and are computed on pseudobulks without the official metrics'
  finite-cell corrections.
- **Metrics.** This repo implements MAE, RMSE, Pearson delta and the VCC
  perturbation discrimination score. Its four differential-expression metrics
  are not reimplemented here: where they appear above, they come from the
  challenge's own `cell-eval2`, called by `scripts/15_local_vcc_score_h1.py`.
- **Splits.** Perturbation-level splits allow a held-out single gene to appear
  inside a training pair; Norman pair scores depend strongly on whether their
  single-gene perturbations were seen (see `docs/findings.md`).
- **Effective sample size.** The design-effect estimate depends on a per-gene
  ICC summarised by its median, assumes a normal random-effects model, and is
  depressed by single-cell noise. See
  [`docs/effective-n-method.md`](docs/effective-n-method.md).
- **Scope of the claim.** A baseline matching or beating a foundation model on
  these tasks, metrics and splits does not mean foundation models are useless
  elsewhere; no foundation model has been run in this repo.

## Citations

- Elmarakeby H, Roman A, Johri S, Van Allen EM. Empirical evaluation of
  single-cell foundation models for predicting cancer outcomes. *bioRxiv*
  2025.10.31.685892 (2025). HVG baseline (4,096 genes) mean AUPRC ≈ 0.90 ±
  0.15 vs. scGPT [cancer] ≈ 0.89 ± 0.11.
- Turbine. [How did we get a regression model to the top of the Virtual Cell
  Challenge](https://turbine.ai/how-did-we-get-a-regression-model-to-the-top-of-the-virtual-cell-challenge/)
  (about 300 effective data points: one per condition).
- Arc Institute. [Virtual Cell Challenge 2025 wrap-up](https://arcinstitute.org/news/virtual-cell-challenge-2025-wrap-up);
  [the 2026 challenge](https://arcinstitute.org/news/virtual-cell-challenge-2026);
  scoring code and metric definitions: [cell-eval2](https://github.com/ArcInstitute/cell-eval2).
- Ahlmann-Eltze C, Huber W, Anders S. Deep-learning-based gene perturbation
  effect prediction does not yet outperform simple linear baselines.
  *Nature Methods* (2025).
- Norman TM et al. Exploring genetic interaction manifolds constructed from
  rich single-cell phenotypes. *Science* 365, 786–793 (2019).
- Replogle JM et al. Mapping information-rich genotype-phenotype landscapes
  with genome-scale Perturb-seq. *Cell* 185, 2559–2575 (2022).
- Nadig A et al. Transcriptome-wide analysis of differential expression in
  perturbation atlases. *Nature Genetics* (2025). doi:10.1038/s41588-025-02169-3.
- Roohani Y, Huang K, Leskovec J. Predicting transcriptional outcomes of novel
  multigene perturbations with GEARS. *Nature Biotechnology* (2024).
- Kish L. *Survey Sampling*. Wiley (1965).

## Licence

MIT. See [LICENSE](LICENSE). Data keep their own licences (see
`data/MANIFEST.md`); challenge data are not redistributed.
