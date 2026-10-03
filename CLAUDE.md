# CLAUDE.md — `baseline-first`

> Run the dumb baseline before you train the big model.
> **Mission: before anyone fine-tunes a single-cell foundation model, tell them what a ten-line baseline scores on their data, and how much information their dataset actually contains.**

---

## 0. Context for you, the agent

Built by Aryan Sinha (3rd-year B.Tech Biotechnology, IIT Kharagpur), and intended for a real submission to Arc Institute's 2026 Virtual Cell Challenge. Reviewers will be people who evaluate research professionally.

- **Honest beats impressive.** This repo's entire thesis is that the field over-claims. It cannot itself over-claim.
- **Every number in the README must be reproducible** from a command here.
- Leakage is the enemy. This repo exists partly to catch it, so it must be ruthless about its own splits.
- Small and finished beats large and half-built.

Small commits, real messages, never backdated.

---

## 1. The problem, stated precisely

A cell has ~20,000 genes, each expressed at some level. Single-cell RNA sequencing measures all of them, per cell, across hundreds of thousands of cells. The ambition on top is the **virtual cell**: predict what happens when you knock out gene X, without running the experiment.

So the field built foundation models — scGPT (33M cells), Geneformer (30M), scFoundation (50M+) — on the assumption that scaling works here as it did for language.

**It keeps losing to trivial methods.**

- An **Oct 2025** evaluation on predicting cancer outcomes: selecting the **4,096 most variable genes** and feeding them to a basic classifier scored **mean AUPRC ≈ 0.90**, beating the best foundation model (scGPT-cancer, ≈ 0.89). No pretraining.
- In Arc's **2025 Virtual Cell Challenge** (1,200+ teams, 114 countries), a team at Turbine placed a **regression method first published in 1970** near the top. Their diagnosis is the sharpest line in the field: despite hundreds of thousands of cells, the challenge had roughly **300 effective data points**, because cells from the same condition are near-replicates. *"Your model should match the complexity of your data, not the complexity of your problem."*
- Arc's own 2025 wrap-up named open gaps in model accuracy, metric design, and biological generalisation, and noted that almost all entries lost to the baseline on MAE.

**The two gaps.**

1. **Nobody runs the baseline first.** There is no standard tool that, before training, reports what the trivial approach scores and what bar a complex model must clear to justify itself.
2. **Nobody reports effective sample size.** A paper says "500,000 cells," which sounds enormous and is statistically often nearer 300. The raw cell count is a near-meaningless denominator.

**What this repo is.** One command, run before any deep learning, that answers both.

---

## 2. Scope

### v0.1 — must ship
1. **Honest baselines**: predict-the-mean, per-condition mean, highly-variable-genes + ridge, PCA + linear.
2. **Group-aware splitting** with a leakage detector that *raises an error* rather than warning.
3. **Effective sample size estimator** with a stated, documented method.
4. **Verdict card**: a one-page summary — baseline score, n_eff, and the bar a bigger model must clear.
5. CLI + README with real numbers from public datasets.

### v0.2 — if time allows
6. Reproduce a published "baseline beats foundation model" result end to end.
7. Cross-dataset summary: median gap between foundation models and baselines; factor by which raw cell count overstates n_eff.
8. VCC-compatible metric implementations.

### Explicitly out of scope
- Training a foundation model.
- Claiming foundation models are useless. **The honest claim is narrower: on these datasets, under these metrics, with these splits, the gap was X.** Say exactly that.

---

## 3. Data

Load via `pertpy` / `scperturb` / `cellxgene-census` where possible. Cache to `data/raw/` (gitignored), manifest in `data/MANIFEST.md`.

| Dataset | Why |
|---|---|
| Norman 2019 (Perturb-seq, K562) | Standard perturbation benchmark; many conditions |
| Replogle 2022 (genome-scale Perturb-seq) | Large, well-structured, good n_eff contrast |
| Arc VCC training data | Direct relevance to the live challenge |
| Tabula Sapiens (subset) | For the annotation-style baselines |

Commit a small `.h5ad` fixture (a few hundred cells, a handful of conditions) under `tests/fixtures/` so the suite runs offline.

---

## 4. The three components — exact specifications

### 4.1 Baselines (`baseline_first/baselines/`)

Common interface:
```python
class Baseline(Protocol):
    name: str
    def fit(self, X_train, y_train, groups_train) -> None: ...
    def predict(self, X_test) -> np.ndarray: ...
```

Implement at minimum:
- `GlobalMean` — predict the training mean for every gene. The floor. **Many published models fail to clear this.**
- `ConditionMean` — predict the mean of the nearest/most similar training condition.
- `HVGRidge` — top-k highly variable genes → ridge regression. k configurable, default 4096 (matching the cited result).
- `PCALinear` — PCA → linear model.
- `NearestNeighbour` — in a simple embedding space.

Each must run in well under a minute on the fixture.

### 4.2 Splitting and leakage detection (`baseline_first/splits.py`)

**This is the most important module in the repo.** Half the inflated results in this literature come from splitting cells randomly, so the same perturbation appears in train and test.

- `GroupSplitter(group_key=...)` — splits by perturbation, donor, or cell line. Never by cell.
- `assert_no_leakage(train, test, keys=[...])` — **raises** `LeakageError` if any group value appears on both sides. Not a warning. An exception.
- Run this assertion inside the evaluation loop, not just at setup, so a resplit can't bypass it.
- Write tests that deliberately construct a leaky split and assert the error fires.

### 4.3 Effective sample size (`baseline_first/effective_n.py`)

Report **two** estimates, because there is no single canonical definition, and say so plainly in the docs:

1. **Unique-condition count** — the simple, defensible number: how many genuinely distinct perturbation × context combinations exist.
2. **Variance-decomposition estimate** — decompose total variance into within-condition and between-condition components; compute a design-effect correction
   `n_eff ≈ n / (1 + (m − 1)·ICC)`
   where *m* is mean cells per condition and ICC is the intraclass correlation. Document the estimator, its assumptions, and when it breaks.

Report the **inflation factor**: `raw_cells / n_eff`. That single number is the repo's headline.

**Do not invent a novel information-theoretic metric.** Use standard, citable statistics. The credibility of this repo depends on the estimator being boring and correct.

---

## 5. Metrics

In `baseline_first/metrics.py`. Where the VCC defines a metric, match its definition and cite it:

- MAE and RMSE on expression deltas.
- Pearson/Spearman on differential-expression vectors.
- Perturbation discrimination: can the model tell which perturbation was applied?
- AUPRC for classification-style tasks.

Every metric reported with bootstrap CIs over **groups** (not cells — bootstrapping over cells fakes precision, which is the exact error this repo is about).

---

## 6. Repo layout

```
baseline_first/
  __init__.py
  baselines/        # base.py + one file per baseline
  data/             # loaders per dataset
  splits.py         # GroupSplitter, assert_no_leakage, LeakageError
  effective_n.py
  metrics.py
  verdict.py        # renders the verdict card (markdown + figure)
  cli.py            # typer: baseline-first run / card / check-leakage
scripts/
  01_run_baselines.py
  02_effective_n_survey.py
  03_vcc_submission.py      # v0.2
tests/
  fixtures/
  test_splits.py            # leakage tests — the most important test file
docs/
  effective-n-method.md     # the estimator, assumptions, limitations
  findings.md
results/                    # committed summaries + figures
README.md
pyproject.toml
```

---

## 7. Build order

- **M0** — Skeleton, pyproject, ruff, pytest, CI on push, MIT licence, `.gitignore`, README stub marked "v0.1, in development."
- **M1** — `splits.py` + leakage tests. **Build this before any model code.** The tests must include a deliberately leaky split that raises.
- **M2** — Data loaders + fixture + manifest.
- **M3** — `GlobalMean` and `HVGRidge` running end to end on one public dataset with group splits. Record numbers in `docs/findings.md`.
- **M4** — Remaining baselines; results table across datasets.
- **M5** — `effective_n.py` + `docs/effective-n-method.md`. Report the inflation factor per dataset.
- **M6** — Verdict card + CLI.
- **M7** — README with real numbers and figures. Tag `v0.1.0`.
- **M8+** (v0.2) — Reproduce a published comparison; cross-dataset survey; VCC entry.

---

## 8. The Virtual Cell Challenge angle

Arc Institute's 2026 Virtual Cell Challenge is a zero-shot task: predict how unseen cell lines respond to gene knockdowns.

**Before writing anything about it in the README or an application: open the official Arc Institute challenge page and verify the current dates, task definition, and rules.** Reported timings (test set release ~22 Oct, submissions ~5 Nov) must be confirmed, not assumed, and they may have changed.

If it is open and the rules permit, enter the baselines. A public leaderboard is the cleanest possible evidence that this tool does something real — and given the 2025 results, honest baselines are genuinely competitive.

In the README, state participation factually: entered, with what, and the result once known. **Never imply a placement that hasn't happened.**

---

## 9. README requirements

1. One-sentence what and why, with the headline inflation factor as a number.
2. Signature figure: baseline vs foundation-model scores, or raw cells vs n_eff across datasets.
3. Quickstart: install, then one command producing a verdict card.
4. What a verdict card contains, with an example.
5. Results table: dataset × baseline × score × n_eff × inflation factor, with CIs.
6. **Limitations**: which datasets, which metrics, which split definitions, where the n_eff estimator's assumptions fail, and the fact that a baseline winning on these tasks doesn't mean foundation models are useless everywhere.
7. Status: "v0.1 — active development."
8. Citations to everything in §1.

---

## 10. Honesty checklist

- [ ] Every number reproducible from a repo command.
- [ ] All splits group-aware; leakage assertion active in every evaluation path.
- [ ] CIs bootstrapped over groups, not cells.
- [ ] Claims scoped to the datasets and metrics actually tested.
- [ ] The n_eff estimator's assumptions documented, including where it fails.
- [ ] Foundation models credited where they do win; no blanket dismissal.
- [ ] Competition participation described factually, never aspirationally.

---

## 11. Definition of done for v0.1

A reviewer clones, runs one command on a public dataset, and gets a verdict card saying: here's the baseline score, here's your effective sample size, here's the inflation factor, here's the bar a bigger model has to clear — with the leakage check provably enforced.
