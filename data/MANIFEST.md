# Data manifest

Raw data is downloaded to `data/raw/` (gitignored) on first use and verified by
checksum. Only the small fixture under `tests/fixtures/` is committed.

## Norman 2019 (CRISPRa Perturb-seq, K562)

| | |
|---|---|
| Paper | Norman, Horlbeck, Replogle et al. *Exploring genetic interaction manifolds constructed from rich single-cell phenotypes.* Science 365, 786–793 (2019). doi:10.1126/science.aax4438. Raw data: GEO GSE133344. |
| Copy used | GEARS preprocessed release (Roohani, Huang & Leskovec, *Nat. Biotechnol.* 2024), Harvard Dataverse doi:10.7910/DVN/Q2ZV3E, file `norman.zip` (datafile id 6154020, in dataset versions 2 and 3) |
| URL | https://dataverse.harvard.edu/api/access/datafile/6154020 |
| Licence | CC0 1.0 |
| Archive | `norman.zip`, 168,758,985 bytes, md5 `cdc41d6050e619c37fd9dd44d440e2b4` (matches the Dataverse record) |
| File read | `norman/perturb_processed.h5ad`, 2,228,610,012 bytes |
| Shape | 91,205 cells × 5,045 genes |
| Loader | `baseline_first.data.load_norman2019()` |
| Downloaded | 2026-10-03 |

**Why this copy.** It is the version most perturbation-prediction papers
benchmark on. The scPerturb copy on Zenodo was the first choice, but Zenodo
refused requests from the network this was built on (HTTP 403).

**What was done upstream (by GEARS, not by this repo).**
- `.X` is log-normalised expression. Checked on load: `expm1(X)` is a fixed
  multiple of `layers["counts"]` within each cell, and the implied library sizes
  are larger than the counts over the 5,045 kept genes. That is consistent with
  normalising each cell to a fixed total over the full transcriptome, applying
  `log1p`, then keeping 5,045 highly variable genes. The exact target sum is not
  recorded in the file.
- `layers["counts"]` holds integer UMI counts for the same 5,045 genes.
- `uns` carries GEARS's precomputed differentially expressed gene lists. This
  repo does not use them.

**Corrections applied on load.**
1. **Cell line.** GEARS labels every cell `A549`; Norman et al. used K562. The
   loader sets `cell_line = "K562"` and records the original in
   `uns["baseline_first"]["cell_line_corrected_from"]`.
2. **Duplicate perturbation labels (a leakage trap).** 47 single-gene
   perturbations appear under two raw labels, `GENE+ctrl` and `ctrl+GENE`.
   Both perturb the same gene. A split on the raw `condition` column can put
   one label in train and the other in test, which `assert_no_leakage` cannot
   see because the strings differ. The loader adds `perturbation`, which maps
   both forms (and pair orders `A+B` / `B+A`) to one canonical label. **Always
   split on `perturbation`.** `tests/test_data.py` demonstrates the leak.

After correction: 1 control group (7,353 cells), 105 single-gene and 131
two-gene perturbations, 237 groups in total.

## Test fixture: `tests/fixtures/norman2019_mini.h5ad`

Built by `python scripts/00_make_fixture.py` (seed 0) from the file above.
440 cells × 200 genes: control, 7 single-gene and 3 two-gene perturbations, 40
cells each. The pairs were chosen so that both of their single-gene
perturbations are included, and 6 of the singles carry both raw label forms.
Genes are the measured targets plus the most variable remaining genes in the
subset. 314 KB.

## Not yet added

Replogle 2022, the Arc Virtual Cell Challenge training data and a Tabula
Sapiens subset (see `CLAUDE.md` §3) will be added here as their loaders land.
