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

## Virtual Cell Challenge 2026: validation controls

| | |
|---|---|
| Source | `vcc datasets download controls` (vcc-cli 0.2.2), requires a registered challenge account |
| Archive | `vcc_2026_controls.zip`, 662,118,680 bytes, md5 `369f9ec7922c6201a98b49a44b452d98`; the CLI verified its crc32c |
| Panel | `vcc2026-val-1`, partition `val`, contexts A, B, C |
| Contents | `context_{A,B,C}.h5ad` (18,400 non-targeting cells each: 46 guides x 400 cells), `gene_names.csv` (18,533 genes, with a `gene_name` header), `pert_counts.csv` (the 300 target genes, same for all contexts), `manifest.json` |
| Values | raw integer counts (stored as float32), median about 20,000 UMI per cell |
| Downloaded | 2026-10-04 |
| Redistribution | Challenge data: not committed, not redistributed |

Per `manifest.json`, each context's reference holds 138,400 cells: these 18,400
controls plus 300 x 400 perturbed cells. The control cells are therefore the
same cells the scorer uses as the reference origin.

Marker genes hint at the cell types (an observation, not an identification):
A expresses CD3E (T-cell-like), B is VIM-high with COL1A1 (mesenchymal-like),
C expresses KRT5, TP63 and SOX2 (squamous-epithelial-like). Mean log-CPM
profiles correlate 0.75 to 0.84 between contexts.

## Not yet added

Replogle 2022 and a Tabula
Sapiens subset (see `CLAUDE.md` §3) will be added here as their loaders land.
