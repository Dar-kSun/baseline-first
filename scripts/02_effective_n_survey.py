"""Effective sample size of each dataset, and how much the raw cell count overstates it.

    python scripts/02_effective_n_survey.py

Datasets:
  norman2019       Norman et al. 2019, grouped by canonical perturbation
  vcc2026_ctrl_A/B/C  VCC 2026 control cells, grouped by their 46 non-targeting
                   guides. A negative control: guides that target nothing
                   should not differ, so ICC should be near 0 and the
                   inflation factor near 1.

All expression is log1p of counts per 10,000 (Norman's X is used as GEARS
provides it, which is log-normalised; see data/MANIFEST.md). Method:
baseline_first.effective_n and docs/effective-n-method.md.

Writes results/effective_n/{summary.csv, summary.md}.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from baseline_first.data import load_norman2019
from baseline_first.effective_n import effective_n

OUT = Path("results/effective_n")
VCC = Path("data/raw/vcc2026")


def lognorm(counts: sp.csr_matrix) -> sp.csr_matrix:
    totals = np.asarray(counts.sum(axis=1)).ravel()
    scaled = sp.diags(1e4 / np.where(totals > 0, totals, 1)) @ counts
    scaled.data = np.log1p(scaled.data)
    return scaled.tocsr()


def main() -> None:
    rows = []
    norman = load_norman2019()
    rows.append(
        {
            "dataset": "norman2019",
            "grouped_by": "perturbation",
            **effective_n(norman.X, norman.obs["perturbation"]).as_dict(),
        }
    )
    del norman
    for c in "ABC":
        ctrl = ad.read_h5ad(VCC / f"context_{c}.h5ad")
        rows.append(
            {
                "dataset": f"vcc2026_ctrl_{c}",
                "grouped_by": "ntc_id (46 guides)",
                **effective_n(lognorm(sp.csr_matrix(ctrl.X)), ctrl.obs["ntc_id"]).as_dict(),
            }
        )
        print(rows[-1], flush=True)

    summary = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "summary.csv", index=False, float_format="%.6g")
    lines = [
        "| dataset | grouped by | cells | conditions | cells / condition "
        "| median ICC [IQR] | design-effect n_eff | cells / n_eff |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in summary.itertuples():
        lo, hi = r.icc_iqr
        lines.append(
            f"| {r.dataset} | {r.grouped_by} | {r.n_cells:,} | {r.n_conditions} "
            f"| {r.inflation_conditions:.0f}x "
            f"| {r.icc_median:.3f} [{lo:.3f}, {hi:.3f}] | {r.n_eff:,.0f} | {r.inflation:.1f}x |"
        )
    text = "\n".join(lines) + "\n"
    (OUT / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
