"""Build the small offline test fixture from Norman 2019.

    python scripts/00_make_fixture.py

Writes tests/fixtures/norman2019_mini.h5ad: control plus a handful of single
and paired perturbations, a fixed number of cells each, on a few hundred genes.
Deterministic for a given seed, so the committed fixture can be regenerated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from baseline_first.data import CONTROL, load_norman2019

SEED = 0
N_PAIRS = 3
N_EXTRA_SINGLES = 2
CELLS_PER_GROUP = 40
N_GENES = 200
OUT = Path("tests/fixtures/norman2019_mini.h5ad")


def main() -> None:
    rng = np.random.default_rng(SEED)
    adata = load_norman2019()
    obs = adata.obs
    perts = set(obs["perturbation"].astype(str))

    # Pairs whose two single perturbations are also measured, so the fixture
    # supports baselines that combine single-gene effects.
    pairs = sorted(p for p in perts if "+" in p and all(g in perts for g in p.split("+")))
    pairs = [str(p) for p in rng.choice(pairs, size=N_PAIRS, replace=False)]
    singles = sorted({g for p in pairs for g in p.split("+")})
    # One single that was recorded under both 'GENE+ctrl' and 'ctrl+GENE', so the
    # fixture exercises the label merge.
    raw_labels = obs.groupby("perturbation", observed=True)["condition"].nunique()
    merged = sorted(set(raw_labels[raw_labels > 1].index) - set(singles))
    extra = [str(rng.choice(merged))]
    others = sorted(
        p for p in perts if "+" not in p and p != CONTROL and p not in {*singles, *extra}
    )
    extra += [str(p) for p in rng.choice(others, size=N_EXTRA_SINGLES - 1, replace=False)]
    groups = [CONTROL, *singles, *extra, *pairs]

    cells = []
    for group in groups:
        idx = np.flatnonzero(obs["perturbation"].to_numpy() == group)
        cells.extend(rng.choice(idx, size=min(CELLS_PER_GROUP, len(idx)), replace=False))
    sub = adata[np.sort(cells)].copy()

    # Genes: the perturbed genes that are measured, then the most variable rest.
    names = sub.var["gene_name"].astype(str).to_numpy()
    targets = {g for p in groups if p != CONTROL for g in p.split("+")}
    keep = [i for i, n in enumerate(names) if n in targets]
    X = sub.X.toarray()
    by_var = [i for i in np.argsort(-X.var(axis=0), kind="stable") if i not in keep]
    keep = sorted(keep + by_var[: N_GENES - len(keep)])
    sub = sub[:, keep].copy()

    sub.uns = {
        "baseline_first": {
            **adata.uns["baseline_first"],
            "fixture": "scripts/00_make_fixture.py",
            "fixture_seed": SEED,
        }
    }
    sub.obs = sub.obs[["condition", "perturbation", "is_control", "n_targets", "cell_line"]]
    sub.obs["perturbation"] = sub.obs["perturbation"].cat.remove_unused_categories()
    sub.obs["condition"] = sub.obs["condition"].cat.remove_unused_categories()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sub.write_h5ad(OUT, compression="gzip")
    print(f"wrote {OUT}: {sub.n_obs} cells x {sub.n_vars} genes, groups: {groups}")


if __name__ == "__main__":
    main()
