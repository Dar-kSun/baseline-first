from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from typer.testing import CliRunner

from baseline_first.cli import app

FIXTURE = Path(__file__).parent / "fixtures" / "norman2019_mini.h5ad"
runner = CliRunner()


def test_card_on_the_fixture(tmp_path):
    result = runner.invoke(
        app, ["card", "--dataset", "fixture", "--folds", "3", "--out", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    card = (tmp_path / "fixture" / "verdict.md").read_text(encoding="utf-8")
    assert "distinct conditions | 11" in card
    assert "The bar a bigger model must clear" in card
    assert (tmp_path / "fixture" / "verdict.png").stat().st_size > 0


def write_split(tmp_path, leaky: bool) -> Path:
    adata = ad.read_h5ad(FIXTURE)
    perts = adata.obs["perturbation"].astype(str).to_numpy()
    if leaky:
        split = np.where(np.arange(adata.n_obs) % 5 == 0, "test", "train")
    else:
        split = np.where(np.isin(perts, ["ETS2", "FOXF1+HOXB9"]), "test", "train")
    # A minimal file of categoricals only: anndata < 0.11 (used on Python 3.10)
    # cannot write the nullable string columns newer anndata reads the fixture into.
    obs = pd.DataFrame(
        {"perturbation": pd.Categorical(list(perts)), "split": pd.Categorical(list(split))},
        index=pd.Index([f"c{i}" for i in range(len(perts))], dtype=object),
    )
    path = tmp_path / "split.h5ad"
    ad.AnnData(obs=obs).write_h5ad(path)
    return path


def test_check_leakage_fails_on_a_cell_level_split(tmp_path):
    result = runner.invoke(app, ["check-leakage", str(write_split(tmp_path, leaky=True))])
    assert result.exit_code == 1
    assert "LEAKAGE" in result.output


def test_check_leakage_passes_on_a_group_split(tmp_path):
    result = runner.invoke(app, ["check-leakage", str(write_split(tmp_path, leaky=False))])
    assert result.exit_code == 0, result.output
    assert "no leakage" in result.output
