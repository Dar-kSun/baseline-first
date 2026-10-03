"""Loader tests. They run offline against the committed fixture."""

import hashlib
from pathlib import Path

import anndata as ad
import pandas as pd
import pytest

from baseline_first.data import CONTROL, STANDARD_COLUMNS
from baseline_first.data._download import fetch
from baseline_first.data.norman import canonical_perturbation, standardise
from baseline_first.splits import GroupSplitter, LeakageError, assert_no_leakage

FIXTURE = Path(__file__).parent / "fixtures" / "norman2019_mini.h5ad"


@pytest.fixture(scope="module")
def mini():
    return ad.read_h5ad(FIXTURE)


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ("ctrl", CONTROL),
        ("KLF1+ctrl", "KLF1"),
        ("ctrl+KLF1", "KLF1"),
        ("CEBPE+RUNX1T1", "CEBPE+RUNX1T1"),
        ("RUNX1T1+CEBPE", "CEBPE+RUNX1T1"),
    ],
)
def test_canonical_perturbation(condition, expected):
    assert canonical_perturbation(condition) == expected


def test_standardise_adds_columns_and_corrects_cell_line():
    obs = pd.DataFrame({"condition": ["ctrl", "KLF1+ctrl", "ctrl+KLF1", "A+B"]})
    obs.index = obs.index.astype(str)
    adata = standardise(ad.AnnData(obs=obs), {"url": "test"})
    assert list(adata.obs["perturbation"]) == [CONTROL, "KLF1", "KLF1", "A+B"]
    assert list(adata.obs["is_control"]) == [True, False, False, False]
    assert list(adata.obs["n_targets"]) == [0, 1, 1, 2]
    assert set(adata.obs["cell_line"]) == {"K562"}
    assert adata.uns["baseline_first"]["n_perturbations_merged"] == 1


def test_raw_condition_split_can_leak_but_perturbation_split_cannot(mini):
    """The trap the canonical column exists for.

    'GENE+ctrl' and 'ctrl+GENE' are different raw labels for the same knockout, so a
    split that is clean on 'condition' can still put one gene on both sides.
    """
    obs = mini.obs
    leaky = GroupSplitter("condition", test_size=0.5, seed=0)
    leaked = False
    for seed in range(20):
        leaky.seed = seed
        train, test = leaky.split(obs).apply(obs)  # clean as far as 'condition' goes
        try:
            assert_no_leakage(train, test, keys="perturbation")
        except LeakageError:
            leaked = True
            break
    assert leaked, "expected a condition-level split to leak at least one perturbation"

    for seed in range(20):
        split = GroupSplitter("perturbation", test_size=0.5, seed=seed).split(obs)
        train, test = split.apply(obs)
        assert_no_leakage(train, test, keys=["perturbation", "condition"])


def test_fixture_follows_the_standard_schema(mini):
    for column in STANDARD_COLUMNS:
        assert column in mini.obs
    assert mini.obs["is_control"].any()
    assert set(mini.obs.loc[mini.obs["is_control"], "perturbation"]) == {CONTROL}
    assert mini.obs["n_targets"].max() == 2
    assert mini.obs["perturbation"].nunique() >= 5
    assert 100 <= mini.n_obs <= 1000


def test_fixture_matches_canonical_mapping(mini):
    """Re-deriving 'perturbation' from the raw labels gives what is stored."""
    rederived = mini.obs["condition"].astype(str).map(canonical_perturbation)
    assert (rederived == mini.obs["perturbation"].astype(str)).all()


def test_fetch_rejects_a_bad_checksum(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"not the data you are looking for")
    with pytest.raises(ValueError, match="md5 mismatch"):
        fetch(source.as_uri(), tmp_path / "out.bin", md5="0" * 32)
    assert not (tmp_path / "out.bin").exists()
    assert not (tmp_path / "out.bin.part").exists()


def test_fetch_downloads_and_reuses_cache(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    md5 = hashlib.md5(b"payload").hexdigest()
    dest = fetch(source.as_uri(), tmp_path / "out.bin", md5=md5)
    assert dest.read_bytes() == b"payload"
    source.unlink()  # a second call must not need the source
    assert fetch(source.as_uri(), dest, md5=md5) == dest
