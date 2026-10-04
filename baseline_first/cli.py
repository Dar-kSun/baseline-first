"""Command-line interface: baseline-first card / check-leakage / version."""

from __future__ import annotations

from pathlib import Path

import typer

from baseline_first import __version__

app = typer.Typer(help="Run the dumb baseline before you train the big model.")

DATASETS = ("norman2019", "fixture")
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "norman2019_mini.h5ad"


@app.callback()
def main() -> None:
    """Keep the CLI a command group."""


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command()
def card(
    dataset: str = typer.Option("norman2019", help=f"one of {', '.join(DATASETS)}"),
    folds: int = typer.Option(5, help="cross-validation folds over perturbations"),
    seed: int = typer.Option(0),
    out: Path = typer.Option(Path("results"), help="results directory"),
) -> None:
    """Run the baselines and effective-n estimate, and write a verdict card."""
    import anndata as ad

    from baseline_first.baselines import GlobalMean, HVGRidge
    from baseline_first.data import load_norman2019
    from baseline_first.effective_n import effective_n
    from baseline_first.evaluate import evaluate, summarise
    from baseline_first.verdict import make_card

    if dataset not in DATASETS:
        raise typer.BadParameter(f"unknown dataset {dataset!r}; choose from {DATASETS}")
    adata = ad.read_h5ad(FIXTURE) if dataset == "fixture" else load_norman2019()
    for layer in [name for name in adata.layers if name is not None]:
        del adata.layers[layer]
    typer.echo(f"{dataset}: {adata.n_obs:,} cells; evaluating baselines ({folds} folds)...")
    results = evaluate(adata, [GlobalMean(), HVGRidge()], n_folds=folds, seed=seed)
    summary = summarise(results, seed=seed)
    n = effective_n(adata.X, adata.obs["perturbation"])
    path = make_card(
        dataset,
        summary,
        n,
        out / dataset,
        command=f"baseline-first card --dataset {dataset} --folds {folds} --seed {seed}",
        split=f"{folds}-fold cross-validation over perturbations; control always in training",
    )
    typer.echo(f"wrote {path}")


@app.command("check-leakage")
def check_leakage(
    h5ad: Path = typer.Argument(..., help="AnnData file with a train/test column in obs"),
    split_col: str = typer.Option("split", help="obs column holding 'train' / 'test'"),
    keys: list[str] = typer.Option(["perturbation"], "--key", help="group keys to check"),
) -> None:
    """Fail (exit 1) if any group value appears in both train and test."""
    import anndata as ad

    from baseline_first.splits import LeakageError, assert_no_leakage

    obs = ad.read_h5ad(h5ad, backed="r").obs
    try:
        assert_no_leakage(obs[obs[split_col] == "train"], obs[obs[split_col] == "test"], keys)
    except LeakageError as err:
        typer.echo(f"LEAKAGE: {err}", err=True)
        raise typer.Exit(1) from err
    typer.echo(f"no leakage on {', '.join(keys)}")


if __name__ == "__main__":
    app()
