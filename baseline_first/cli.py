"""Command-line interface. Commands are added as each milestone lands."""

import typer

from baseline_first import __version__

app = typer.Typer(help="Run the dumb baseline before you train the big model.")


@app.callback()
def main() -> None:
    """Keep the CLI a command group even while it has a single command."""


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
