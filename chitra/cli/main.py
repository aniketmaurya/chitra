import typer

from chitra import __version__
from chitra.cli import builder

app = typer.Typer(
    name="chitra CLI ✨",
    add_completion=False,
)

app.add_typer(
    builder.app,
    name="builder",
    help="Auto Build Docker image for Chitra Model Server 🐳",
)


@app.command()
def version():
    typer.echo(f"Hey 👋! You're running chitra version={__version__} ✨")
