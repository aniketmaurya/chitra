import typer

from chitra import __version__
from chitra.cli import dockerize

app = typer.Typer(name="chitra CLI ✨")

app.add_typer(dockerize.app, name="dockerizer")


@app.command()
def hello():
    typer.echo(f"Hey 👋! You're running chitra version={__version__} ✨")


@app.command()
def bye():
    typer.echo("bye bye! 😊")
