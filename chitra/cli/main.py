import typer

from chitra import __version__
from chitra.cli import app


@app.command()
def hello():
    typer.echo(f"Hey! You're running chitra version={__version__}")
