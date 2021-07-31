import typer
import uvicorn

from chitra import __version__
from chitra.cli import dockerize

app = typer.Typer(name="chitra CLI ✨")

app.add_typer(dockerize.app, name="dockerizer")


@app.command()
def version():
    typer.echo(f"Hey 👋! You're running chitra version={__version__} ✨")
