import typer

from chitra import __version__

app = typer.Typer()


@app.command()
def hello():
    typer.echo(f"Hey 👋! You're running chitra version={__version__} ✨")


@app.command()
def bye():
    typer.echo("bye bye! 😊")
