from glob import glob
from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def run(path: str = "./", port: str = "8080"):
    typer.echo("I will dockerize your ML app ⛴")
    path = Path(path)
    files = glob(str(path / "*"))
    typer.echo(f"Everything under {path} will be added to Docker image!")
    show_files = typer.confirm("Do you wish to see the files to be added?")
    if show_files:
        typer.echo(files)

    typer.echo("rest of the things are being implemented 👷‍♂️")
