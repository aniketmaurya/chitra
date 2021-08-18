import os
import shlex
import subprocess
from glob import glob
from pathlib import Path
from typing import List, Optional

import typer

import chitra

app = typer.Typer(
    help="""Auto Builds docker image for chitra ModelServer.
    path should contain a `main.py` file which will have an object of type `chitra.serve.ModelServer` and
    its name should be `app`. If you have any external Python dependency then create a `requirements.txt` file and
    keep in the same directory."""
)


def get_dockerfile() -> str:
    path = Path(os.path.dirname(chitra.__file__)) / "assets/API.Dockerfile"
    with open(path, "r") as fr:
        data = fr.read()
    return data


def file_check(files: List) -> None:
    files = map(os.path.basename, files)
    if "requirements.txt" not in files:
        raise UserWarning("requirements.txt not found!")

    if "main.py" not in files:
        raise UserWarning(
            "main.py not found! Your main.py should contain app \
            object of type chitra.serve.ModelServer"
        )


def text_to_file(text: str, path: str):
    with open(path, "w") as fw:
        fw.write(text)


@app.command()
def create(
    path: str = "./",
    port: Optional[str] = None,
    tag: Optional[str] = None,
):
    """
    Auto-builds Docker image for chitra ModelServer

    Args:

        path: file-location of main.py

        port: port to expose

        tag: tag of docker image

    """
    if not port:
        port = "8080"
    if not tag:
        tag = "chitra-server"

    path = Path(path)
    files = glob(str(path / "*"))
    file_check(files)
    style_path = typer.style(str(path), fg=typer.colors.GREEN)
    typer.echo(f"Everything under {style_path} will be added to Docker image!")
    show_files = typer.confirm("Do you wish to see the files to be added?")
    if show_files:
        typer.echo(files)

    dockerfile = get_dockerfile()
    dockerfile = dockerfile.replace("PORT", port)
    if typer.confirm("Show Dockerfile"):
        typer.echo(dockerfile)
    text_to_file(dockerfile, "Dockerfile")

    typer.echo(f"Building Docker {tag} 🐳")
    cmd = f"docker build --tag {tag} ."
    cmd = shlex.split(cmd)
    process = subprocess.run(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    if process.returncode == 0:
        typer.secho(
            "Docker image has been created for your app! Check the image using ",
            nl=False,
            fg=typer.colors.GREEN,
        )
        typer.secho("docker images", fg=typer.colors.BRIGHT_BLUE)
    else:
        typer.secho("Docker build failed!", fg=typer.colors.RED)
