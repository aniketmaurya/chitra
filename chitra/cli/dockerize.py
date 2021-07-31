import os
import shlex
from glob import glob
from pathlib import Path
from typing import List, Optional

import typer

from chitra.utility import docker_templates as template

app = typer.Typer()


def file_check(files: List) -> None:
    if "requirements.txt" not in files:
        raise UserWarning("requirements.txt not found!")

    if "main.py" not in files:
        raise UserWarning(
            "main.py not found! Your main.py should contain app object of type chitra.serve.ModelServer"
        )


def text_to_file(text: str, path: str):
    with open(path, "w") as fw:
        fw.write(text)


@app.command()
def run(
    path: str = "./",
    port: Optional[str] = None,
    tag: Optional[str] = None,
):
    if not port:
        port = "8080"
    if not tag:
        tag = "chitra-server"

    path = Path(path)
    files = glob(str(path / "*"))
    file_check(files)

    typer.echo(f"Everything under {path} will be added to Docker image!")
    show_files = typer.confirm("Do you wish to see the files to be added?")
    if show_files:
        typer.echo(files)

    dockerfile = template.API_DOCKERFILE
    dockerfile = dockerfile.replace("PORT", port)
    typer.echo(dockerfile)
    text_to_file(dockerfile, "Dockerfile")

    typer.echo(f"Building Docker {tag} ⛴")
    cmd = shlex.quote(f"docker build --tag {tag} .")
    os.system(cmd)
