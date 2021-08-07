import os
from glob import glob
from pathlib import Path
from typing import List, Optional

import typer

import chitra

app = typer.Typer(name="builder")


def get_dockerfile() -> str:
    path = Path(os.path.dirname(chitra.__file__)) / "assets/API.Dockerfile"
    with open(path, "r") as fr:
        data = fr.read()
    return data


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
def create(
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

    dockerfile = get_dockerfile()
    dockerfile = dockerfile.replace("PORT", port)
    typer.echo(dockerfile)
    text_to_file(dockerfile, "Dockerfile")

    typer.echo(f"Building Docker {tag} 🐳")
    cmd = f"docker build --tag {tag} ."
    # cmd = shlex.quote(cmd)
    os.system(cmd)
