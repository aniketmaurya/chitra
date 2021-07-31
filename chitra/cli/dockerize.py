from glob import glob
from pathlib import Path
from typing import List, Optional

import typer

from chitra.utility import docker_templates as template
from chitra.utility import docker_utility as du

app = typer.Typer()


def file_check(files: List) -> None:
    if "requirements.txt" not in files:
        raise UserWarning("requirements.txt not found!")
    if "configs.py" not in files:
        raise UserWarning("configs.py not found")
    if "main.py" not in files:
        raise UserWarning(
            "main.py not found! Your main.py should contain app object of type chitra.serve.ModelServer"
        )


@app.command(help="Dockerize chitra.serve.model_server")
def run(
    path: str = "./",
    port: Optional[str] = None,
    tag: Optional[str] = None,
    docker_kwargs: Optional[dict] = None,
):
    if not port:
        port = "8080"

    path = Path(path)
    files = glob(str(path / "*"))
    file_check(files)

    typer.echo(f"Everything under {path} will be added to Docker image!")
    show_files = typer.confirm("Do you wish to see the files to be added?")
    if show_files:
        typer.echo(files)

    dockerfile = template.API_DOCKERFILE
    dockerfile.replace("PORT", port)
    du.build(dockerfile, tag, docker_kwargs)
