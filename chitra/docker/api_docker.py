import os
import tempfile
from typing import Optional

COPY_PL = "COPY_PL"
PORT_PL = "PORT_PL"
MAIN_PL = "MAIN_PL"

GUNICORN_DEFAULT_CONF = """
import multiprocessing

workers = multiprocessing.cpu_count()
"""

DOCKER_FILE = f"""
FROM python:3.7

LABEL maintainer="Aniket Maurya <theaniketmaurya@gmail.com>"

RUN pip install --no-cache-dir "chitra[serve]" gunicorn

{COPY_PL}

EXPOSE PORT_PL

ENTRYPOINT gunicorn -c gunicorn_conf.py MAIN_PL.api:app

"""


def str_to_file(text: str):
    temp = tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete=False)
    temp.write(text)
    return temp


def dockerize_api(
    source_path: str,
    main_module_name: str,
    gunicorn_conf_path: Optional[str] = None,
    port: str = "8080",
):
    if gunicorn_conf_path is None:
        gunicorn_conf_path = str_to_file(GUNICORN_DEFAULT_CONF).name

    if not os.path.exists(gunicorn_conf_path):
        raise FileNotFoundError(
            f"gunicorn_conf_path not found at - {gunicorn_conf_path}"
        )

    if not os.path.exists(source_path):
        raise FileNotFoundError(
            "You must provide the folder/file path of your Serving code"
        )

    copy_cmd = (
        f"COPY {source_path} ./ \n" + f"COPY {gunicorn_conf_path} ./gunicorn_conf.py"
    )

    copy_cmd = copy_cmd.strip()
    docker_cmd = DOCKER_FILE[:]
    docker_cmd = docker_cmd.replace(COPY_PL, copy_cmd)
    docker_cmd = docker_cmd.replace(PORT_PL, port)
    docker_cmd = docker_cmd.replace(MAIN_PL, main_module_name)

    return docker_cmd
