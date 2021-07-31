import os
from io import BytesIO
from typing import Optional

import docker

from chitra.constants import DOCKER_BASE_URL
from chitra.logging import logger

client = docker.APIClient(base_url=os.environ.get("docker_base_url", DOCKER_BASE_URL))

example = """
# Shared Volume
FROM busybox:buildroot-2014.02
VOLUME /data
CMD ["/bin/sh"]
"""


def is_connected() -> bool:
    """
    Check if Docker engine is running
    Returns:
        true if Docker engine is running
    """
    return client.ping()


def build(
    dockerfile: str, tag: Optional[str] = None, docker_kwargs: Optional[dict] = None
):
    if not docker_kwargs:
        docker_kwargs = {}

    fileobj = BytesIO(dockerfile.encode("utf-8"))
    response = []
    for line in client.build(fileobj=fileobj, rm=True, tag=tag, **docker_kwargs):
        logger.info(line)
        response.append(line)
    return response
