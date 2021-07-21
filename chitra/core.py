import os
import pathlib
from typing import Optional, Tuple

import requests
import tensorflow as tf

from chitra.constants import IMAGENET_LABEL_URL
from chitra.logging import logger

IMAGENET_LABELS: Optional[Tuple[str]] = None


def remove_dsstore(path) -> None:
    """Deletes .DS_Store files from path and sub-folders of path."""
    path = pathlib.Path(path)

    for e in path.glob("*.DS_Store"):
        os.remove(e)

    for e in path.glob("*/*.DS_Store"):
        os.remove(e)


def get_basename(path: tf.string) -> tf.string:
    if not isinstance(path, tf.Tensor):
        raise AssertionError
    return tf.strings.split(path, os.path.sep)[-1]


def load_imagenet_labels() -> Tuple[str]:
    global IMAGENET_LABELS
    if IMAGENET_LABELS is None:
        logger.debug("Downloading imagenet labels...")
        IMAGENET_LABELS = (
            requests.get(IMAGENET_LABEL_URL).content.decode("UTF-8").split("\n")[1:]
        )
    return IMAGENET_LABELS
