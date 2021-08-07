import os
from io import BytesIO
from pathlib import Path
from typing import Any, List, Union

import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image

from chitra.constants import _TF, _TORCH, CHITRA_URL_SEP, IMAGE_CACHE_DIR
from chitra.coordinates import BoundingBoxes
from chitra.import_utils import is_installed

tf = None
torch = None

if is_installed(_TF):
    import tensorflow as tf

if is_installed(_TORCH):
    import torch

DATA_FORMATS = Union[str, Image.Image, np.ndarray, "tf.Tensor", "torch.Tensor"]
DEFAULT_MODE = os.environ.get("CHITRA_DEFAULT_MODE", "TF")


def _cache_image(image: Image.Image, image_path: str):
    cache_dir = Path(IMAGE_CACHE_DIR)
    filename = image_path.replace("/", CHITRA_URL_SEP)
    os.makedirs(cache_dir, exist_ok=True)
    image.save(cache_dir / filename)


def _url_to_image(url: str, cache: bool) -> Image.Image:
    """returns Image from url."""
    filename = url.replace("/", CHITRA_URL_SEP)
    cache_file = Path(IMAGE_CACHE_DIR) / filename
    if cache and os.path.exists(cache_file):
        return Image.open(cache_file)

    if not url.lower().startswith("http"):
        raise AssertionError("invalid url, must start with http")
    content = requests.get(url).content
    image = Image.open(BytesIO(content))
    if cache:
        _cache_image(image, url)
    return image


class Chitra:
    """Ultimate image utility class.

    1. Load image from file, web url, numpy or bytes
    2. Plot image
    3. Draw bounding boxes
    """

    def __init__(
        self,
        data: Any,
        bboxes: List = None,
        labels: List = None,
        box_format: str = BoundingBoxes.CORNER,
        cache: bool = False,
        *args,
        **kwargs
    ) -> None:
        """

        Args:
            data: numpy, url, filelike
            bboxes:
            labels:
            box_format:
            cache[bool]: Whether to cache downloaded image
            *args:
            **kwargs:
        """
        super().__init__()
        self.image = self._load_image(data, cache=cache)
        self.bboxes = None

        if bboxes is not None:
            self.bboxes = BoundingBoxes(bboxes, labels)

    @staticmethod
    def _load_image(data: DATA_FORMATS, cache: bool):
        if isinstance(data, Image.Image):
            return data

        if isinstance(data, (tf.Tensor, torch.Tensor)):
            data = data.numpy().astype("uint8")

        if isinstance(data, str):
            if data.startswith("http"):
                image = _url_to_image(data, cache)

            else:
                image = Image.open(data)

        elif isinstance(data, np.ndarray):
            image = Image.fromarray(data)

        else:
            raise UserWarning("unable to load image!")

        return image

    def numpy(self):
        return np.asarray(self.image)

    def to_tensor(self, mode: str = DEFAULT_MODE):
        """mode: tf/torch/pt"""
        mode = mode.upper()
        np_image = self.numpy()

        if mode == "TF":
            tensor = tf.constant(np_image)
        elif mode in ("TORCH", "PT"):
            tensor = torch.from_numpy(np_image)
        else:
            raise UserWarning("invalid mode!")
        return tensor

    @property
    def shape(self):
        return self.numpy().shape

    @property
    def size(self):
        return self.image.size

    def imshow(self, cmap=plt.cm.Blues, *args, **kwargs):
        plt.imshow(self.numpy(), cmap, *args, **kwargs)

    def draw_boxes(
        self,
        marker_size: int = 2,
        color=(0, 255, 0),
    ):
        if self.bboxes is None:
            raise UserWarning("bboxes is None")

        bbox_on_image = self.bboxes.get_bounding_boxes_on_image(self.shape)
        return bbox_on_image.draw_on_image(
            self.numpy()[..., :3], color=color, size=marker_size
        )

    def resize_image_with_bbox(self, size: List[int]):
        old_size = self.shape
        self.image = self.image.resize(size)
        self.bboxes.resize_with_image(old_size, self.numpy())
        return self.image, self.bboxes
