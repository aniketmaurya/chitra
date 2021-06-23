__all__ = ['DATA_FORMATS', 'DEFAULT_MODE', 'BoundingBoxes', 'Chitra']

import os
from io import BytesIO
from os.path import basename

import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image

from .constants import _TF, _TORCH
from .utility.import_utils import INSTALLED_MODULES

tf = None
torch = None

if INSTALLED_MODULES.get(_TF, None):
    import tensorflow as tf

if INSTALLED_MODULES.get(_TORCH, None):
    import torch

# Cell
from typing import Any, List, Optional, Union

DATA_FORMATS = Union[str, Image.Image, np.ndarray, 'tf.Tensor', 'torch.Tensor']
DEFAULT_MODE = os.environ.get("CHITRA_DEFAULT_MODE", "TF")

from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


def _cache_image(image: Image.Image, image_path: str):
    cache_dir = f'{os.path.curdir}/chitra_cache/image/'
    os.makedirs(cache_dir, exist_ok=True)
    image.save(cache_dir + basename(image_path))


def _url_to_image(url: str, cache: bool) -> Image.Image:
    """returns Image from url"""
    cache_file = f'{os.path.curdir}/chitra_cache/image/' + basename(url)
    if cache and os.path.exists(cache_file):
        return Image.open(cache_file)

    assert url.lower().startswith("http"), "invalid url, must start with http"
    content = requests.get(url).content
    image = Image.open(BytesIO(content))
    if cache:
        _cache_image(image, url)
    return image


# Cell
class BoundingBoxes:
    CENTER = "XXYY"
    CORNER = "XYXY"

    def __init__(self,
                 bboxes: Optional[List[list]] = None,
                 labels: Optional[List[Union[int, str]]] = None,
                 format: str = 'xyxy'):
        """Args:
            bboxes: list of bounding boxes [(x1, y1, x2, y2), ...] or [(xc, yc, h, w), ...]
            labels: list of strings or integers
            format:
                - `xyxy` for corner points of bbox
                - `xyhw` for x-center, y-center, height and width format of bbox
        """
        assert format.upper() in (
            self.CENTER,
            self.CORNER), f"bbox format must be either xyxy or xyhw"
        bboxes = self._listify(bboxes, 4)
        labels = self._listify(labels)
        assert len(bboxes) == len(
            labels
        ), f"len of boxes and labels not matching: {len(bboxes), len(labels)}"

        self._format = format.upper()
        self.bboxes = self._list_to_bbox(bboxes, labels)
        self._state = {}

    def _listify(self, item, dim_trigger=None):
        if item is None:
            return item

        if not isinstance(item, (list, tuple)):
            return [item]

        if isinstance(item, (list, tuple)):
            if self.num_dim(item) == dim_trigger:
                item = [item]
        return item

    @staticmethod
    def num_dim(item):
        return len(item)

    @staticmethod
    def center_to_corner(cx, cy, h, w):
        xmin = cx - w / 2
        xmax = cx + w / 2
        ymin = cy - h / 2
        ymax = cy + h / 2

        return xmin, ymin, xmax, ymax

    @staticmethod
    def corner_to_center(xmin, ymin, xmax, ymax):
        w = xmax - xmin
        h = ymax - ymin

        cx = xmin + w / 2
        cy = ymin + h / 2

        return cx, cy, h, w

    def _list_to_bbox(
            self,
            bbox_list: Optional[List[List[Union[int, float]]]],
            labels: List[Union[str, int]] = None) -> List[BoundingBox]:
        """Converts bbox list into `imgaug BoundigBox` object
        """
        format = self._format

        if not bbox_list:
            return None

        if not labels:
            labels = [None] * self.num_dim(bbox_list)

        bbox_objects = []
        for bbox, label in zip(bbox_list, labels):
            if format == self.CENTER:
                bbox = self.center_to_corner(*bbox)
            bbox_objects.append(BoundingBox(*bbox, label))
        return bbox_objects

    def __getitem__(self, idx):
        return self.bboxes[idx]

    def __repr__(self):
        return str(self.bboxes)

    def get_bounding_boxes_on_image(self, image_shape):
        """returns `imgaug BoundingBoxesOnImage` object which can be used to boxes on the image
        """
        return BoundingBoxesOnImage(self.bboxes, image_shape)


# Cell
class Chitra:
    """Ultimate image utility class.
          1. Load image from file, web url, numpy or bytes
          2. Plot image
          3. Draw bounding boxes
    """
    def __init__(self,
                 data: Any,
                 bboxes: List = None,
                 labels: List = None,
                 format: str = BoundingBoxes.CORNER,
                 cache: bool = False,
                 *args,
                 **kwargs) -> None:
        """

        Args:
            data: numpy, url, filelike
            bboxes:
            labels:
            format:
            cache[bool]: Whether to cache downloaded image
            *args:
            **kwargs:


        """
        super().__init__()
        self.image = self._load_image(data, cache=cache)
        self.bboxes = None

        if bboxes is not None:
            self.bboxes = BoundingBoxes(bboxes, labels)

    def _load_image(self, data: DATA_FORMATS, cache: bool):
        if isinstance(data, Image.Image):
            return data

        if isinstance(data, (tf.Tensor, torch.Tensor)):
            data = data.numpy()

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
            raise UserWarning('bboxes is None')

        bbox_on_image = self.bboxes.get_bounding_boxes_on_image(self.shape)
        return bbox_on_image.draw_on_image(self.numpy(),
                                           color=color,
                                           size=marker_size)
