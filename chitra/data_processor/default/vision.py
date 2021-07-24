from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from chitra.image import Chitra

from ..processor import DataProcessor


def default_preprocess(
    data,
    image_shape: Optional[Tuple[int, int]] = None,
    rescale: bool = True,
    expand_dims: bool = True,
) -> np.ndarray:
    """
    Supports image resize, rescaling and dimension expansion along 0th index
    Args:
        data: Image File buffer or numpy array
        image_shape: Target image size
        rescale: If true then image will be rescaled as [-1, 1]
        expand_dims:

    Returns:
        preprocessed numpy array image
    """
    if isinstance(data, str):
        image = Image.open(BytesIO(data)).convert("RGB")
    elif isinstance(data, np.ndarray):
        image = Chitra(data).image
    else:
        raise UserWarning(f"preprocessing not implemented for this data type -> {data}")

    if image_shape:
        image = image.resize(image_shape)

    image = np.asarray(image).astype(np.float32)
    if rescale:
        image = image / 127.5 - 1.0
    if expand_dims:
        image = np.expand_dims(image, 0)
    return image


def default_postprocess(data, return_type: Optional[str] = "list") -> List:
    if not isinstance(data, (np.ndarray, list, tuple, int, float)):
        data = data.numpy()
    if return_type == "list":
        if isinstance(data, np.ndarray):
            data = data.tolist()
        else:
            list(data)
    return data


class DefaultVisionProcessor:
    vision = DataProcessor(default_preprocess, default_postprocess)
