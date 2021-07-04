from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


def default_preprocess(
    image_file,
    **kwargs,
) -> np.ndarray:
    image = Image.open(BytesIO(image_file)).convert("RGB")

    if kwargs.get("image_size", None):
        image = image.resize(kwargs.get("image_size"))

    image = np.asarray(image).astype(np.float32)
    if kwargs.get("rescale", None):
        image = image / 127.5 - 1.0
    image = np.expand_dims(image, 0)
    return image


def default_postprocess(data) -> List:
    if not isinstance(data, (np.ndarray, list, tuple, int, float)):
        data = data.numpy()
    if isinstance(data, np.ndarray):
        data = data.tolist()
    return data
