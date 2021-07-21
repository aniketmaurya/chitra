from typing import Union

import tensorflow as tf


def read_image(path: str, channels: int = 3):
    """Reads an image file from the path and return the rgb image in tf.Tensor
    format."""
    img: tf.Tensor = tf.io.read_file(path)
    img: tf.Tensor = tf.io.decode_image(img, channels=channels, expand_animations=False)
    return img


def resize_image(image: tf.Tensor, size: Union[tf.Tensor, tuple], **kwargs):
    """Resize image to the target `size`: Union[tf.Tensor, tuple]"""
    if not isinstance(image, tf.Tensor):
        raise AssertionError(
            f"image must be of type tf.Tensor but passed {type(image)}"
        )
    if not isinstance(size, (tuple, tf.Tensor)):
        raise AssertionError
    method = kwargs.get("method", "bilinear")
    return tf.image.resize(image, size, method)
