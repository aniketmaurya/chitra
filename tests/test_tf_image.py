import os

import tensorflow as tf

from chitra.image import Chitra
from chitra.tf_image import read_image, resize_image

chitra_banner = (
    "https://raw.githubusercontent.com/aniketmaurya/"
    "chitra/master/docs/assets/chitra_banner.png"
)
image = Chitra(chitra_banner).image


def test_read_image():
    image_file = "./test_read_image.png"
    image.save(image_file)
    tf_image = read_image(image_file)
    assert 3 <= tf_image.shape[-1] <= 4
    assert isinstance(tf_image, tf.Tensor)
    os.remove(image_file)


def test_resize_image():
    assert resize_image(tf.random.normal((32, 32, 3)), (24, 24)).shape == (24, 24, 3)
