import os

import tensorflow as tf

from chitra import Chitra
from chitra.tf_image import read_image
from chitra.tf_image import resize_image

chitra_banner = 'https://raw.githubusercontent.com/aniketmaurya/chitra/master/docs/assets/chitra_banner.png'
image = Chitra(chitra_banner).image


def test_read_image():
    image.save('./test_read_image.png')
    tf_image = read_image('./temp.png')
    assert 3 <= tf_image.shape[-1] <= 4
    assert isinstance(tf_image, tf.Tensor)
    os.remove('./test_read_image.png')


def test_resize_image():
    assert resize_image(tf.random.normal((32, 32, 3)),
                        (24, 24)).shape == (24, 24, 3)
