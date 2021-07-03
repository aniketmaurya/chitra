import numpy as np
from PIL import Image

from chitra.image import Chitra


def test__load_image():
    url = "https://raw.githubusercontent.com/aniketmaurya/chitra/master/docs/assets/logo.png"
    image = Chitra(url, cache=True)
    assert isinstance(image.image, Image.Image)


def test_numpy():
    assert True


def test_to_tensor():
    assert True


def test_shape():
    assert True


def test_size():
    assert True


def test_imshow():
    assert True


def test_draw_boxes():
    assert True


def test_resize_image_with_bbox():
    box = [10, 20, 30, 40]
    label = ["chitra"]
    dummy = np.random.randn(100, 100, 3).astype("uint8")
    image = Chitra(dummy, bboxes=box, labels=label)
    image.resize_image_with_bbox((10, 10))
    rescaled_bounding_box = image.bboxes[0]

    assert np.isclose(rescaled_bounding_box.x1, 1)
    assert np.isclose(rescaled_bounding_box.y1, 2)
    assert np.isclose(rescaled_bounding_box.x2, 3)
    assert np.isclose(rescaled_bounding_box.y2, 4)
