import numpy as np
from imgaug.augmentables import bbs

from chitra.coordinates import BoundingBoxes


def test_bounding_boxes():
    box = [1, 2, 3, 4]
    label = ["Dog"]
    bounding_box = BoundingBoxes(box, label)
    assert len(bounding_box.bboxes) == 1
    assert isinstance(bounding_box.bboxes, list)
    assert isinstance(bounding_box.bboxes[0], bbs.BoundingBox)


def test_corner_to_center():
    xmin, ymin, xmax, ymax = 0, 0, 10, 10
    cx, cy, h, w = BoundingBoxes.corner_to_center(xmin, ymin, xmax, ymax)
    assert (cx, cy, h, w) == (5, 5, 10, 10)


def test_resize_with_image():
    box = [1, 2, 3, 4]
    label = ["chitra"]
    bounding_box = BoundingBoxes(box, label)
    bbs = bounding_box.resize_with_image((10, 10, 3), np.random.randn(100, 100, 3))
    rescaled_bounding_box = bbs.bounding_boxes[0]
    assert np.isclose(rescaled_bounding_box.x1, 10)
    assert np.isclose(rescaled_bounding_box.y1, 20)
    assert np.isclose(rescaled_bounding_box.x2, 30)
    assert np.isclose(rescaled_bounding_box.y2, 40)
