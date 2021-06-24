from imgaug.augmentables import bbs

from chitra.coordinates import BoundingBoxes


def test_bounding_boxes():
    box = [1, 2, 3, 4]
    label = ['Dog']
    bounding_box = BoundingBoxes(box, label)
    assert len(bounding_box.bboxes) == 1
    assert isinstance(bounding_box.bboxes, list)
    assert isinstance(bounding_box.bboxes[0], bbs.BoundingBox)


def test_corner_to_center():
    xmin, ymin, xmax, ymax = 0, 0, 10, 10
    cx, cy, h, w = BoundingBoxes.corner_to_center(xmin, ymin, xmax, ymax)
    assert (cx, cy, h, w) == (5, 5, 10, 10)
