from typing import List, Optional, Union

from imgaug.augmentables import bbs


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
            self.CORNER), "bbox format must be either xyxy or xyhw"
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

        if isinstance(item,
                      (list, tuple)) and self.num_dim(item) == dim_trigger:
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
            labels: List[Union[str, int]] = None) -> List[bbs.BoundingBox]:
        """Converts bbox list into `imgaug BoundigBox` object
        """
        _format = self._format

        if not bbox_list:
            return []

        if not labels:
            labels = [None] * self.num_dim(bbox_list)

        bbox_objects = []
        for bbox, label in zip(bbox_list, labels):
            if _format == self.CENTER:
                bbox = self.center_to_corner(*bbox)
            bbox_objects.append(bbs.BoundingBox(*bbox, label))
        return bbox_objects

    def __getitem__(self, idx):
        return self.bboxes[idx]

    def __repr__(self):
        return str(self.bboxes)

    def get_bounding_boxes_on_image(self, image_shape):
        """returns `imgaug BoundingBoxesOnImage` object which can be used to boxes on the image
        """
        return bbs.BoundingBoxesOnImage(self.bboxes, image_shape)
