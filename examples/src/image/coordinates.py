import numpy as np
import rich

from chitra.coordinates import BoundingBoxes

box = [1, 2, 3, 4]
label = ["Dog"]
bounding_box = BoundingBoxes(box, label)
rich.print(bounding_box)
bboxes = bounding_box.resize_with_image((10, 10, 3), np.random.randn(100, 100, 3))

rich.print(bboxes)
