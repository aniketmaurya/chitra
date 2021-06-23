# Cell
"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import warnings


def label_color(label):
    """ Return a color from a set of predefined colors. Contains 80 colors in total.

    Args
        label: The label to get the color for.

    Returns
        A list of three values representing a RGB color.

        If no color is defined for a certain label, the color green is returned and a warning is printed.
    """
    if label < len(colors):
        return colors[label]
    else:
        warnings.warn(
            'Label {} has no color, returning default.'.format(label))
        return (0, 255, 0)


"""
Generated using:

```
colors = [list((matplotlib.colors.hsv_to_rgb([x, 1.0, 1.0]) * 255).astype(int)) for x in np.arange(0, 1, 1.0 / 80)]
shuffle(colors)
pprint(colors)
```
"""
colors = [[31, 0, 255], [0, 159, 255], [255, 95, 0], [255, 19, 0], [255, 0, 0],
          [255, 38, 0], [0, 255, 25], [255, 0, 133], [255, 172, 0],
          [108, 0, 255], [0, 82, 255], [0, 255, 6], [255, 0,
                                                     152], [223, 0, 255],
          [12, 0, 255], [0, 255, 178], [108, 255, 0], [184, 0, 255],
          [255, 0, 76], [146, 255, 0], [51, 0, 255], [0, 197, 255],
          [255, 248, 0], [255, 0, 19], [255, 0, 38], [89, 255, 0],
          [127, 255, 0], [255, 153, 0], [0, 255, 255], [0, 255, 216],
          [0, 255, 121], [255, 0, 248], [70, 0, 255], [0, 255, 159],
          [0, 216, 255], [0, 6, 255], [0, 63, 255], [31, 255, 0], [255, 57, 0],
          [255, 0, 210], [0, 255, 102], [242, 255, 0], [255, 191, 0],
          [0, 255, 63], [255, 0, 95], [146, 0, 255], [184, 255, 0],
          [255, 114, 0], [0, 255, 235], [255, 229, 0], [0, 178, 255],
          [255, 0, 114], [255, 0, 57], [0, 140, 255], [0, 121, 255],
          [12, 255, 0], [255, 210, 0], [0, 255, 44], [165, 255,
                                                      0], [0, 25, 255],
          [0, 255, 140], [0, 101, 255], [0, 255, 82], [223, 255, 0],
          [242, 0, 255], [89, 0, 255], [165, 0, 255], [70, 255, 0],
          [255, 0, 172], [255, 76, 0], [203, 255, 0], [204, 0, 255],
          [255, 0, 229], [255, 133, 0], [127, 0, 255], [0, 235, 255],
          [0, 255, 197], [255, 0, 191], [0, 44, 255], [50, 255, 0]]

# Cell
import cv2
import numpy as np


def draw_box(image, box, color, thickness=2):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(np.int32)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness,
                  cv2.LINE_AA)


def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.

    # Arguments
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1,
                (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1,
                (255, 255, 255), 1)


def draw_boxes(image, boxes, color, thickness=2):
    """ Draws boxes on an image with a given color.

    # Arguments
        image     : The image to draw on.
        boxes     : A [N, 4] matrix (x1, y1, x2, y2).
        color     : The color of the boxes.
        thickness : The thickness of the lines to draw boxes with.
    """
    for b in boxes:
        draw_box(image, b, color, thickness=thickness)


def draw_detections(image,
                    boxes,
                    scores,
                    labels,
                    color=None,
                    label_to_name=None,
                    score_threshold=0.5):
    """ Draws detections in an image.

    # Arguments
        image           : The image to draw on.
        boxes           : A [N, 4] matrix (x1, y1, x2, y2).
        scores          : A list of N classification scores.
        labels          : A list of N labels.
        color           : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name   : (optional) Functor for mapping a label to a name.
        score_threshold : Threshold used for determining what detections to draw.
    """
    selection = np.where(scores > score_threshold)[0]

    for i in selection:
        c = color if color is not None else label_color(labels[i])
        draw_box(image, boxes[i, :], color=c)

        # draw labels
        caption = (label_to_name(labels[i]) if label_to_name else
                   labels[i]) + ': {0:.2f}'.format(scores[i])
        draw_caption(image, boxes[i, :], caption)


def draw_annotations(image,
                     annotations,
                     color=(0, 255, 0),
                     label_to_name=None):
    """ Draws annotations in an image.

    # Arguments
        image         : The image to draw on.
        annotations   : A [N, 5] matrix (x1, y1, x2, y2, label) or dictionary containing bboxes (shaped [N, 4]) and labels (shaped [N]).
        color         : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name : (optional) Functor for mapping a label to a name.
    """
    if isinstance(annotations, np.ndarray):
        annotations = {
            'bboxes': annotations[:, :4],
            'labels': annotations[:, 4]
        }

    assert ('bboxes' in annotations)
    assert ('labels' in annotations)
    assert (annotations['bboxes'].shape[0] == annotations['labels'].shape[0])

    for i in range(annotations['bboxes'].shape[0]):
        label = annotations['labels'][i]
        c = color if color is not None else label_color(label)
        caption = '{}'.format(label_to_name(label) if label_to_name else label)
        draw_caption(image, annotations['bboxes'][i], caption)
        draw_box(image, annotations['bboxes'][i], color=c)
