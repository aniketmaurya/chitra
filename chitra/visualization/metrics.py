"""https://www.kaggle.com/grfiv4/plot-a-confusion-matrix."""
from itertools import product
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from chitra.logging import logger


def detect_multilabel(labels: Union[List, np.ndarray]):
    if isinstance(labels, (np.ndarray, list, tuple)):
        labels = np.unique(labels)
    else:
        raise UserWarning(
            f"expect data type for label was list or np.ndarray but got {type(labels)}"
        )

    if len(labels) > 2:
        return True
    return False


def cm_accuracy(cm: np.ndarray):
    return np.trace(cm) / float(np.sum(cm))


def plot_confusion_matrix(
    y_pred: Union[np.ndarray, List],
    y_true: Union[np.ndarray, List],
    display_labels=None,
    include_values: bool = True,
    title: str = "Confusion Matrix",
    cmap: str = None,
):
    if detect_multilabel(y_true):
        logger.warning("You might want to use multi-label version!")

    if display_labels is None:
        display_labels = np.unique(y_true)

    n_classes = len(display_labels)
    tick_marks = np.arange(n_classes)

    if cmap is None:
        cmap = plt.get_cmap("Blues")

    cm = confusion_matrix(y_true, y_pred)
    accuracy = cm_accuracy(cm)
    error = 1 - accuracy

    plt.imshow(cm, cmap=cmap)

    if include_values:
        for i, j in product(range(n_classes), range(n_classes)):
            plt.text(i, j, "{:,}".format(cm[i, j]))

    plt.xticks(tick_marks, display_labels, rotation=45)
    plt.yticks(tick_marks, display_labels)
    plt.title(title)
    plt.xlabel(f"Predicted Label\nAccuracy={accuracy:0.4f}; Error={error:0.4f}")
    plt.ylabel("True Label")

    plt.show()
