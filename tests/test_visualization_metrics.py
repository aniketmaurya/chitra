import numpy as np
import pytest

from chitra.visualization.metrics import cm_accuracy
from chitra.visualization.metrics import detect_multilabel
from chitra.visualization.metrics import plot_confusion_matrix


def test_detect_multilabel():
    with pytest.raises(UserWarning):
        detect_multilabel({"label1": "this will raise UserWarning"})

    assert detect_multilabel([1, 2, 3, 4])
    assert not detect_multilabel([0, 1, 1, 0])


def test_cm_accuracy():
    x = np.asarray([[1, 2], [1, 2]])
    assert cm_accuracy(x) == 0.5


def test_plot_confusion_matrix():
    y_pred = [1, 1, 0, 1]
    y_true = [0, 1, 0, 1]
    display_labels = ('watermark', 'non watermark')
    assert plot_confusion_matrix(y_pred, y_true,
                                 display_labels=display_labels) is None
