from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from chitra.visualization.metrics import (
    cm_accuracy,
    detect_multilabel,
    plot_confusion_matrix,
)


def test_detect_multilabel():
    with pytest.raises(UserWarning):
        detect_multilabel({"label1": "this will raise UserWarning"})

    assert detect_multilabel([1, 2, 3, 4])
    assert not detect_multilabel([0, 1, 1, 0])


def test_cm_accuracy():
    x = np.asarray([[1, 2], [1, 2]])
    assert cm_accuracy(x) == 0.5


@patch("chitra.visualization.metrics.plt")
def test_plot_confusion_matrix(mock_plt: Mock):
    mock_plt.show = MagicMock()

    y_pred = [1, 1, 0, 1]
    y_true = [0, 1, 0, 1]

    assert plot_confusion_matrix(y_pred, y_true) is None
    mock_plt.show.assert_called_once()
