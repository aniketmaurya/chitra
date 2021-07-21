import numpy as np

from chitra.data_processing.default.vision import default_preprocess


def test_default_preprocess():
    image_arr = np.random.randn(16, 16, 3).astype("uint8")

    result = default_preprocess(
        data=image_arr, image_shape=(32, 32), rescale=True, expand_dims=True
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 32, 32, 3)
