import pytest

from chitra.data_processing import DataProcessor


def dummy_preprocess(x: int):
    return x + 1


def dummy_postprocess(x: int):
    return x - 1


def test_data_processor():
    x = 5
    data_processor_empty = DataProcessor()
    assert not data_processor_empty._preprocess_fn
    assert not data_processor_empty._postprocess_fn

    with pytest.raises(UserWarning):
        data_processor_empty.preprocess(x)

    with pytest.raises(UserWarning):
        data_processor_empty.postprocess(x)

    data_processor = DataProcessor(
        preprocess_fn=dummy_preprocess, postprocess_fn=dummy_postprocess
    )

    x = data_processor.preprocess(x)
    x = data_processor.postprocess(x)
    assert x == 5

    data_processor_empty.set_preprocess_fn(dummy_preprocess)
    data_processor_empty.set_postprocess_fn(dummy_postprocess)

    x = data_processor_empty.preprocess(x)
    x = data_processor_empty.postprocess(x)
    assert x == 5
