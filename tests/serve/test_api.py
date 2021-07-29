import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI

from chitra.serve import API
from chitra.serve import constants as const
from chitra.serve import create_api


def dummy_model(x):
    return np.asarray([1, 2])


def test_create_app():
    api_types = API.get_available_api_types()

    for api_type in api_types:
        api = create_api(dummy_model, api_type)
        assert isinstance(api.app, FastAPI)


def test_api():
    api_types = API.get_available_api_types()
    for api_type in api_types:
        api = API(api_type, dummy_model)
        assert isinstance(api.app, FastAPI)


@pytest.mark.asyncio
async def test_predict_image():
    def async_return(result):
        f = asyncio.Future()
        f.set_result(result)
        return f

    app = API(api_type=const.IMAGE_CLF, model=dummy_model)
    app.data_processor = MagicMock()
    app.data_processor.preprocess_fn = MagicMock()
    app.data_processor.postprocess_fn = lambda x: x
    file = MagicMock()
    file.read = MagicMock(return_value=async_return("Sample string"))
    output = await app.predict_image(file)
    assert isinstance(output, np.ndarray)
    assert len(output) == 2
