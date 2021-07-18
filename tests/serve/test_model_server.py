import numpy as np
from fastapi import FastAPI

from chitra.serve import create_api, get_available_api_types


def dummy_model(x):
    return np.asarray([1, 2])


def test_create_app():
    api_types = get_available_api_types()

    for api_type in api_types:
        api = create_api(dummy_model, api_type)
        assert isinstance(api.app, FastAPI)
