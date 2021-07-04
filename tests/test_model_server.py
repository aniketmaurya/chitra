import numpy as np
from fastapi import FastAPI

from chitra.serve import create_api


def dummy_model(x):
    return np.asarray([1, 2])


def test_create_app():
    api = create_api(dummy_model, "IMAGE-CLASSIFICATION")
    assert isinstance(api.app, FastAPI)
