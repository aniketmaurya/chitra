from fastapi import FastAPI
import numpy as np

from chitra.serve import create_api
from chitra.serve.model_server import _default_postprocess


def test_create_app():
    model = lambda x: x + 1
    app = create_api(model)
    assert isinstance(app, FastAPI)


def test__default_postprocess():
    data = _default_postprocess(np.random.randn(5))
    assert len(data) == 5
    assert isinstance(data, list)
