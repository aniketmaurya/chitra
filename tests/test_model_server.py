from fastapi import FastAPI

from chitra.serve import create_api


def test_create_app():
    model = lambda x: x + 1
    app = create_api(model)
    assert isinstance(app, FastAPI)
