from fastapi import FastAPI

from chitra.serve import create_app


def test_create_app():
    model = lambda x: x + 1
    app = create_app(model)
    assert isinstance(app, FastAPI)
