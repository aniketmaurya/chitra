import pytest

from chitra.serve import ModelServer
from chitra.serve import constants as const


def dummy_model(x: str):
    return "hello" + x


def test_get_available_api_types():
    model_server = ModelServer.get_available_api_types()
    assert isinstance(model_server, list)


def test_model_server_run():
    model_server = ModelServer(api_type=const.TXT_CLF, model=dummy_model)
    with pytest.raises(NotImplementedError):
        model_server.run()
