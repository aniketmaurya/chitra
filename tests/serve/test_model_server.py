from chitra.serve import ModelServer


def dummy_model(x: str):
    return "hello" + x


def test_get_available_api_types():
    model_server = ModelServer.get_available_api_types()
    assert isinstance(model_server, list)
