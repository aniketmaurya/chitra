from unittest.mock import MagicMock, patch

import numpy as np
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub

from chitra.serve.tf_serving.client import GrpcClient, create_grpc_stub, grpc_request


def test_create_grpc_stub():
    assert isinstance(create_grpc_stub(), PredictionServiceStub)


def test_request():
    client = GrpcClient()
    assert isinstance(client.stub, PredictionServiceStub)


@patch("chitra.serve.tf_serving.client.predict_pb2")
def test_grpc_request(mock_predict_pb2):
    mock_predict_pb2.PredictRequest = MagicMock()
    stub = MagicMock()
    stub.Predict = MagicMock()
    stub.Predict.return_value = True

    data = np.random.randn(224, 224, 3)

    result = grpc_request(
        stub,
        data,
        input_name="input",
        model_name="test_model",
        signature_name="test",
    )

    assert result
    stub.Predict.assert_called()
    mock_predict_pb2.PredictRequest.assert_called()
    stub.Predict.assert_called_with(mock_predict_pb2.PredictRequest(), 20)
