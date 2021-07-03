from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub

from chitra.serve.tf_serving.client import GrpcClient, create_grpc_stub


def test_create_grpc_stub():
    assert isinstance(create_grpc_stub(), PredictionServiceStub)


def test_request():
    client = GrpcClient()
    assert isinstance(client.stub, PredictionServiceStub)
