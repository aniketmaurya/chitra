from typing import Any, Callable, Optional

import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc


def create_grpc_stub(
    host: str = "localhost", port: str = "8500"
) -> prediction_service_pb2_grpc.PredictionServiceStub:
    hostport = f"{host}:{port}"
    channel = grpc.insecure_channel(hostport)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    return stub


def grpc_request(
    stub: prediction_service_pb2_grpc.PredictionServiceStub,
    data_sample: Any,
    input_name: str,
    model_name: str,
    signature_name: str,
    callback: Optional = None,
    grpc_timeout: int = 20,
    async_: bool = False,
):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = signature_name

    request.inputs[input_name].CopyFrom(
        tf.make_tensor_proto(data_sample, shape=data_sample.shape)
    )

    if async_:
        result_future = stub.Predict.future(request, 5)  # 5 seconds
    else:
        result_future = stub.Predict(request, grpc_timeout)

    if callback is not None:
        return callback(result_future)

    return result_future


class GrpcClient:
    def __init__(self, host: str = "localhost", port: str = "8500"):
        super().__init__()
        self.stub = create_grpc_stub(host, port)

    def request(
        self,
        data_sample,
        input_name: str,
        model_name: str,
        signature_name: str,
        callback: Callable = None,
        grpc_timeout: int = 20,
        async_: bool = False,
    ):
        stub = self.stub
        response = grpc_request(
            stub,
            data_sample,
            input_name,
            model_name,
            signature_name,
            callback,
            grpc_timeout,
            async_,
        )
        return response
