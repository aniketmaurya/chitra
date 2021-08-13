from typing import Callable

import requests
from chalice import Chalice, Rate

from chitra.serve.model_server import ModelServer

S3 = "s3"
GCS = "gcs"

RATE_UNIT = {"m": Rate.MINUTES, "h": Rate.HOURS, "d": Rate.DAYS}


def infer_location_type(path: str):
    if path.startswith("s3"):
        return S3
    elif path.startswith("gcs"):
        return GCS
    else:
        raise ValueError(f"Location type is not supported yet for path={path}")


def download_model(path: str):
    return requests.get(path, stream=True).raw


class ChaliceServer(ModelServer):
    INVOKE_METHODS = ("route", "schedule", "on_s3_event")

    def __init__(
        self,
        api_type: str,
        model_path: str,
        model_loader: Callable,
        preprocess_fn: Callable = None,
        postprocess_fn: Callable = None,
        **kwargs,
    ):
        infer_location_type(model_path)
        model: Callable = model_loader(download_model(model_path))
        super().__init__(api_type, model, preprocess_fn, postprocess_fn, **kwargs)

        self.app = Chalice(app_name=kwargs.get("name", "chitra-server"))

    def predict(self, x) -> dict:
        data_processor = self.data_processor

        if data_processor.preprocess_fn:
            x = data_processor.preprocess(x, **self.preprocess_conf)
        x = self.model(x)
        if data_processor.postprocess_fn:
            x = data_processor.postprocess(x, **self.postprocess_conf)
        return x

    def run(self, invoke_method: str, **kwargs):
        invoke_method = invoke_method.lower()
        if invoke_method not in self.INVOKE_METHODS:
            raise NotImplementedError(
                f"invoke method={invoke_method} not implemented yet. Please select {self.INVOKE_METHODS}"
            )

        if invoke_method == "route":
            route_path = kwargs.get("path", "/predict")
            self.app.route(route_path, methods=["GET"])(self.predict)

        else:
            raise NotImplementedError()
