from typing import Callable, List, Optional

from chalice import Chalice, Rate

from chitra.logging import logger
from chitra.serve.cloud.base import CloudServer

S3 = "s3"
GCS = "gcs"

RATE_UNIT = {"m": Rate.MINUTES, "h": Rate.HOURS, "d": Rate.DAYS}


class ChaliceServer(CloudServer):
    INVOKE_METHODS = ("route",)

    def __init__(
        self,
        api_type: str,
        model_path: str,
        model_loader: Callable,
        preprocess_fn: Callable = None,
        postprocess_fn: Callable = None,
        **kwargs,
    ):
        super().__init__(
            api_type,
            model_path=model_path,
            model_loader=model_loader,
            preprocess_fn=preprocess_fn,
            postprocess_fn=postprocess_fn,
            **kwargs,
        )

        self.app = Chalice(app_name=kwargs.get("name", "chitra-server"))

    @staticmethod
    def index():
        return {"hello": "world"}

    def predict(self) -> dict:

        data_processor = self.data_processor
        x = self.app.current_request.raw_body
        logger.debug(f"raw body type={type(x)}")
        if data_processor.preprocess_fn:
            x = data_processor.preprocess(x, **self.preprocess_conf)
        x = self.model(x)
        if data_processor.postprocess_fn:
            x = data_processor.postprocess(x, **self.postprocess_conf)
        return x

    def run(self, invoke_method: str, content_types: Optional[List] = None, **kwargs):
        invoke_method = invoke_method.lower()
        if not content_types:
            content_types = []

        if invoke_method not in self.INVOKE_METHODS:
            raise NotImplementedError(
                f"invoke method={invoke_method} not implemented yet. Please select {self.INVOKE_METHODS}"
            )

        if invoke_method == "route":
            route_path = kwargs.get("path", "/predict")
            self.app.route("/", methods=["GET"])(self.index)
            self.app.route(route_path, methods=["POST"], content_types=content_types)(self.predict)
