import io
from abc import ABC
from typing import Callable

from chitra.serve.base import ModelServer


class CloudServer(ModelServer, ABC):
    def __init__(
        self,
        api_type: str,
        model_path: str,
        model_loader: Callable,
        preprocess_fn: Callable,
        postprocess_fn: Callable,
        **kwargs
    ):
        raw_model = self.download_model(model_path, **kwargs)
        model = model_loader(raw_model)

        super().__init__(
            api_type,
            model,
            preprocess_fn=preprocess_fn,
            postprocess_fn=postprocess_fn,
            **kwargs
        )

    def download_model(self, path: str, **kwargs) -> io.BytesIO:
        """Download Model from cloud"""
        raise NotImplementedError
