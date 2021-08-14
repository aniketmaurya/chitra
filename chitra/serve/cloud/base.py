import abc
import io
from abc import ABC
from typing import Callable, Optional

import smart_open

from chitra.serve.base import ModelServer


class CloudServer(ModelServer, ABC):
    def __init__(
        self,
        api_type: str,
        model_path: str,
        model_loader: Callable,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None,
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

    @staticmethod
    def download_model(path: str, **kwargs) -> io.BytesIO:
        """
        Download model from cloud
        ref: http://5.9.10.113/67706477/load-pytorch-model-from-s3-bucket
        Args:
            path:
            **kwargs:

        Returns:

        """

        with smart_open.open(path, mode="rb", **kwargs) as fr:
            data = io.BytesIO(fr.read())
        return data
