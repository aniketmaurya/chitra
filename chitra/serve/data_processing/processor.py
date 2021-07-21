from typing import Callable, Optional

from chitra.logging import logger
from chitra.serve.data_processing import vision


class DataProcessor:
    def __init__(
        self,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None,
    ):
        self._preprocess_fn = preprocess_fn
        self._postprocess_fn = postprocess_fn

    def preprocess(self, x):
        if self._preprocess_fn is None:
            raise UserWarning("preprocess method is not defined")
        return self._preprocess_fn(x)

    def postprocess(self, x):
        if self._postprocess_fn is None:
            raise UserWarning("postprocess method not defined")
        return self._postprocess_fn(x)


class DefaultProcessor:
    vision = DataProcessor(vision.default_preprocess,
                           vision.default_postprocess)
    nlp = DataProcessor(lambda x: x, lambda x: x)
