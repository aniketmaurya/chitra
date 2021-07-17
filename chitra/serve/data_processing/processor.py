from typing import Callable, Optional

from chitra.logging import logger

from . import vision


class DataProcessor:
    def __init__(
        self,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None,
    ):
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn

    def preprocess(self, x):
        if self.preprocess_fn is None:
            logger.warning("preprocess method is not defined")
        return self.preprocess_fn(x)

    def postprocess(self, x):
        if self.postprocess_fn is None:
            logger.warning("postprocess method not defined")
        return self.postprocess_fn(x)


class DefaultProcessor:
    vision = DataProcessor(vision.default_preprocess, vision.default_postprocess)
    nlp = DataProcessor(lambda x: x, lambda x: x)
