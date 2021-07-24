from typing import Callable, Optional


class DataProcessor:
    def __init__(
        self,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None,
    ):
        self._preprocess_fn = preprocess_fn
        self._postprocess_fn = postprocess_fn

    @property
    def preprocess_fn(self):
        return self._preprocess_fn

    @property
    def postprocess_fn(self):
        return self._postprocess_fn

    def set_preprocess_fn(self, func):
        self._preprocess_fn = func

    def set_postprocess_fn(self, func):
        self._postprocess_fn = func

    def preprocess(self, x, **kwargs):
        if self._preprocess_fn is None:
            raise UserWarning("preprocess method is not defined")
        return self._preprocess_fn(x, **kwargs)

    def postprocess(self, x, **kwargs):
        if self._postprocess_fn is None:
            raise UserWarning("postprocess method not defined")
        return self._postprocess_fn(x, **kwargs)
