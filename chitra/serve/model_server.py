import itertools
from typing import Callable, Optional

from chitra.serve.constants import IMAGE_CLF, OBJECT_DETECTION, QNA, TXT_CLF
from chitra.serve.data_processing import DataProcessor, DefaultProcessor

API_TYPES = {"VISION": (IMAGE_CLF, OBJECT_DETECTION), "NLP": (TXT_CLF, QNA)}


def get_available_api_types():
    return list(itertools.chain.from_iterable(API_TYPES.values()))


class ModelServer:
    def __init__(
        self,
        api_type: str,
        model: Callable,
        preprocess_fn=None,
        postprocess_fn=None,
        **kwargs,
    ):
        self.api_type = api_type.upper()
        self.model = model
        self.data_processor: Optional[DataProcessor] = self.set_data_processor(
            preprocess_fn, postprocess_fn
        )

    def set_data_processor(self, preprocess_fn: Callable, postprocess_fn: Callable):
        data_preprocessor = self.set_default_processor()
        if preprocess_fn:
            data_preprocessor.preprocess_fn = preprocess_fn
        if postprocess_fn:
            data_preprocessor.postprocess_fn = postprocess_fn
        return data_preprocessor

    def set_default_processor(self) -> DataProcessor:
        api_type = self.api_type
        if api_type in API_TYPES.get("VISION"):
            self.data_processor = DefaultProcessor.vision
        elif api_type in API_TYPES.get("NLP"):
            self.data_processor = DefaultProcessor.nlp
        else:
            raise UserWarning(
                f"{api_type} is not implemented! Available types are - {get_available_api_types()}"
            )
        return self.data_processor

    @staticmethod
    def get_available_api_types():
        return get_available_api_types()
