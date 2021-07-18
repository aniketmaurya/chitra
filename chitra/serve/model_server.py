import itertools
from typing import Callable, Optional

from chitra.serve.data_processing import DataProcessor, DefaultProcessor

IMAGE_CLF = "IMAGE-CLASSIFICATION"
OBJECT_DETECTION = "OBJECT-DETECTION"
TXT_CLF = "TEXT-CLASSIFICATION"
QNA = "QUESTION-ANS"

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

    @staticmethod
    def set_data_processor(preprocess_fn: Callable, postprocess_fn: Callable):
        if (preprocess_fn or postprocess_fn) is None:
            return None
        data_processor = DataProcessor(preprocess_fn, postprocess_fn)
        return data_processor

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
