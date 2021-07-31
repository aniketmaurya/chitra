import itertools
from typing import Callable, List, Optional

from chitra.data_processing import (
    DataProcessor,
    DefaultTextProcessor,
    DefaultVisionProcessor,
)
from chitra.serve import constants as const


class ModelServer:
    API_TYPES = {
        "VISION": (const.IMAGE_CLF, const.OBJECT_DETECTION),
        "NLP": (const.TXT_CLF, const.QNA),
    }

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

    @classmethod
    def get_available_api_types(cls) -> List[str]:
        return list(itertools.chain.from_iterable(cls.API_TYPES.values()))

    def set_data_processor(
        self, preprocess_fn: Callable, postprocess_fn: Callable
    ) -> DataProcessor:
        data_preprocessor = self.set_default_processor()
        if preprocess_fn:
            data_preprocessor.set_preprocess_fn(preprocess_fn)
        if postprocess_fn:
            data_preprocessor.set_postprocess_fn(postprocess_fn)
        return data_preprocessor

    def set_default_processor(self) -> DataProcessor:
        api_type = self.api_type
        if api_type in ModelServer.API_TYPES.get("VISION"):
            self.data_processor = DefaultVisionProcessor.vision
        elif api_type in ModelServer.API_TYPES.get("NLP"):
            self.data_processor = DefaultTextProcessor.nlp
        else:
            raise UserWarning(
                f"{api_type} is not implemented! Available types are -\
                 {ModelServer.get_available_api_types()}"
            )
        return self.data_processor

    def run(self, *_, **__):
        raise NotImplementedError
