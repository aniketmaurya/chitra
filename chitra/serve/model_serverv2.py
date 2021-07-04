import itertools
from functools import partial
from typing import Callable, Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile

from .data_processing import DataProcessor, DefaultProcessor

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

    def set_data_processor(self, preprocess_fn: Callable, postprocess_fn: Callable):
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


class API(ModelServer):
    def __init__(
        self,
        api_type: str,
        model: Callable,
        preprocess_fn: Callable = None,
        postprocess_fn: Callable = None,
        **kwargs,
    ):
        super(API, self).__init__(
            api_type, model, preprocess_fn, postprocess_fn, **kwargs
        )
        docs_url = kwargs.get("docs_url", "/docs")
        title = kwargs.get("title", "Chitra Model Server 🔥")
        desc = kwargs.get(
            "description",
            '<a href="https://chitra.readthedocs.io/en/latest">Goto Chitra Docs</a> 🔗',
        )
        self.app = FastAPI(title=title, description=desc, docs_url=docs_url)
        self.setup_api(**kwargs)

    async def predict_image(self, file: UploadFile = File(...)):
        preprocess_fn = self.data_processor.preprocess_fn
        postprocess_fn = self.data_processor.postprocess_fn

        x = preprocess_fn(await file.read())
        x = self.model(x)
        x = postprocess_fn(x)
        return x

    def setup_api(self, **kwargs):
        if self.data_processor is None:
            data_processor = self.set_default_processor()
            data_processor.preprocess_fn = partial(
                data_processor.preprocess_fn, **kwargs
            )
            data_processor.postprocess_fn = partial(
                data_processor.postprocess_fn, **kwargs
            )
            self.data_processor = data_processor

        if self.api_type == IMAGE_CLF:
            kwargs.get("image")
            self.app.post("/api/predict-image")(self.predict_image)

    def run(self):
        uvicorn.run(self.app)
