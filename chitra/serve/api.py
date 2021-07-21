from functools import partial
from typing import Callable, Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile

from chitra.__about__ import documentation_url
from chitra.serve import schema
from chitra.serve.constants import IMAGE_CLF, OBJECT_DETECTION, QNA, TXT_CLF
from chitra.serve.model_server import ModelServer


class API(ModelServer):
    def __init__(
        self,
        api_type: str,
        model: Callable,
        preprocess_fn: Callable = None,
        postprocess_fn: Callable = None,
        **kwargs,
    ):
        """
        Creates FastAPI app for `api_type`
        Args:
            api_type: Type of the API. See `API.available_api_types()`
            model: Any ML/DL model
            preprocess_fn: Override Data Preprocessing Function, data will
            be processed with this function
            before calling model.
            postprocess_fn: Override Data Postprocessing Function, model
            output will be passed into this function.
            **kwargs:
        """
        super(API, self).__init__(
            api_type, model, preprocess_fn, postprocess_fn, **kwargs
        )
        docs_url = kwargs.get("docs_url", "/docs")
        title = kwargs.get("title", "Chitra Model Server 🔥")
        desc = kwargs.get(
            "description",
            f"<a href={documentation_url}>Goto Chitra Docs</a> 🔗",
        )
        self.app: FastAPI = FastAPI(title=title, description=desc, docs_url=docs_url)
        self.setup(**kwargs)

    async def predict_image(self, file: UploadFile = File(...)):
        preprocess_fn = self.data_processor._preprocess_fn
        postprocess_fn = self.data_processor._postprocess_fn

        x = preprocess_fn(await file.read())
        x = self.model(x)
        x = postprocess_fn(x)
        return x

    async def predict_text(self, data: schema.Query):
        data_processor = self.data_processor
        x = data.query
        if data_processor._preprocess_fn:
            x = data_processor.preprocess(x)
        x = self.model(x)
        if data_processor._postprocess_fn:
            x = data_processor.postprocess(x)
        return x

    async def predict_question_answer(self, data: schema.QnARequest):
        data_processor = self.data_processor
        x = data.query, data.question
        if data_processor._preprocess_fn:
            x = data_processor.preprocess(x)
        x = self.model(x)
        if data_processor._postprocess_fn:
            x = data_processor.postprocess(x)
        return x

    def setup(self, **kwargs):
        data_processor = self.data_processor
        self.data_processor._preprocess_fn = partial(
            data_processor._preprocess_fn, **kwargs
        )
        self.data_processor._postprocess_fn = partial(
            data_processor._postprocess_fn, **kwargs
        )

        if self.api_type in (IMAGE_CLF, OBJECT_DETECTION):
            self.app.post("/api/predict-image")(self.predict_image)

        elif self.api_type == TXT_CLF:
            self.app.post("/api/predict-text")(self.predict_text)

        elif self.api_type == QNA:
            self.app.post("/api/QnA")(self.predict_question_answer)

    def run(self):
        uvicorn.run(self.app)


def create_api(
    model: Callable,
    api_type: str = "IMAGE-CLASSIFICATION",
    preprocess: Optional[Callable] = None,
    postprocess: Optional[Callable] = None,
    run: bool = False,
    **kwargs,
) -> API:
    """
    Launch FastAPI app
    Args:
        model: Any ML/DL model
        api_type: Type of the API task, see `chitra.serve.get_available_api_types()`
        preprocess: Override default preprocessing function
        postprocess: Override default postprocessing function
        run: Set True to run the app
        **kwargs:

    Returns:
        Object of `chitra.serve.API` class
    """
    api = API(api_type, model, preprocess, postprocess, **kwargs)

    if run:
        api.run()
    return api
