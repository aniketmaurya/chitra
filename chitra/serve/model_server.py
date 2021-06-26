from functools import partial
from io import BytesIO
from typing import Callable, List, Optional, Tuple

from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
import numpy as np
from PIL import Image
import uvicorn

from chitra.serve.schema import QnARequest
from chitra.serve.schema import QnAResponse
from chitra.serve.schema import Query
from chitra.serve.schema import QueryResult

IMAGE_CLF = "IMAGE-CLASSIFICATION"
TXT_CLF = "TEXT-CLASSIFICATION"
QNA = "QUESTION-ANS"
API_TYPES = (IMAGE_CLF, TXT_CLF, QNA)


def _default_preprocess(image_file, image_size: Tuple[int, int],
                        rescale: bool) -> np.ndarray:
    image = Image.open(BytesIO(image_file)).convert('RGB')
    image = image.resize(image_size)
    image = np.asarray(image).astype(np.float32)
    if rescale:
        image = image / 127.5 - 1.0
    image = np.expand_dims(image, 0)
    return image


def _default_postprocess(data) -> List:
    if not isinstance(data, (np.ndarray, list, tuple)):
        data = data.numpy()
    data = data.tolist()
    return data


def create_api(model: Callable,
               api_type: str = 'IMAGE-CLASSIFICATION',
               preprocess: Optional[Callable] = None,
               postprocess: Optional[Callable] = None,
               run: bool = False,
               **kwargs) -> FastAPI:
    """
    Create ASGI API with using FastAPI.
    Args:
        model[Callable]: any callable model -> pred = model(x)
        api_type[str]: Type of API to be created -> {"IMAGE-CLASSIFICATION", "TEXT-CLASSIFICATION", "QUESTION-ANS"}
        preprocess[Callable]: Preprocessing function for your data. (Default fn will be applied if None.)
        postprocess[Callable]: Postprocessing function for your data. (Default fn will be applied if None.)
        run[bool]: Whether to the the API or not.
        **kwargs:
            image_size[tuple]: if the api type is image-classification then size of target image.

    Returns:
        FastAPI app

    """
    api_type = api_type.upper()
    docs_url = kwargs.get('docs_url', '/docs')
    title = kwargs.get('title', 'Chitra Model Server 🔥')
    desc = kwargs.get(
        'description',
        '<a href="https://chitra.readthedocs.io/en/latest">Goto Chitra Docs</a> 🔗'
    )
    app = FastAPI(title=title, description=desc, docs_url=docs_url)

    @app.get('/healthz')
    def health():
        return {'OK': True}

    if api_type == IMAGE_CLF:

        @app.post('/api/predict-image')
        async def predict_image(file: UploadFile = File(...)):
            preprocess_fn = preprocess
            postprocess_fn = postprocess
            if preprocess is None:
                preprocess_fn = partial(_default_preprocess,
                                        image_size=kwargs.get(
                                            'image_size', (224, 224)),
                                        rescale=kwargs.get('rescale', True))
            if postprocess_fn is None:
                postprocess_fn = _default_postprocess

            x = preprocess_fn(await file.read())
            x = model(x)
            x = postprocess_fn(x)
            return x

        if api_type == TXT_CLF:

            @app.post('/api/predict-text', response_model=QueryResult)
            async def predict_text(data: Query):
                x = data.query
                if preprocess:
                    x = preprocess(x)
                x = model(x)
                if postprocess:
                    x = postprocess(x)
                return x

    if api_type == QNA:

        @app.post('/api/QnA', response_model=QnAResponse)
        async def predict_question_answer(data: QnARequest):
            query = data.query
            question = data.question
            if preprocess:
                x = preprocess(query, question)
            x = model(x)
            if postprocess:
                x = postprocess(x)
            return x

    if run:
        uvicorn.run(app)
    return app
