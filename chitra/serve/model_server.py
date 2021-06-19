from typing import Callable, Optional

import uvicorn
from fastapi import FastAPI


def create_app(model: Callable,
               preprocess: Optional[Callable] = None,
               postprocess: Optional[Callable] = None,
               run: bool = False,
               **kwargs):
    title = kwargs.get('title', 'Chitra Model Server 🔥')
    desc = kwargs.get(
        'description',
        '<a href="https://chitra.readthedocs.io/en/latest">Goto Chitra Docs</a> 🔗'
    )
    app = FastAPI(title=title, description=desc)

    @app.get('/healthz')
    def health():
        return {'OK': True}

    # TODO: make data configurable
    @app.post('/api/predict')
    def predict_api(data):
        if preprocess:
            x = preprocess(data)
        x = model(x)
        if postprocess:
            x = postprocess(x)

        return x

    if run:
        uvicorn.run(app)
    return app
