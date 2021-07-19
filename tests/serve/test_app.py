import numpy as np

from chitra.serve import GradioApp
from chitra.serve import constants as const


def dummy_model(x):
    return None


def test_gradio_app():
    api_types = GradioApp.get_available_api_types()
    for api_type in api_types:
        app = GradioApp(api_type, model=dummy_model)
        assert app

    for api_type in api_types:
        app = GradioApp(
            api_type,
            model=dummy_model,
            preprocess_fn=lambda dummy: dummy,
            postprocess_fn=lambda dummy: dummy,
        )
        assert app


def test_image_classification():
    def dummy_clf(x):
        return np.random.randn(1)

    # def preprocess_fn(x, **kwargs):
    #     print(kwargs)
    #     return x

    def postprocess_fn(x, thresh: float):
        return (x > thresh)[0]

    preprocess_conf = {"rescale": True, "expand_dims": True}
    postprocess_conf = {"thresh": 0.5}

    dummy_image = np.random.randn(224, 224, 3)

    app = GradioApp(
        const.IMAGE_CLF,
        model=dummy_clf,
        image_shape=dummy_image.shape[:2],
        preprocess_fn=None,
        preprocess_conf=preprocess_conf,
        postprocess_fn=postprocess_fn,
        postprocess_conf=postprocess_conf,
    )

    assert app.image_classification(dummy_image) in (0, 1)
