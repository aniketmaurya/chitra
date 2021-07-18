from functools import partial
from typing import Callable

import gradio as gr
import numpy as np

from chitra.serve import constants as const
from chitra.serve.model_server import ModelServer


class GradioApp(ModelServer):
    def __init__(
        self,
        api_type: str,
        model: Callable,
        preprocess_fn: Callable = None,
        postprocess_fn: Callable = None,
        **kwargs,
    ):
        super(GradioApp, self).__init__(
            api_type, model, preprocess_fn, postprocess_fn, **kwargs
        )
        self.api_type_func = {}
        self.setup(**kwargs)

    def setup(self, **kwargs):
        self.api_type_func[const.IMAGE_CLF] = self.image_classification
        self.input_type = self.get_input_type(**kwargs)
        if self.data_processor is None:
            self.data_processor = self.set_default_processor()
            self.data_processor.preprocess_fn = partial(
                self.data_processor.preprocess_fn,
                image_shape=kwargs.get("image_shape"),
                rescale=kwargs.get("rescale"),
            )

    def get_input_type(self, **kwargs):
        label = kwargs.get("label", None)
        if self.api_type in (const.IMAGE_CLF, const.OBJECT_DETECTION):
            return gr.inputs.Image(shape=kwargs.get("image_shape", None), label=label)

        elif self.api_type == const.TXT_CLF:
            return gr.inputs.Textbox(
                lines=2, placeholder=kwargs.get("text_placeholder", None), label=label
            )

    def image_classification(self, x: np.ndarray):
        preprocess_fn = self.data_processor.preprocess_fn
        postprocess_fn = self.data_processor.postprocess_fn

        if preprocess_fn:
            x = preprocess_fn(x)
        x = self.model(x)
        if postprocess_fn:
            x = postprocess_fn(x)
        return x

    def run(self):
        gr.Interface(
            fn=self.api_type_func[self.api_type],
            inputs=self.input_type,
            outputs="label",
        ).launch()


if __name__ == "__main__":
    app = GradioApp("image-classification", lambda x: 1)
    app.run()
