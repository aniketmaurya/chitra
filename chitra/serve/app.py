from typing import Callable, List, Optional, Union

import gradio as gr
import numpy as np

from chitra.serve import constants as const
from chitra.serve.model_server import ModelServer


class GradioApp(ModelServer):
    API_TYPES = {"VISION": (const.IMAGE_CLF, const.OBJECT_DETECTION)}

    def __init__(
        self,
        api_type: str,
        model: Callable,
        input_types: Optional[Union[List, str]] = None,
        output_types: Optional[Union[List, str]] = None,
        preprocess_fn: Callable = None,
        postprocess_fn: Callable = None,
        preprocess_conf: Optional[dict] = None,
        postprocess_conf: Optional[dict] = None,
        **kwargs,
    ):
        super(GradioApp, self).__init__(
            api_type, model, preprocess_fn, postprocess_fn, **kwargs
        )
        if not preprocess_conf:
            preprocess_conf = {}
        if not postprocess_conf:
            postprocess_conf = {}
        self.input_types = input_types
        self.output_types = output_types
        self.api_type_func = {}
        self.preprocess_conf = preprocess_conf
        self.postprocess_conf = postprocess_conf
        self.setup(**kwargs)

    def setup(
        self,
        **kwargs,
    ):
        assert self.data_processor._preprocess_fn is not None, "Preprocess func is None"
        assert (
            self.data_processor._postprocess_fn is not None
        ), "Postprocess func is None"

        self.api_type_func[const.IMAGE_CLF] = self.image_classification

        if not self.input_types:
            self.input_types = self.get_input_type(**kwargs)

        if not self.output_types:
            self.output_types = "json"

    def get_input_type(self, **kwargs):
        label = kwargs.get("label")
        if self.api_type in (const.IMAGE_CLF, const.OBJECT_DETECTION):
            return gr.inputs.Image(shape=kwargs.get("image_shape"), label=label)

        if self.api_type == const.TXT_CLF:
            return gr.inputs.Textbox(
                lines=2, placeholder=kwargs.get("text_placeholder"), label=label
            )
        raise NotImplementedError(
            f"{self.api_type} API Type is not implemented yet!")

    def image_classification(self, x: np.ndarray):
        preprocess_fn = self.data_processor._preprocess_fn
        postprocess_fn = self.data_processor._postprocess_fn

        if preprocess_fn:
            x = preprocess_fn(x, **self.preprocess_conf)
        x = self.model(x)
        if postprocess_fn:
            x = postprocess_fn(x, **self.postprocess_conf)
        return x

    def run(self):
        gr.Interface(
            fn=self.api_type_func[self.api_type],
            inputs=self.input_types,
            outputs=self.output_types,
        ).launch()
