from functools import partial
from typing import Callable, List, Optional, Union

import gradio as gr
import numpy as np

from chitra.serve import constants as const
from chitra.serve.model_server import ModelServer


class GradioApp(ModelServer):
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
        self.setup(preprocess_conf, postprocess_conf, **kwargs)

    def setup(
        self,
        preprocess_conf,
        postprocess_conf,
        **kwargs,
    ):
        assert self.data_processor.preprocess_fn is not None, "Preprocess func is None"
        assert (
            self.data_processor.postprocess_fn is not None
        ), "Postprocess func is None"

        self.api_type_func[const.IMAGE_CLF] = self.image_classification

        if not self.input_types:
            self.input_types = self.get_input_type(**kwargs)

        if not self.output_types:
            self.output_types = "json"

        self.data_processor.preprocess_fn = partial(
            self.data_processor.preprocess_fn, **preprocess_conf
        )
        self.data_processor.postprocess_fn = partial(
            self.data_processor.postprocess_fn, **postprocess_conf
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
            inputs=self.input_types,
            outputs=self.output_types,
        ).launch()
