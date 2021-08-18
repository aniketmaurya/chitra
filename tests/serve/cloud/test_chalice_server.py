import io
import os

import numpy as np
import torch
from chalice import Chalice
from timm import create_model

from chitra.core import load_imagenet_labels
from chitra.image import Chitra
from chitra.serve.cloud import ChaliceServer

LABELS = load_imagenet_labels()


MODEL_PATH = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth"

url = (
    "https://raw.githubusercontent.com/aniketmaurya/chitra/master/docs/assets/logo.png"
)


def preprocess(content_raw_body) -> torch.Tensor:
    image = Chitra(content_raw_body)
    image.resize((256, 256))
    x = image.numpy().astype(np.float32)
    x = x / 255.0
    x = torch.from_numpy(x)
    x = x.permute(2, 0, 1).unsqueeze(0)
    return x


def postprocess(data: torch.Tensor) -> str:
    return LABELS[data.argmax(1)]


def model_loader(buffer: io.BytesIO) -> torch.nn.Module:
    model: torch.nn.Module = create_model("efficientnet_b0", pretrained=False).eval()
    model.load_state_dict(torch.load(buffer))
    return model


def test_cloudserver():
    server = ChaliceServer(
        "image-classification",
        model_path=MODEL_PATH,
        model_loader=model_loader,
    )
    assert isinstance(server.app, Chalice)
    assert isinstance(server.model, torch.nn.Module)


def test_index():
    assert ChaliceServer.index() == {"hello": "world"}


def test_predict():
    class Dummy:
        raw_body = url

    server = ChaliceServer(
        "image-classification",
        model_path=MODEL_PATH,
        model_loader=model_loader,
        preprocess_fn=preprocess,
        postprocess_fn=postprocess,
    )
    server.app.current_request = Dummy
    assert isinstance(server.predict(), str)
