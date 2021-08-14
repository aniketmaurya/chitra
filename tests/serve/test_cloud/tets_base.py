import io

import torch
from chalice import Chalice
from timm import create_model

from chitra.serve.cloud.base import CloudServer

MODEL_PATH = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth"


def model_loader(buffer: io.BytesIO) -> torch.nn.Module:
    model: torch.nn.Module = create_model("efficientnet_b0", pretrained=False).eval()
    model.load_state_dict(torch.load(buffer))
    return model


def test_cloudserver():
    server = CloudServer(
        "image-classification",
        model_path=MODEL_PATH,
        model_loader=model_loader,
    )

    assert isinstance(server, Chalice)
    assert isinstance(server.model, torch.nn.Module)
