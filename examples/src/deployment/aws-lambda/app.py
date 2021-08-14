import io

import numpy as np
import torch
from timm import create_model

from chitra.core import load_imagenet_labels
from chitra.image import Chitra
from chitra.serve.cloud.aws_serverless import ChaliceServer

MODEL_PATH = "../../../nbs/model.pth"

LABELS = load_imagenet_labels()


def preprocess(content_raw_body) -> torch.Tensor:
    image = Chitra(content_raw_body)
    image.resize((256, 256)).show()
    x = image.numpy().astype(np.float32)
    x = x / 255.0
    x = torch.from_numpy(x)
    x = x.permute(2, 0, 1).unsqueeze(0)
    return x


def postprocess(data: torch.Tensor) -> str:
    return LABELS[data.argmax(1)]


def model_loader(buffer: io.BytesIO) -> torch.nn.Module:
    model: torch.nn.Module = create_model("efficientnet_b0", pretrained=False)
    model.load_state_dict(torch.load(buffer))
    return model


server = ChaliceServer(
    "image-classification",
    MODEL_PATH,
    model_loader=model_loader,
    preprocess_fn=preprocess,
    postprocess_fn=postprocess,
)
app = server.app
server.run("route", content_types=["image/jpeg"])
