import io

import numpy as np
import torch
from loguru import logger
from timm import create_model

from chitra.core import load_imagenet_labels
from chitra.image import Chitra
from chitra.serve.cloud.aws_serverless import ChaliceServer

# This path can be anything from filesystem to cloud storage
MODEL_PATH = (
    "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth"
)

LABELS = load_imagenet_labels()
logger.debug(f"labels={LABELS[:5]}...")


# Preprocess the image
def preprocess(content_raw_body) -> torch.Tensor:
    image = Chitra(content_raw_body)
    image.resize((256, 256))
    x = image.numpy().astype(np.float32)
    x = x / 255.0
    x = torch.from_numpy(x)
    x = x.permute(2, 0, 1).unsqueeze(0)
    return x


# Convert Imagenet Index to string label name
def postprocess(data: torch.Tensor) -> str:
    logger.debug(f"predictions = {data}")
    result = LABELS[data.argmax(1)]
    return result


# Loads model from io.BytesIO
def model_loader(buffer: io.BytesIO) -> torch.nn.Module:
    model = create_model("efficientnet_b0", pretrained=False).eval()
    model.load_state_dict(torch.load(buffer))
    return model


server = ChaliceServer(
    api_type="image-classification",
    model_path=MODEL_PATH,
    model_loader=model_loader,
    preprocess_fn=preprocess,
    postprocess_fn=postprocess,
)
app = server.app
server.run("route", content_types=["image/jpeg"])
