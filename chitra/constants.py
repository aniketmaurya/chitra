import os

IMAGENET_LABEL_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
)

_TF = "tensorflow"
_TF_GPU = "tensorflow-gpu"
_TORCH = "torch"
_TORCHVISION = "torchvision"
_JAX = "jax"
_FLAX = "flax"

CHITRA_URL_SEP = "[-]"
CACHE_DIR = os.environ.get("HOME", ".") + "/.chitra_cache"
IMAGE_CACHE_DIR = f"{CACHE_DIR}/images/"
DOCKER_BASE_URL = "unix://var/run/docker.sock"
