from chitra.constants import _FLAX, _JAX, _TF, _TF_GPU, _TORCH, _TORCHVISION
from chitra.import_utils import is_installed

INSTALLED_MODULES = {
    module: is_installed(module)
    for module in (_TF, _TF_GPU, _TORCH, _TORCHVISION, _JAX, _FLAX)
}

_FASTAPI_INSTALLED = is_installed("fastapi")
_UVICORN_INSTALLED = is_installed("uvicorn")
_PYDANTIC_INSTALLED = is_installed("pydantic")
_MULTIPART_INSTALLED = is_installed("multipart")
_SERVE_INSTALLED = (
    _FASTAPI_INSTALLED
    and _UVICORN_INSTALLED
    and _PYDANTIC_INSTALLED
    and _MULTIPART_INSTALLED
)

_LOGURU_INSTALLED = is_installed("loguru")
_RICH_INSTALLED = is_installed("rich")
