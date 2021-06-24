import importlib

from chitra.constants import _FLAX
from chitra.constants import _JAX
from chitra.constants import _TF
from chitra.constants import _TF_GPU
from chitra.constants import _TORCH
from chitra.constants import _TORCHVISION


def is_installed(module_name: str):
    return importlib.util.find_spec(module_name) is not None


INSTALLED_MODULES = {
    module: is_installed(module)
    for module in (_TF, _TF_GPU, _TORCH, _TORCHVISION, _JAX, _FLAX)
}
