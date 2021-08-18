import importlib


def is_installed(module_name: str):
    return importlib.util.find_spec(module_name) is not None
