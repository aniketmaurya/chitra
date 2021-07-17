"""A Deep Learning library"""

__version__ = "0.1.0b3"
__license__ = "Apache License 2.0"

from chitra.image import Chitra
from chitra.utility.import_utils import _SERVE_INSTALLED

if _SERVE_INSTALLED:
    from chitra.serve import API, ModelServer, create_api
