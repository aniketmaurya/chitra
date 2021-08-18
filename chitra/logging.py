from chitra.imports import _LOGURU_INSTALLED

if _LOGURU_INSTALLED:
    from loguru import logger
else:
    import logging

    logger = logging.getLogger()
