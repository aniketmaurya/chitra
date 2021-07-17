from chitra.utility.import_utils import _LOGURU_INSTALLED

if _LOGURU_INSTALLED:
    from loguru import logger

    logger.debug("Using loguru for logging!")
else:
    import logging

    logger = logging.getLogger()
