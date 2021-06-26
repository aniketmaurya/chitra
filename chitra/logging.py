from chitra.utility.import_utils import is_installed

if is_installed('loguru'):
    from loguru import logger

    logger.debug('Using loguru for logging!')
else:
    import logging

    logger = logging.getLogger()
