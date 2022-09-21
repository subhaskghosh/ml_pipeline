"""
Logging related settings
"""
__author__ = "Subhas K. Ghosh"
__version__ = "1.0"

import logging
from logging.handlers import RotatingFileHandler
import sys

def get_logger(name):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s [%(filename)s:%(lineno)s - %(funcName)s()] %(message)s",
        datefmt="%Y-%m-%d,%H:%M:%S",
    )

    logger = logging.getLogger(name)

    # create a stream handler associated with the console (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    # set the logging level for this console handler (default: INFO)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # create a rotating file handler associated with an external file
    file_handler = RotatingFileHandler(
        'ml_pipeline.log', mode='w', maxBytes=(1048576 * 5), backupCount=2, encoding=None, delay=0)
    # set the logging level for this file handler (default: DEBUG)
    file_handler.setLevel(logging.INFO)
    # attach this file handler to the logger
    logger.addHandler(file_handler)

    return logger
