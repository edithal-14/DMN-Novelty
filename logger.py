"""
Initialize logger object to print to console
and write to a file
"""

import logging

def init_logger(logger, filename):
    """
    Init given logger object
    """
    log_format = logging.Formatter("%(asctime)s %(message)s")

    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)
