"""
Initialize logger object to be used across all apwsj scripts
"""

import logging

def init_logger(logger):
    """
    Init given logger object
    """
    log_format = logging.Formatter("%(asctime)s %(message)s")

    file_handler = logging.FileHandler("dlnd_logs")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)
