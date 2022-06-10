# -*- coding: utf-8 -*-

"""
usage:
"""

import uuid
import logging

task_uuid = str(uuid.uuid4())[:8]

def get_logger(name, filename):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="[{}] - ".format(task_uuid) + '%(asctime)s [%(filename)s %(lineno)d] %(name)s - %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S')

    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
