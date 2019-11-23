import logging


def build_logger(logger_file_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)-8s - %(asctime)s - [%(name)s %(funcName)s %(lineno)d] %(message)s')

    ch = logging.FileHandler(logger_file_path, mode ='a')
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
