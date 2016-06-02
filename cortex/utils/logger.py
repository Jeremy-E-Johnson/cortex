'''Module for general logger.

'''

import logging


logger = logging.getLogger('cortex')
logger.setLevel(logging.DEBUG)
logger.propagate = False
file_formatter = logging.Formatter(
    '%(asctime)s:%(name)s[%(levelname)s]:%(message)s')
stream_formatter = logging.Formatter(
    '[%(levelname)s]:%(message)s')

def set_stream_logger(verbosity):
    global logger

    if verbosity == 0:
        level = logging.WARNING
        lstr = 'WARNING'
    elif verbosity == 1:
        level = logging.INFO
        lstr = 'INFO'
    elif verbosity == 2:
        level = logging.DEBUG
        lstr = 'DEBUG'
    else:
        level = logging.INFO
        lstr = 'INFO'
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(stream_formatter)
    logger.addHandler(ch)
    logger.info('Setting logging to %s' % lstr)


def set_file_logger(file_path):
    global logger
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)
    logger.info('Saving logs to %s' % file_path)
