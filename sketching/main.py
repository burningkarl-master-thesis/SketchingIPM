"""
Numerical experiments with sketching for least-squares type equations
"""

__author__ = "Karl Welzel"
__license__ = "GPLv3"

import argparse
from logzero import logger
from config import SketchingConfig

CONFIG_FILE_PARAM = "config_file"


def main(args):
    logger.info("Starting sketching experiment")
    logger.info(args)
    config_file = vars(args)[CONFIG_FILE_PARAM]
    if config_file is None:
        configuration = SketchingConfig()
    else:
        configuration = SketchingConfig.from_file(config_file)
    logger.info(configuration)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", action="store", dest=CONFIG_FILE_PARAM)
    main(parser.parse_args())
