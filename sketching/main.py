"""
Numerical experiments with sketching for least-squares type equations
"""

__author__ = "Karl Welzel"
__license__ = "GPLv3"

import argparse
import numpy as np
import scipy
import scipy.sparse
from logzero import logger
from config import SketchingConfig
from utils import random_sparse_coefficient_matrix, sparse_sketch

CONFIG_FILE_PARAM = "config_file"


def main(args):
    logger.info("Starting sketching experiment")
    logger.info(args)
    config_file = vars(args)[CONFIG_FILE_PARAM]
    if config_file is None:
        config = SketchingConfig()
    else:
        config = SketchingConfig.from_file(config_file)
    logger.info(config)

    coeff_matrix = random_sparse_coefficient_matrix(config.m, config.n, density=config.density)
    diag_partition = np.random.choice([0, 1], size=config.n)

    for mu in np.logspace(0, -2, 30):
        diag = scipy.sparse.diags(diag_partition * mu + (np.ones(config.n) - diag_partition) * (1 / mu))
        spd_matrix = coeff_matrix * diag * coeff_matrix.T
        logger.info(f'Unpreconditioned {mu:.1E} {np.linalg.cond(spd_matrix.toarray())}')

        half_diag = scipy.sparse.diags(
            diag_partition * np.sqrt(mu) + (np.ones(config.n) - diag_partition) * (1 / np.sqrt(mu)))
        sketched_matrix = sparse_sketch(config.w, config.n, config.s) * half_diag * coeff_matrix.T
        q, r = np.linalg.qr(sketched_matrix.toarray())
        preconditioned_spd_matrix = np.linalg.inv(r.T) @ spd_matrix.toarray() @ np.linalg.inv(r)
        logger.info(f'Preconditioned   {mu:.1E} {np.linalg.cond(preconditioned_spd_matrix)}')
        # logger.info(np.linalg.cond(diag_matrix.toarray()))


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", action="store", dest=CONFIG_FILE_PARAM)
    main(parser.parse_args())
