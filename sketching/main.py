"""
Numerical experiments with sketching for least-squares type equations
"""

__author__ = "Karl Welzel"
__license__ = "GPLv3"

import argparse
import dataclasses
import numpy as np
import scipy
import scipy.sparse
import time
import wandb
from logzero import logger
from config import SketchingConfig, Preconditioning
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

    for i in range(config.number_of_runs):
        run = wandb.init(project='sketching-ipm-condition-number', config=dataclasses.asdict(config), reinit=True)

        coeff_matrix = random_sparse_coefficient_matrix(config.m, config.n, density=config.density)
        basis_probability = config.m / config.n  # At most m elements are in a basis
        basis_partition = np.random.choice([1, 0], p=[basis_probability, 1 - basis_probability], size=config.n)
        non_basis_partition = np.ones(config.n) - basis_partition

        for mu in np.logspace(-3, 0, 30):
            sqrt_mu = np.sqrt(mu)
            half_diag = scipy.sparse.diags(basis_partition * (1 / sqrt_mu) + non_basis_partition * sqrt_mu)
            spd_matrix = coeff_matrix * half_diag * half_diag * coeff_matrix.T

            start_time = time.time()
            if config.preconditioning is Preconditioning.NONE:
                condition_number = np.linalg.cond(spd_matrix.toarray())
                wandb.log({'mu': mu, 'condition_number': condition_number, 'duration': time.time() - start_time})
                logger.info(f'Unpreconditioned {mu:.1E} {condition_number}')
            elif config.preconditioning is Preconditioning.SKETCHING:
                sketched_matrix = sparse_sketch(config.w, config.n, config.s) * half_diag * coeff_matrix.T
                # logger.info(f'Sketched dimensions: {sketched_matrix.shape}')
                q, r = np.linalg.qr(sketched_matrix.toarray())
                preconditioned_spd_matrix = np.linalg.inv(r.T) @ spd_matrix.toarray() @ np.linalg.inv(r)
                condition_number = np.linalg.cond(preconditioned_spd_matrix)
                wandb.log({'mu': mu, 'condition_number': condition_number, 'duration': time.time() - start_time})
                logger.info(f'SkPreconditioned {mu:.1E} {condition_number}')
            elif config.preconditioning is Preconditioning.FULL:
                # logger.info(f'Original dimensions: {(half_diag * coeff_matrix.T).shape}')
                q, r = np.linalg.qr((half_diag * coeff_matrix.T).toarray())
                preconditioned_spd_matrix = np.linalg.inv(r.T) @ spd_matrix.toarray() @ np.linalg.inv(r)
                condition_number = np.linalg.cond(preconditioned_spd_matrix)
                wandb.log({'mu': mu, 'condition_number': condition_number, 'duration': time.time() - start_time})
                logger.info(f'QRPreconditioned {mu:.1E} {condition_number}')
        run.finish()


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", action="store", dest=CONFIG_FILE_PARAM)
    main(parser.parse_args())
