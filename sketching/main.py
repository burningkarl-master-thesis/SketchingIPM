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
import sparseqr
import typing
import wandb
from codetiming import Timer
from config import SketchingConfig, SketchingConfigProduct, Preconditioning
from logzero import logger
from utils import random_sparse_coefficient_matrix, sparse_sketch

CONFIG_FILE_PARAM = "config_file"


def precondition(config: SketchingConfig, half_spd_matrix: scipy.sparse.spmatrix) -> typing.Tuple[
    np.ndarray, typing.Any]:
    sketching_timer, decomposition_timer, product_timer = Timer(), Timer(), Timer()

    if config.use_sketching and config.preconditioning is not Preconditioning.NONE:
        with Timer(logger=None) as sketching_timer:
            sketched_half_spd_matrix = sparse_sketch(config.w, config.n, config.s) * half_spd_matrix
    else:
        sketched_half_spd_matrix = half_spd_matrix

    if config.preconditioning is Preconditioning.NONE:
        with Timer(logger=None) as product_timer:
            spd_matrix = half_spd_matrix.T * half_spd_matrix
            preconditioned_spd_matrix = spd_matrix.toarray()
    elif config.preconditioning is Preconditioning.QR:
        with Timer(logger=None) as decomposition_timer:
            q, r = np.linalg.qr(sketched_half_spd_matrix.toarray())
        with Timer(logger=None) as product_timer:
            spd_matrix = half_spd_matrix.T * half_spd_matrix
            preconditioned_spd_matrix = np.linalg.inv(r.T) @ spd_matrix.toarray() @ np.linalg.inv(r)
    elif config.preconditioning is Preconditioning.SPARSE_QR:
        with Timer(logger=None) as decomposition_timer:
            q, r, e, rank = sparseqr.qr(sketched_half_spd_matrix, economy=True)
            r = r.toarray()
            p = sparseqr.permutation_vector_to_matrix(e)
        with Timer(logger=None) as product_timer:
            spd_matrix = half_spd_matrix.T * half_spd_matrix
            preconditioned_spd_matrix = np.linalg.inv(r.T) @ (p.T * spd_matrix * p).toarray() @ np.linalg.inv(r)
    elif config.preconditioning is Preconditioning.CHOLESKY:
        with Timer(logger=None) as decomposition_timer:
            sketched_spd_matrix = sketched_half_spd_matrix.T * sketched_half_spd_matrix
            cholesky_factor = np.linalg.cholesky(sketched_spd_matrix.toarray())
        with Timer(logger=None) as product_timer:
            spd_matrix = half_spd_matrix.T * half_spd_matrix
            preconditioned_spd_matrix = np.linalg.inv(cholesky_factor) @ spd_matrix.toarray() @ np.linalg.inv(
                cholesky_factor.T)

    return preconditioned_spd_matrix, (sketching_timer, decomposition_timer, product_timer)


def main(args):
    logger.info("Starting sketching experiment")
    logger.info(args)
    config_file = vars(args)[CONFIG_FILE_PARAM]
    if config_file is None:
        config_product = SketchingConfigProduct()
    else:
        config_product = SketchingConfigProduct.from_file(config_file)
    logger.info(config_product)

    for config in config_product.configs():
        for i in range(config.number_of_runs):
            run = wandb.init(
                project='sketching-ipm-condition-number',
                config=dataclasses.asdict(config),
                reinit=True
            )

            coeff_matrix = random_sparse_coefficient_matrix(config.m, config.n, density=config.density)
            basis_probability = config.m / config.n  # At most m elements are in a basis
            basis_partition = np.random.choice([1, 0], p=[basis_probability, 1 - basis_probability], size=config.n)
            non_basis_partition = np.ones(config.n) - basis_partition

            for mu in np.logspace(0, -12, 10):
                sqrt_mu = np.sqrt(mu)
                half_diag = scipy.sparse.diags(basis_partition * (1 / sqrt_mu) + non_basis_partition * sqrt_mu)
                half_spd_matrix = half_diag * coeff_matrix.T

                preconditioned_spd_matrix, timers = precondition(config, half_spd_matrix)
                with Timer(logger=None) as condition_number_timer:
                    condition_number = np.linalg.cond(preconditioned_spd_matrix)

                sketching_timer, decomposition_timer, product_timer = timers
                wandb.log({
                    'mu': mu, 'condition_number': condition_number,
                    'sketching_duration': sketching_timer.last,
                    'decomposition_duration': decomposition_timer.last,
                    'product_duration': product_timer.last,
                    'condition_number_duration': condition_number_timer.last,
                })

                logger.info(f'{mu:.1E} {condition_number:.1E}')
            run.finish()


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", action="store", dest=CONFIG_FILE_PARAM)
    main(parser.parse_args())
