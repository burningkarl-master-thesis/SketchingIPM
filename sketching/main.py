"""
Numerical experiments with sketching for least-squares type equations
"""

__author__ = "Karl Welzel"
__license__ = "GPLv3"

import argparse
import collections
import dataclasses
import multiprocessing
import numpy as np
import scipy
import scipy.sparse
import sparseqr
import typing
import wandb
from codetiming import Timer
from logzero import logger
from typing import Tuple

from config import (
    ExperimentConfig,
    ProblemConfig,
    SketchingConfig,
    PreconditioningConfig,
    Preconditioning,
)
from utils import random_sparse_coefficient_matrix, sparse_sketch

CONFIG_FILE_PARAM = "config_file"


def generate_problem_instance(
    problem_config: ProblemConfig,
) -> Tuple[scipy.sparse.spmatrix, np.ndarray]:
    """ Generates a problem instance from the given problem configuration """
    coeff_matrix = random_sparse_coefficient_matrix(
        problem_config.m, problem_config.n, density=problem_config.density
    )
    basis_probability = (
        problem_config.m / problem_config.n
    )  # At most m elements are in a basis
    basis_partition = np.random.choice(
        [1, 0],
        p=[basis_probability, 1 - basis_probability],
        size=problem_config.n,
    )
    return coeff_matrix, basis_partition


def precondition(
    config: PreconditioningConfig,
    half_spd_matrix: scipy.sparse.spmatrix,
    sketched_half_spd_matrix: scipy.sparse.spmatrix,
) -> typing.Tuple[np.ndarray, typing.Dict[str, typing.Any]]:
    decomposition_timer, product_timer = Timer(), Timer()

    if config.preconditioning is Preconditioning.NONE:
        with Timer(logger=None) as product_timer:
            spd_matrix = half_spd_matrix.T * half_spd_matrix
            preconditioned_spd_matrix = spd_matrix.toarray()
    elif config.preconditioning is Preconditioning.QR:
        with Timer(logger=None) as decomposition_timer:
            q, r = np.linalg.qr(sketched_half_spd_matrix.toarray())
        with Timer(logger=None) as product_timer:
            spd_matrix = half_spd_matrix.T * half_spd_matrix
            preconditioned_spd_matrix = (
                np.linalg.inv(r.T) @ spd_matrix.toarray() @ np.linalg.inv(r)
            )
    elif config.preconditioning is Preconditioning.SPARSE_QR:
        with Timer(logger=None) as decomposition_timer:
            q, r, e, rank = sparseqr.qr(sketched_half_spd_matrix, economy=True)
            r = r.toarray()
            p = sparseqr.permutation_vector_to_matrix(e)
        with Timer(logger=None) as product_timer:
            spd_matrix = half_spd_matrix.T * half_spd_matrix
            preconditioned_spd_matrix = (
                np.linalg.inv(r.T) @ (p.T * spd_matrix * p).toarray() @ np.linalg.inv(r)
            )
    elif config.preconditioning is Preconditioning.CHOLESKY:
        with Timer(logger=None) as decomposition_timer:
            sketched_spd_matrix = sketched_half_spd_matrix.T * sketched_half_spd_matrix
            cholesky_factor = np.linalg.cholesky(sketched_spd_matrix.toarray())
        with Timer(logger=None) as product_timer:
            spd_matrix = half_spd_matrix.T * half_spd_matrix
            preconditioned_spd_matrix = (
                np.linalg.inv(cholesky_factor)
                @ spd_matrix.toarray()
                @ np.linalg.inv(cholesky_factor.T)
            )

    with Timer(logger=None) as condition_number_timer:
        condition_number = np.linalg.cond(preconditioned_spd_matrix)

    return (
        preconditioned_spd_matrix,
        {
            "condition_number": condition_number,
            "preconditioned_spd_rank": np.linalg.matrix_rank(preconditioned_spd_matrix),
            "condition_number_duration": condition_number_timer.last,
            "decomposition_duration": decomposition_timer.last,
            "product_duration": product_timer.last,
        },
    )


def run_experiment(
    experiment_config: ExperimentConfig,
    problem_config: ProblemConfig,
    sketching_configs: typing.List[SketchingConfig],
    preconditioning_configs: typing.List[PreconditioningConfig],
) -> None:
    logger.info(f"Starting sketching experiment: {experiment_config=}")
    (coeff_matrix, basis_partition) = generate_problem_instance(problem_config)
    non_basis_partition = np.ones(problem_config.n) - basis_partition

    sketched_matrices = collections.defaultdict(dict)
    sketching_metrics = collections.defaultdict(dict)
    for mu in np.logspace(
        problem_config.mu_max_exponent,
        problem_config.mu_min_exponent,
        problem_config.mu_steps,
    ):
        sqrt_mu = np.sqrt(mu)
        half_diag = scipy.sparse.diags(
            basis_partition * (1 / sqrt_mu) + non_basis_partition * sqrt_mu
        )
        half_spd_matrix = half_diag * coeff_matrix.T

        for sketching_config in sketching_configs:
            with Timer(logger=None) as sketching_timer:
                sketched_half_spd_matrix = (
                    sparse_sketch(
                        int(sketching_config.w_factor * problem_config.m),
                        problem_config.n,
                        sketching_config.s,
                    )
                    * half_spd_matrix
                )

            sketched_matrices[sketching_config][mu] = {
                "half_spd_matrix": half_spd_matrix,
                "sketched_half_spd_matrix": sketched_half_spd_matrix,
            }
            sketching_metrics[sketching_config][mu] = {
                "sketching_duration": sketching_timer.last,
                "half_spd_rank": np.linalg.matrix_rank(half_spd_matrix.toarray()),
                "sketched_half_spd_rank": np.linalg.matrix_rank(
                    sketched_half_spd_matrix.toarray()
                ),
            }
            logger.debug(
                f"Prepared sketching matrices for {sketching_config=} and {mu=}"
            )

    for sketching_config in sketching_configs:
        for preconditioning_config in preconditioning_configs:
            chained_configs = dict(
                collections.ChainMap(
                    dataclasses.asdict(problem_config),
                    dataclasses.asdict(sketching_config),
                    dataclasses.asdict(preconditioning_config),
                )
            )
            logger.info(chained_configs)
            run = wandb.init(
                project="sketching-ipm-condition-number",
                group=experiment_config.group,
                config=chained_configs,
                reinit=True,
            )

            for mu, instance in sketched_matrices[sketching_config].items():
                try:
                    preconditioned_spd_matrix, preconditioning_metrics = precondition(
                        preconditioning_config, **instance
                    )
                except np.linalg.LinAlgError as error:
                    logger.error(error)
                    continue

                statistics = {
                    "mu": mu,
                    **sketching_metrics[sketching_config][mu],
                    **preconditioning_metrics,
                }
                logger.info(f"{statistics=}")
                wandb.log(statistics)
            run.finish()


def main(args):
    logger.info("Reading config files")
    logger.info(args)
    config_file = vars(args)[CONFIG_FILE_PARAM]
    if config_file is None:
        config = ExperimentConfig()
    else:
        config = ExperimentConfig.from_file(config_file)
    logger.info(config)

    multiprocessing.set_start_method("spawn")
    with multiprocessing.Pool(processes=6) as pool:
        for i in range(config.number_of_runs):
            for problem_config in config.problem_configs():
                pool.apply_async(
                    run_experiment,
                    kwds={
                        "experiment_config": config,
                        "problem_config": problem_config,
                        "sketching_configs": config.sketching_configs(),
                        "preconditioning_configs": config.preconditioning_configs(),
                    },
                )
        pool.close()
        pool.join()


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", action="store", dest=CONFIG_FILE_PARAM)
    main(parser.parse_args())
