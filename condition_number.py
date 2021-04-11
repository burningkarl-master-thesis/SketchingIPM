"""
Numerical experiments with sketching for least-squares type equations
"""

__author__ = "Karl Welzel"
__license__ = "GPLv3"

import argparse
import collections
import dataclasses
import typing

import numpy as np
import scipy
import scipy.sparse
import sparseqr
import wandb
from codetiming import Timer
from logzero import logger

from config import (
    ConditionNumberExperimentConfig,
    Preconditioning,
    PreconditioningConfig,
    ProblemConfig,
    SketchingConfig,
)
from utils import random_sparse_coefficient_matrix, sparse_sketch

CONFIG_FILE_PARAM = "config_file"


def generate_problem_instance(
    problem_config: ProblemConfig, rng: np.random.Generator
) -> typing.Tuple[scipy.sparse.spmatrix, np.ndarray]:
    """ Generates a problem instance from the given problem configuration """
    coeff_matrix = random_sparse_coefficient_matrix(
        problem_config.m,
        problem_config.n,
        nnz_per_column=problem_config.nnz_per_column,
        rng=rng,
    )
    basis_partition = np.zeros(problem_config.n)
    basis_partition[
        rng.choice(problem_config.n, size=problem_config.m, replace=False)
    ] = np.ones(problem_config.m)
    return coeff_matrix, basis_partition


def precondition(
    config: PreconditioningConfig,
    half_spd_matrix: scipy.sparse.spmatrix,
    sketched_half_spd_matrix: scipy.sparse.spmatrix,
) -> typing.Tuple[np.ndarray, typing.Dict[str, typing.Any]]:
    decomposition_timer = Timer()
    decomposition_timer.last = 0

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
            "half_spd_rank": np.linalg.matrix_rank(half_spd_matrix.toarray()),
            "sketched_half_spd_rank": np.linalg.matrix_rank(
                sketched_half_spd_matrix.toarray()
            ),
            "preconditioned_spd_rank": np.linalg.matrix_rank(preconditioned_spd_matrix),
            "condition_number_duration": condition_number_timer.last,
            "decomposition_duration": decomposition_timer.last,
            "product_duration": product_timer.last,
        },
    )


def run_experiment(
    experiment_config: ConditionNumberExperimentConfig,
    problem_config: ProblemConfig,
    sketching_configs: typing.List[SketchingConfig],
    preconditioning_configs: typing.List[PreconditioningConfig],
) -> None:
    logger.info(f"Starting sketching experiment: {problem_config=}")
    coeff_matrix, basis_partition = generate_problem_instance(
        problem_config, rng=problem_config.rng
    )

    sketched_matrices = collections.defaultdict(dict)
    sketching_metrics = collections.defaultdict(dict)
    for mu in np.logspace(
        problem_config.mu_max_exponent,
        problem_config.mu_min_exponent,
        problem_config.mu_steps,
    ):
        sqrt_mu = np.sqrt(mu)
        half_diag = scipy.sparse.diags(
            (basis_partition == 1) * (1 / sqrt_mu) + (basis_partition == 0) * sqrt_mu
        )
        half_spd_matrix = half_diag * coeff_matrix.T

        for sketching_config in sketching_configs:
            with Timer(logger=None) as generate_sketch_timer:
                sketching_matrix = sparse_sketch(
                    int(sketching_config.w_factor * problem_config.m),
                    problem_config.n,
                    sketching_config.s,
                )
            with Timer(logger=None) as sketching_timer:
                sketched_half_spd_matrix = sketching_matrix * half_spd_matrix

            sketched_matrices[sketching_config][mu] = {
                "half_spd_matrix": half_spd_matrix,
                "sketched_half_spd_matrix": sketched_half_spd_matrix,
            }
            sketching_metrics[sketching_config][mu] = {
                "generate_sketch_duration": generate_sketch_timer.last,
                "sketching_duration": sketching_timer.last,
            }
            logger.debug(f"Sketched matrix for {sketching_config=} and {mu=}")
    logger.debug("Finished sketching")

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

            condition_numbers = []
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
                condition_numbers.append(preconditioning_metrics["condition_number"])

            run.summary["condition_number"] = np.median(condition_numbers)
            run.finish()


def main(args):
    logger.info("Reading config files")
    logger.info(args)
    config_file = vars(args)[CONFIG_FILE_PARAM]
    if config_file is None:
        config = ConditionNumberExperimentConfig()
    else:
        config = ConditionNumberExperimentConfig.from_file(config_file)
    logger.info(config)

    for i in range(config.number_of_runs):
        for problem_config in config.problem_configs():
            run_experiment(
                experiment_config=config,
                problem_config=problem_config,
                sketching_configs=config.sketching_configs(),
                preconditioning_configs=config.preconditioning_configs(),
            )


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", action="store", dest=CONFIG_FILE_PARAM)
    main(parser.parse_args())
