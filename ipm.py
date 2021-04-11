"""
Numerical experiments with sketching for linear programming with IPMs
"""

__author__ = "Karl Welzel"
__license__ = "GPLv3"

import argparse
import collections
import dataclasses
import numpy as np
import scipy
import scipy.sparse
import scipy.optimize
import sparseqr
import typing
import wandb
from codetiming import Timer
from logzero import logger
from typing import Tuple

from config import (
    IpmExperimentConfig,
    ProblemConfig,
    SketchingConfig,
    PreconditioningConfig,
    Preconditioning,
)
from utils import random_sparse_coefficient_matrix, sparse_sketch

CONFIG_FILE_PARAM = "config_file"


def generate_random_ipm_instance(
    m: int, n: int, nnz_per_column: int, rng: np.random.Generator
):
    a = random_sparse_coefficient_matrix(m, n, nnz_per_column, rng)
    b = a * rng.random(size=n)
    c = a.T * rng.normal(size=m) + rng.random(size=n)
    return a, b, c


def run_experiment(
    experiment_config: IpmExperimentConfig,
    problem_config: ProblemConfig,
    sketching_configs: typing.List[SketchingConfig],
    preconditioning_configs: typing.List[PreconditioningConfig],
) -> None:
    logger.info(f"Starting IPM experiment: {problem_config=}")
    a, b, c = generate_random_ipm_instance(
        problem_config.m,
        problem_config.n,
        problem_config.nnz_per_column,
        problem_config.rng,
    )
    result = scipy.optimize.linprog(
        c,
        A_eq=a,
        b_eq=b,
        options={
            "_sparse_presolve": True,
            "disp": True,
            "presolve": False,
            "autoscale": False,
            "ip": False,
            "tol": 1e-12,
        },
    )
    # logger.debug(result)
    logger.debug(f"{sum(np.isclose(result.x, np.zeros(problem_config.n), atol=1e-7))}")


def main(args):
    logger.info("Reading config files")
    logger.info(args)
    config_file = vars(args)[CONFIG_FILE_PARAM]
    if config_file is None:
        config = IpmExperimentConfig()
    else:
        config = IpmExperimentConfig.from_file(config_file)
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
