"""
Numerical experiments with sketching for linear programming with IPMs
"""

__author__ = "Karl Welzel"
__license__ = "GPLv3"

import argparse
import collections
import dataclasses
import typing

import numpy as np
import scipy.optimize
import wandb
from codetiming import Timer
from logzero import logger

from config import (
    IpmExperimentConfig,
    Preconditioning,
    PreconditioningConfig,
    ProblemConfig,
    SketchingConfig,
)
from utils import random_sparse_coefficient_matrix

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
    sketching_config: SketchingConfig,
    preconditioning_config: PreconditioningConfig,
) -> None:
    logger.info(f"Starting IPM experiment: {problem_config=}")
    a, b, c = generate_random_ipm_instance(
        problem_config.m,
        problem_config.n,
        problem_config.nnz_per_column,
        problem_config.rng,
    )

    if preconditioning_config.preconditioning not in [
        Preconditioning.NONE,
        Preconditioning.QR,
    ]:
        raise ValueError("Not iplemented yet!")

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

    result = scipy.optimize.linprog(
        c,
        A_eq=a,
        b_eq=b,
        options={
            "_sparse_presolve": True,
            "sparse": True,
            "cholesky": True,
            "iterative": True,
            "linear_operators": True,
            "sym_pos": True,
            "disp": True,
            "presolve": False,
            "autoscale": False,
            "pc": True,
            "ip": True,
            "tol": 1e-8,
            "maxiter": 1000,
            "preconditioning_method": "none"
            if preconditioning_config.preconditioning is Preconditioning.NONE
            else "sketching",
            "sketching_factor": sketching_config.w_factor,
            "sketching_sparsity": sketching_config.s,
            "triangular_solve": False,
        },
    )
    # logger.debug(result)
    logger.debug(f"{sum(np.isclose(result.x, np.zeros(problem_config.n), atol=1e-7))}")

    run.finish()


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
            for sketching_config in config.sketching_configs():
                for preconditioning_config in config.preconditioning_configs():
                    run_experiment(
                        experiment_config=config,
                        problem_config=problem_config,
                        sketching_config=sketching_config,
                        preconditioning_config=preconditioning_config,
                    )


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", action="store", dest=CONFIG_FILE_PARAM)
    main(parser.parse_args())
