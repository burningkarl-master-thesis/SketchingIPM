"""
Numerical experiments with sketching for linear programming with IPMs
"""

__author__ = "Karl Welzel"
__license__ = "GPLv3"

import argparse
import collections
import dataclasses
import logging
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
    IpmConfig,
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
    ipm_config: IpmConfig,
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
            dataclasses.asdict(ipm_config),
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
            "_sparse_presolve": ipm_config.sparse,
            "sparse": ipm_config.sparse,
            "cholesky": ipm_config.symmetric_positive_definite,
            "sym_pos": ipm_config.symmetric_positive_definite,
            "iterative": ipm_config.iterative,
            "linear_operators": ipm_config.linear_operators,
            "triangular_solve": ipm_config.triangular_solve,
            "solver_rtol": ipm_config.solver_relative_tolerance,
            "solver_atol": ipm_config.solver_absolute_tolerance,
            "solver_maxiter": ipm_config.solver_maxiter,
            "pc": ipm_config.predictor_corrector,
            "ip": ipm_config.predictor_corrector,
            "disp": True,
            "presolve": ipm_config.presolve,
            "autoscale": ipm_config.autoscale,
            "tol": ipm_config.tolerance,
            "maxiter": ipm_config.maxiter,
            "preconditioning_method": "none"
            if preconditioning_config.preconditioning is Preconditioning.NONE
            else "sketching",
            "sketching_factor": sketching_config.w_factor,
            "sketching_sparsity": sketching_config.s,
        },
    )
    # logger.debug(result)
    logger.debug(
        f"{sum(np.isclose(result.x, np.zeros(problem_config.n), atol=1e-7))}"
    )

    run.summary["status"] = result.status
    run.summary["success"] = result.success
    run.summary["outer_iterations"] = result.nit
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

    logger.setLevel(logging.INFO)

    for i in range(config.number_of_runs):
        for problem_config in config.problem_configs():
            for ipm_config in config.ipm_configs():
                for sketching_config in config.sketching_configs():
                    for preconditioning_config in config.preconditioning_configs():
                        run_experiment(
                            experiment_config=config,
                            problem_config=problem_config,
                            ipm_config=ipm_config,
                            sketching_config=sketching_config,
                            preconditioning_config=preconditioning_config,
                        )


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", action="store", dest=CONFIG_FILE_PARAM)
    main(parser.parse_args())
