"""
Code to summarize data of the iterative vs direct comparison
"""

__author__ = "Karl Welzel"
__license__ = "GPLv3"

import argparse

import numpy as np
import pandas as pd
from logzero import logger

from plotting_utils import load_data


def main(args):
    all_data, summary_data = load_data(args.group)
    preconditioning_methods = np.unique(all_data["preconditioning"])
    median_values = pd.DataFrame(index=preconditioning_methods)
    logger.debug(preconditioning_methods)
    logger.debug(median_values)

    for method in preconditioning_methods:
        median_values.loc[method, "sketch_duration"] = (
            all_data.loc[
                all_data["preconditioning"] == method,
                ("generate_sketch_duration", "sketching_duration"),
            ]
            .sum(axis=1)
            .fillna(0)
            .median(axis=0)
        )
        median_values.loc[method, "decomposition_duration"] = all_data.loc[
            all_data["preconditioning"] == method, "decomposition_duration"
        ].median()
        median_values.loc[method, "solve_duration"] = all_data.loc[
            all_data["preconditioning"] == method, "solve_duration"
        ].median()
        median_values.loc[method, "best_accuracy_5th_percentile"] = (
            summary_data.loc[
                summary_data["preconditioning"] == method,
                ("best_rho_p", "best_rho_d", "best_rho_A"),
            ]
            .max(axis=1)
            .quantile(0.05)
        )
        median_values.loc[method, "best_accuracy_median"] = (
            summary_data.loc[
                summary_data["preconditioning"] == method,
                ("best_rho_p", "best_rho_d", "best_rho_A"),
            ]
            .max(axis=1)
            .median()
        )
        median_values.loc[method, "best_accuracy_95th_percentile"] = (
            summary_data.loc[
                summary_data["preconditioning"] == method,
                ("best_rho_p", "best_rho_d", "best_rho_A"),
            ]
            .max(axis=1)
            .quantile(0.95)
        )
    median_values.loc[:, "total_duration"] = median_values.loc[
        :, ("sketch_duration", "decomposition_duration", "solve_duration")
    ].sum(axis=1)
    print(median_values)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("group", help="The wandb group to identify runs")

    main(parser.parse_args())
