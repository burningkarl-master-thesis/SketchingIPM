"""
Code to visualize the impact of limiting the number of CG iterations
"""

__author__ = "Karl Welzel"
__license__ = "GPLv3"

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plotting_utils import load_data, set_plot_aesthetics


# To get the ci="decile" code working add the following code in
# _CategoricalStatPlotter.estimate_statistic (categorical.py, line 1443)
#     elif ci == "decile":
#         confint[i].append(np.percentile(stat_data, (5, 95)))


def graph_residual_norms(ax, all_data, summary_data):
    # Bar chart
    # x-axis: solver_maxiter
    # y-axis: residual[0], M_residual[0]
    pass


def main(args):
    set_plot_aesthetics()

    all_data, summary_data = load_data(args.group)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    graph_residual_norms(axes[0], all_data, summary_data)

    fig.savefig("tolerances.pgf")
    fig.savefig("tolerances.png")


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("group", help="The wandb group to identify runs")

    main(parser.parse_args())
