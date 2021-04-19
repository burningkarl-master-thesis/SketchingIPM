"""
Code to visualize the tradeoff between matrix multiplication and linear operators
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


def graph_sketching_duration(ax, all_data, summary_data):
    # Bar chart
    # x-axis: linear_operators=False, linear_operators=True and
    #         linear_operators=True & triangular_solve=True
    # y-axis: generate_sketch_duration + sketching_duration + decomposition_duration
    pass


def main(args):
    set_plot_aesthetics()

    all_data, summary_data = load_data(args.group)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    graph_sketching_duration(axes[0], all_data, summary_data)
    fig.savefig("linear_operators.pgf")
    fig.savefig("linear_operators.png")


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("group", help="The wandb group to identify runs")

    main(parser.parse_args())
