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
    # x-axis: linear_operators_mode
    # y-axis: generate_sketch_duration + sketching_duration + decomposition_duration

    # Fill new "sketching_duration" column
    filtered_data = all_data.copy()
    filtered_data.loc[:, "sketching_duration"] = (
        filtered_data.loc[:, "generate_sketch_duration"]
        + filtered_data.loc[:, "sketching_duration"]
        + filtered_data.loc[:, "decomposition_duration"]
    )

    sns.barplot(
        data=filtered_data.reset_index(),
        x="linear_operators_mode",
        hue="linear_operators_mode",
        dodge=False,
        y="sketching_duration",
        estimator=np.median,
        ci="decile",
        ax=ax,
    )
    ax.set(
        xlabel="",
        ylabel="Sketching duration [s]",
        xticklabels=[r"Invert $\mathbf{R}$", "Triangular solve"],
    )
    ax.get_legend().remove()


def main(args):
    set_plot_aesthetics()

    all_data, summary_data = load_data(args.group)

    # Add "linear_operators_mode" column
    all_data.loc[
        all_data["linear_operators"] == False, "linear_operators_mode"
    ] = "direct"
    all_data.loc[
        (all_data["linear_operators"] == True)
        & (all_data["triangular_solve"] == False),
        "linear_operators_mode",
    ] = "linear_operator"
    all_data.loc[
        (all_data["linear_operators"] == True) & (all_data["triangular_solve"] == True),
        "linear_operators_mode",
    ] = "triangular_solve"

    # Sort by linear_operators_mode
    all_data.sort_values(by="linear_operators_mode", ascending=True, inplace=True)

    # Filter out R^{-1} as linear operator
    all_data = all_data.loc[all_data["linear_operators_mode"] != "linear_operator", :]

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
