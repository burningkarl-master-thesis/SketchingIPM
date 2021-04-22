"""
Code to visualize the impact of sketching parameters
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


def graph1(ax, all_data, summary_data, s_colors):
    # Line plot
    # x-axis: _step
    # y-axis: condition_number_sketched
    # Show all trajectories for w_factor=2
    # Color differently depending on s

    # Filter the data
    filtered_data = all_data.loc[
        (all_data["w_factor"] == 2) & (all_data["_step"] < 120), :
    ].copy()

    # Include condition numbers without any preconditioning
    filtered_data.loc[:, "s"] = filtered_data.loc[:, "s"].astype(str)
    filtered_data_unsketched = filtered_data.copy()
    filtered_data_unsketched.loc[:, "condition_number_sketched"] = filtered_data.loc[
        :, "condition_number"
    ]
    filtered_data_unsketched.loc[:, "s"] = "0"
    filtered_data = pd.concat([filtered_data, filtered_data_unsketched])

    # Sort by s
    filtered_data.sort_values(by="s", ascending=True, inplace=True)

    ax.set(yscale="log")
    sns.lineplot(
        data=filtered_data.reset_index(),
        estimator=np.median,
        errorbar=("pi", 90),  # 100% interval = min/max values
        x="_step",
        y="condition_number_sketched",
        hue="s",
        palette=s_colors[: filtered_data.nunique()["s"]],
        ax=ax,
    )
    ax.set_ylim(bottom=1)
    ax.set(
        xlabel="IPM iterations",
        ylabel=r"$\kappa_2\left(\mathbf{R}^{-T}\mathbf{A}\mathbf{D}^2"
        r"\mathbf{A}^T\mathbf{R}^{-1}\right)$",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        title="Preconditioning",
        handles=handles,
        labels=[
            "None",
            r"$w = 2m$, $s = 2$",
            r"$w = 2m$, $s = 3$",
            r"$w = 2m$, $s = 4$",
            r"$w = 2m$, $s = 5$",
        ],
        loc="upper left",
    )


def graph2(ax, all_data, summary_data, s_colors):
    # Boxplot
    # x-axis: w_factor -> s
    # y-axis: condition_number_sketched
    # Color differently depending on s

    # Filter the data
    filtered_data = all_data.loc[
        (all_data["w_factor"] != 1) & (all_data["s"] != 2), :
    ].copy()

    ax.set(yscale="log")
    sns.barplot(
        data=filtered_data.reset_index(),
        x="w_factor",
        hue="s",
        y="condition_number_sketched",
        estimator=np.median,
        ci="decile",
        palette=s_colors[2 : 2 + filtered_data.nunique()["s"]],
        ax=ax,
    )
    ax.set_ylim(bottom=1)
    ax.set(
        xlabel="$w/m$",
        ylabel=r"$\kappa_2\left(\mathbf{R}^{-T}\mathbf{A}\mathbf{D}^2"
        r"\mathbf{A}^T\mathbf{R}^{-1}\right)$",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        title="Preconditioning",
        handles=handles,
        labels=["$s = 3$", "$s = 4$", "$s=5$"],
    )


def graph3(ax, all_data, summary_data, s_colors):
    # Bar chart
    # x-axis: w_factor -> s
    # y-axis: density_sketched
    # Color differently depending on s

    # Filter the data
    filtered_data = all_data.loc[
        (all_data["w_factor"] != 1) & (all_data["s"] != 2), :
    ].copy()

    sns.barplot(
        data=filtered_data.reset_index(),
        x="w_factor",
        hue="s",
        y="nnz_sketched",
        estimator=np.median,
        ci="decile",
        palette=s_colors[2 : 2 + filtered_data.nunique()["s"]],
        ax=ax,
    )
    ax.hlines(
        filtered_data["nnz_coefficient"],
        0,
        1,
        transform=ax.get_yaxis_transform(),
        colors="r",
    )
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax.set(
        xlabel="$w/m$",
        ylabel=r"$\mathrm{nnz}\left(\mathbf{W}\mathbf{D}\mathbf{A}^T\right)$",
    )
    ax.get_legend().remove()


def graph4(ax, all_data, summary_data, s_colors, duration_field):
    # Three bar charts
    # x-axis: w_factor -> s
    # y-axis: generate_sketch_duration, sketching_duration or
    #         decomposition_duration
    # Color differently depending on s

    # Filter the data
    filtered_data = all_data.loc[
        (all_data["w_factor"] != 1) & (all_data["s"] != 2), :
    ].copy()

    # Also try boxplot with ax.set_ylim(bottom=0)
    sns.barplot(
        data=filtered_data.reset_index(),
        x="w_factor",
        hue="s",
        y=duration_field,
        estimator=np.median,
        ci="decile",
        palette=s_colors[2 : 2 + filtered_data.nunique()["s"]],
        ax=ax,
    )
    ylabels = {
        "generate_sketch_duration": r"Time to generate $\mathbf{W}$ [s]",
        "sketching_duration": r"Time to multiply $\mathbf{W}$ and "
        r"$\mathbf{D}\mathbf{A}^T$ [s]",
        "decomposition_duration": r"Time to find $\mathbf{Q}\mathbf{R} = "
        r"\mathbf{W}\mathbf{D}\mathbf{A}^T$ [s]",
    }
    ax.set(xlabel="$w/m$", ylabel=ylabels[duration_field])
    ax.get_legend().remove()


def main(args):
    set_plot_aesthetics()

    s_colors = sns.color_palette()
    s_colors = s_colors[3:5][::-1] + s_colors[:3] + s_colors[5:]

    all_data, summary_data = load_data(args.group)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    graph1(axes[0][0], all_data, summary_data, s_colors)
    graph2(axes[0][1], all_data, summary_data, s_colors)
    graph3(axes[0][2], all_data, summary_data, s_colors)
    graph4(axes[1][0], all_data, summary_data, s_colors, "generate_sketch_duration")
    graph4(axes[1][1], all_data, summary_data, s_colors, "sketching_duration")
    graph4(axes[1][2], all_data, summary_data, s_colors, "decomposition_duration")
    fig.savefig("sketching_parameters.png")

    fig, ax = plt.subplots()
    graph1(ax, all_data, summary_data, s_colors)
    fig.savefig("sketching_parameters_00.pgf")
    fig, ax = plt.subplots()
    graph2(ax, all_data, summary_data, s_colors)
    fig.savefig("sketching_parameters_01.pgf")
    fig, ax = plt.subplots()
    graph3(ax, all_data, summary_data, s_colors)
    fig.savefig("sketching_parameters_02.pgf")
    fig, ax = plt.subplots()
    graph4(ax, all_data, summary_data, s_colors, "generate_sketch_duration")
    fig.savefig("sketching_parameters_10.pgf")
    fig, ax = plt.subplots()
    graph4(ax, all_data, summary_data, s_colors, "sketching_duration")
    fig.savefig("sketching_parameters_11.pgf")
    fig, ax = plt.subplots()
    graph4(ax, all_data, summary_data, s_colors, "decomposition_duration")
    fig.savefig("sketching_parameters_12.pgf")


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("group", help="The wandb group to identify runs")

    main(parser.parse_args())
