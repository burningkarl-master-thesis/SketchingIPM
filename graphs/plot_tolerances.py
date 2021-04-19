"""
Code to visualize the impact of limiting the number of CG iterations
"""

__author__ = "Karl Welzel"
__license__ = "GPLv3"

import argparse
import sys

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

    filtered_data = all_data.copy()
    for name in np.unique(all_data["name"]):
        filtered_data.loc[filtered_data["name"] == name, :] = filtered_data.loc[
            (filtered_data["name"] == name)
            & (filtered_data["_step"] < summary_data.loc[name, "best_iteration"]),
            :,
        ]

    filtered_data_1 = filtered_data.copy()
    filtered_data_1.loc[:, "residual"] = filtered_data_1.loc[:, "residual[0]"]
    filtered_data_1.loc[:, "residual_type"] = "preconditioned"
    filtered_data_2 = filtered_data.copy()
    filtered_data_2.loc[:, "residual"] = filtered_data_2.loc[:, "residual_M[0]"]
    filtered_data_2.loc[:, "residual_type"] = "normal"
    filtered_data = pd.concat([filtered_data_1, filtered_data_2])

    ax.set(yscale="log")
    sns.barplot(
        data=filtered_data.reset_index(),
        x="residual_type",
        hue="solver_maxiter",
        y="residual",
        estimator=np.median,
        ci="decile",
        ax=ax,
    )
    ax.set_ylim(bottom=sys.float_info.epsilon)
    ax.set(
        xlabel="",
        ylabel="Relative errors",
        xticklabels=[
            r"$\left\| \mathbf{R}^{-T}\mathbf{A}\mathbf{D}^2\mathbf{A}^T"
            r"\mathbf{R}^{-1} \tilde{\mathbf{x}} - \tilde{\mathbf{r}} \right\|_2"
            r" / \left\| \tilde{\mathbf{r}} \right\|_2$",
            r"$\left\| \mathbf{A}\mathbf{D}^2\mathbf{A}^T \mathbf{x}"
            r" - \mathbf{r} \right\|_2"
            r" / \left\| \mathbf{r} \right\|_2$",
        ],
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        title="CG iterations",
        handles=handles,
        labels=[r"$50$", r"$75$", r"$100$", r"125", r"Direct"],
        loc="upper right",
    )


def graph_rho_p(ax, all_data, summary_data):
    # Line plot
    # x-axis: _step
    # y-axis: rho_p
    # color: solver_maxiter

    # Filter the data
    filtered_data = all_data.loc[(all_data["seed"] == 946047), :].copy()

    ax.set(yscale="log")
    sns.lineplot(
        data=filtered_data.reset_index(),
        estimator=np.median,
        errorbar=("pi", 90),  # 100% interval = min/max values
        x="_step",
        y="rho_p",
        hue="solver_maxiter",
        palette=sns.color_palette()[: filtered_data.nunique()["solver_maxiter"]],
        ax=ax,
    )
    ax.set_ylim(bottom=sys.float_info.epsilon, top=3)
    ax.set(
        title="",
        xlabel="IPM iteration",
        ylabel=r"$\left\|\mathbf{r}_{p}^k\right\|_2 / \left\| \mathbf{r}_p^0 \right\|_2$",
    )
    ax.get_legend().remove()


def graph_accuracy_distribution(ax, all_data, summary_data):
    # Box plot
    # x-axis: solver_maxiter
    # y-axis: rho_p
    # color: solver_maxiter

    filtered_data = summary_data.copy()
    filtered_data.loc[:, "accuracy"] = filtered_data.loc[
        :, ("best_rho_p", "best_rho_d", "best_rho_A")
    ].max(axis=1)

    ax.set(yscale="log")
    sns.barplot(
        data=filtered_data.reset_index(),
        x="solver_maxiter",
        hue="solver_maxiter",
        dodge=False,
        y="accuracy",
        estimator=np.median,
        ci="decile",
        # whis=float("inf"),
        ax=ax,
    )
    ax.set_ylim(bottom=sys.float_info.epsilon, top=3)
    ax.set(
        title="",
        xlabel="CG iterations",
        xticklabels=[r"$50$", r"$75$", r"$100$", r"125", r"Direct"],
        ylabel=r"Accuracy (best iteration)",
    )
    ax.get_legend().remove()


def main(args):
    set_plot_aesthetics()

    all_data, summary_data = load_data(args.group)
    all_data = all_data.where(all_data != "NaN")

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    graph_residual_norms(axes[0], all_data, summary_data)
    graph_rho_p(axes[1], all_data, summary_data)
    graph_rho_p_distribution(axes[2], all_data, summary_data)


    fig.savefig("tolerances.pgf")
    fig.savefig("tolerances.png")


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("group", help="The wandb group to identify runs")

    main(parser.parse_args())
