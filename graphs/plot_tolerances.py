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
    ax.set(
        xlabel="",
        ylabel="Relative errors",
        xticklabels=[
            r"$\mathbf{R}^{-T}\mathbf{A}\mathbf{D}^2\mathbf{A}^T"
            r"\mathbf{R}^{-1} \tilde{\mathbf{q}} = \mathbf{R}^{-T} \mathbf{p}$",
            r"$\mathbf{A}\mathbf{D}^2\mathbf{A}^T \Delta\mathbf{y}"
            r" = \mathbf{p}$",
        ],
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        title="CG iterations",
        handles=handles,
        labels=[r"$50$", r"$75$", r"$100$", r"125", r"Direct"],
    )


def graph_rho_p(ax, all_data, summary_data):
    # Line plot
    # x-axis: _step
    # y-axis: rho_p
    # color: solver_maxiter

    # Filter the data
    filtered_data = all_data.loc[(all_data["seed"] == 946047) & (all_data["_step"] > 50), :].copy()
    filtered_data.loc[:, "accuracy"] = filtered_data.loc[
        :, ("rho_p", "rho_d", "rho_A")
    ].max(axis=1)

    ax.set(yscale="log")
    sns.lineplot(
        data=filtered_data.reset_index(),
        estimator=np.median,
        errorbar=("pi", 90),  # 100% interval = min/max values
        x="_step",
        y="accuracy",
        hue="solver_maxiter",
        palette=sns.color_palette()[: filtered_data.nunique()["solver_maxiter"]],
        ax=ax,
    )
    # ax.set_ylim(bottom=1e-11, top=3)
    ax.set(
        title="",
        xlabel="IPM iteration",
        ylabel=r"$\rho_{\mathrm{tol}}$",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        title="CG iterations",
        handles=handles,
        labels=[r"$50$", r"$75$", r"$100$", r"125", r"Direct"],
    )


def graph_accuracy_vs_time(ax, all_data, summary_data):
    # Scatter plot
    # x-axis: accuracy
    # y-axis: time per iteration
    # color: solver_maxiter

    filtered_data = summary_data.copy()
    filtered_data.loc[:, "accuracy"] = filtered_data.loc[
        :, ("best_rho_p", "best_rho_d", "best_rho_A")
    ].max(axis=1)
    for name in np.unique(all_data["name"]):
        filtered_data.loc[name, "duration"] = (
            all_data.loc[
                all_data["name"] == name,
                (
                    "generate_sketch_duration",
                    "sketching_duration",
                    "decomposition_duration",
                    "product_duration",
                    "solve_duration",
                ),
            ]
            .sum(axis=1)
            .mean(axis=0)
        )

    ax.set(yscale="log")
    sns.scatterplot(
        data=filtered_data.reset_index(),
        x="duration",
        y="accuracy",
        hue="solver_maxiter",
        palette=sns.color_palette()[: filtered_data.nunique()["solver_maxiter"]],
        ax=ax,
    )
    # ax.set_ylim(bottom=1e-11, top=3)
    ax.set(
        title="",
        xlabel="Average time per IPM iteration [s]",
        ylabel=r"$\rho_{\mathrm{tol}}$ (best iteration)",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        title="CG iterations",
        handles=handles,
        labels=[r"$50$", r"$75$", r"$100$", r"125", r"Direct"],
    )


def main(args):
    set_plot_aesthetics()

    all_data, summary_data = load_data(args.group)
    all_data = all_data.where(all_data != "NaN")

    rho_tol_limits = (1e-11, 12)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    graph_residual_norms(axes[0], all_data, summary_data)
    # graph_accuracy_distribution(axes[1], all_data, summary_data)
    graph_rho_p(axes[1], all_data, summary_data)
    axes[1].set_ylim(rho_tol_limits)
    graph_accuracy_vs_time(axes[2], all_data, summary_data)
    axes[2].set_ylim(rho_tol_limits)

    fig.savefig("tolerances.pgf")
    fig.savefig("tolerances.png")


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("group", help="The wandb group to identify runs")

    main(parser.parse_args())
