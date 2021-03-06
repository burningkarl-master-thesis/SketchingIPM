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

from plotting_utils import load_data, set_plot_aesthetics, save_pgf


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

    filtered_data_10 = filtered_data.copy()
    filtered_data_10.loc[:, "residual"] = filtered_data_10.loc[:, "residual[0]"]
    filtered_data_10.loc[:, "residual_type"] = "preconditioned"
    filtered_data_11 = filtered_data.copy()
    filtered_data_11.loc[:, "residual"] = filtered_data_11.loc[:, "residual[1]"]
    filtered_data_11.loc[:, "residual_type"] = "preconditioned"
    filtered_data_20 = filtered_data.copy()
    filtered_data_20.loc[:, "residual"] = filtered_data_20.loc[:, "residual_M[0]"]
    filtered_data_20.loc[:, "residual_type"] = "normal"
    filtered_data_21 = filtered_data.copy()
    filtered_data_21.loc[:, "residual"] = filtered_data_21.loc[:, "residual_M[0]"]
    filtered_data_21.loc[:, "residual_type"] = "normal"
    filtered_data = pd.concat(
        [filtered_data_10, filtered_data_11, filtered_data_20, filtered_data_21]
    )

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
        ylabel="Relative 2-norm errors",
        xticklabels=[
            r"$\mathbf{R}^{-T}\mathbf{A}\mathbf{D}^2\mathbf{A}^T"
            r"\mathbf{R}^{-1} \tilde{\mathbf{u}} = \mathbf{R}^{-T} \mathbf{p}$",
            r"$\mathbf{A}\mathbf{D}^2\mathbf{A}^T \mathbf{u} = \mathbf{p}$",
        ],
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        title="CG iterations",
        handles=handles,
        labels=[r"$50$", r"$75$", r"$100$", r"125", r"Cholesky"],
    )


def graph_accuracy_history(ax, all_data, summary_data):
    # Line plot
    # x-axis: _step
    # y-axis: rho_p
    # color: solver_maxiter

    # Filter the data
    filtered_data = all_data.loc[
        (all_data["seed"] == 946047) & (all_data["_step"] > 50), :
    ].copy()
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
        labels=[r"$50$", r"$75$", r"$100$", r"125", r"Cholesky"],
    )


def graph_accuracy_vs_time(ax, all_data, summary_data):
    # Scatter plot
    # x-axis: accuracy
    # y-axis: time per iteration
    # color: solver_maxiter

    filtered_data = summary_data.loc[summary_data["solver_maxiter"] < 1000, :].copy()
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
    ax.set_xlim(left=0)
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
        labels=[r"$50$", r"$75$", r"$100$", r"125"],
    )


def main(args):
    set_plot_aesthetics()

    all_data, summary_data = load_data(args.group)
    all_data = all_data.where(all_data != "NaN")

    rho_tol_limits = (1e-11, 12)

    fig, ax = plt.subplots()
    graph_residual_norms(ax, all_data, summary_data)
    save_pgf(fig, "tolerances_residual_norms.pgf")
    fig, ax = plt.subplots()
    graph_accuracy_history(ax, all_data, summary_data)
    ax.set_ylim(rho_tol_limits)
    save_pgf(fig, "tolerances_accuracy_history.pgf")
    fig, ax = plt.subplots()
    graph_accuracy_vs_time(ax, all_data, summary_data)
    ax.set_ylim(rho_tol_limits)
    save_pgf(fig, "tolerances_accuracy_vs_time.pgf", fix_negative=False)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("group", help="The wandb group to identify runs")

    main(parser.parse_args())
