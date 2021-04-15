"""
Code to visualize the impact of sketching parameters
"""

__author__ = "Karl Welzel"
__license__ = "GPLv3"

import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb

matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)


def graph1(all_data):
    # First graph
    # x-axis: _step (0 ... 46)
    # y-axis: condition_number_sketched
    # Show all trajectories for w_factor=2
    # Color differently depending on s

    # Filter the data
    filtered_data = all_data.loc[
        (all_data["_step"] <= 46) & (all_data["w_factor"] == 2), :
    ].copy()

    # Include condition numbers without any preconditioning
    filtered_data.loc[:, "s"] = filtered_data.loc[:, "s"].apply(str)
    filtered_data_unsketched = filtered_data.copy()
    filtered_data_unsketched.loc[:, "condition_number_sketched"] = filtered_data.loc[
        :, "condition_number"
    ]
    filtered_data_unsketched.loc[:, "s"] = "0"
    filtered_data = pd.concat([filtered_data, filtered_data_unsketched])

    # Sort by s
    filtered_data.sort_values(by="s", ascending=True, inplace=True)

    facet_grid = sns.relplot(
        data=filtered_data,
        kind="line",
        estimator=np.median,
        ci=100,
        x="_step",
        y="condition_number_sketched",
        hue="s",
        facet_kws={"legend_out": False},
    )
    facet_grid.set(
        yscale="log",
        title="Condition numbers",
        xlabel="IPM iterations",
        ylabel=r"$\kappa_2\left(\mathbf{R}^{-T}\mathbf{A}\mathbf{D}^2"
        r"\mathbf{A}^T\mathbf{R}^{-1}\right)$",
    )
    plt.legend(
        title="Preconditioning",
        labels=[
            "None",
            r"$w = 2m$, $s = 2$",
            r"$w = 2m$, $s = 3$",
            r"$w = 2m$, $s = 4$",
        ],
    )
    facet_grid.savefig("sketching_parameters_1.png")
    facet_grid.savefig("sketching_parameters_1.pgf")


def main(args):
    api = wandb.Api()
    runs = api.runs(
        "karl-welzel/sketching-ipm-condition-number",
        filters={"group": args.group, "config.w_factor": 2},
    )
    all_data = pd.concat(
        [run.history().assign(name=run.name, **run.config) for run in runs]
    )

    graph1(all_data)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("group", help="The wandb group to identify runs")

    main(parser.parse_args())
