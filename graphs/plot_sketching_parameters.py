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
import pathlib
import seaborn as sns
import wandb

from logzero import logger

dataframe_directory = pathlib.Path.cwd()

# matplotlib.use("pgf")
# matplotlib.rcParams.update(
#     {
#         "pgf.texsystem": "pdflatex",
#         "font.family": "serif",
#         "text.usetex": True,
#         "pgf.rcfonts": False,
#     }
# )


def rename_legend_labels(facet_grid, title, new_labels):
    legend_data = dict(zip(new_labels, facet_grid._legend_data.values()))
    facet_grid.add_legend(legend_data=legend_data, title=title)


def graph1(ax, all_data, summary_data):
    # Line plot
    # x-axis: _step (0 ... 46)
    # y-axis: condition_number_sketched
    # Show all trajectories for w_factor=2
    # Color differently depending on s

    logger.info("Starting to plot graph 1")

    # Filter the data
    filtered_data = all_data.loc[
        (all_data["_step"] <= 46) & (all_data["w_factor"] == 2), :
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

    ax.set(
        yscale="log",
        xlabel="IPM iterations",
        ylabel=r"$\kappa_2\left(\mathbf{R}^{-T}\mathbf{A}\mathbf{D}^2"
        r"\mathbf{A}^T\mathbf{R}^{-1}\right)$",
    )
    sns.lineplot(
        data=filtered_data.reset_index(),
        estimator=np.median,
        errorbar=("pi", 100),  # 100% interval = min/max values
        x="_step",
        y="condition_number_sketched",
        hue="s",
        ax=ax,
    )
    ax.legend(
        title="Preconditioning",
        labels=[
            "None",
            r"$w = 2m$, $s = 2$",
            r"$w = 2m$, $s = 3$",
            r"$w = 2m$, $s = 4$",
            r"$w = 2m$, $s = 5$",
        ],
    )


def graph2(all_data, summary_data):
    # Boxplot
    # x-axis: w_factor -> s
    # y-axis: condition_number_sketched
    # Color differently depending on s

    # Filter the data
    # Restrict the step to ensure all runs are equally weighted
    filtered_data = all_data.loc[
        (all_data["_step"] <= 46) & (all_data["w_factor"] != 1) & (all_data["s"] != 2),
        :,
    ].copy()

    facet_grid = sns.catplot(
        data=filtered_data,
        kind="box",
        x="w_factor",
        hue="s",
        y="condition_number_sketched",
        whis=5,
        legend_out=False,
    )
    facet_grid.set(
        yscale="log",
        title="Condition numbers",
        xlabel="$w/m$",
        ylabel=r"$\kappa_2\left(\mathbf{R}^{-T}\mathbf{A}\mathbf{D}^2"
        r"\mathbf{A}^T\mathbf{R}^{-1}\right)$",
    )
    rename_legend_labels(
        facet_grid=facet_grid,
        title="Preconditioning",
        new_labels=["$s = 3$", "$s = 4$", "$s=5$"],
    )

    facet_grid.savefig("sketching_parameters_2.pgf")
    logger.info("Saved sketching_parameters_2.pgf")
    facet_grid.savefig("sketching_parameters_2.png")
    logger.info("Saved sketching_parameters_2.png")


def graph3(all_data, summary_data):
    # Stacked histogram
    # x-axis: w_factor -> s
    # y-axis: generate_sketch_duration / sketching_duration
    # Color differently depending on s

    # Filter the data
    # Restrict the step to ensure all runs are equally weighted
    filtered_data = all_data.loc[
        (all_data["_step"] <= 46) & (all_data["w_factor"] != 1) & (all_data["s"] != 2),
        :,
    ].copy()
    filtered_data = pd.concat(
        [
            filtered_data.rename(
                columns={"generate_sketch_duration": "duration"}
            ).assign(duration_mode="generate"),
            filtered_data.rename(columns={"sketching_duration": "duration"}).assign(
                duration_mode="multiply"
            ),
            # filtered_data.rename(columns={"decomposition_duration": "duration"}).assign(
            #     duration_mode="decompose"
            # ),
            # filtered_data.rename(columns={"product_duration": "duration"}).assign(
            #     duration_mode="invert"
            # ),
        ]
    )

    facet_grid = sns.catplot(
        data=filtered_data,
        kind="bar",
        x="w_factor",
        hue="s",
        col="duration_mode",
        y="duration",
        ci="sd",
    )
    # facet_grid.set(
    #     title="Sketching duration",
    #     xlabel="$w/m$",
    #     ylabel=r"Time [s]",
    # )

    # rename_legend_labels(
    #     facet_grid=facet_grid,
    #     title="Preconditioning",
    #     new_labels=["$s = 3$", "$s = 4$"],
    # )

    facet_grid.savefig("sketching_parameters_3.pgf")
    logger.info("Saved sketching_parameters_3.pgf")
    facet_grid.savefig("sketching_parameters_3.png")
    logger.info("Saved sketching_parameters_3.png")


def main(args):
    all_data_filename = dataframe_directory / (args.group + "_all_data.pkl")
    summary_data_filename = dataframe_directory / (args.group + "_summary_data.pkl")
    if all_data_filename.exists() and summary_data_filename.exists():
        logger.info(f"{all_data_filename} and {summary_data_filename} found")
        all_data = pd.read_pickle(all_data_filename)
        summary_data = pd.read_pickle(summary_data_filename)
    else:
        logger.info(f"Downloading and processing the data...")
        api = wandb.Api()
        runs = api.runs(
            "karl-welzel/sketching-ipm-condition-number",
            filters={"group": args.group},
        )
        all_data = pd.concat(
            [run.history().assign(name=run.name, **run.config) for run in runs]
        )
        summary_data = pd.DataFrame(
            [{"name": run.name, **run.summary, **run.config} for run in runs]
        )
        all_data.to_pickle(all_data_filename)
        summary_data.to_pickle(summary_data_filename)
        logger.info(f"Saved to {all_data_filename} and {summary_data_filename}")
    logger.info("Done loading.")

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    graph1(axes[0][0], all_data, summary_data)
    # graph2(axes[0][1], all_data, summary_data)
    # graph3(axes[1][0], all_data, summary_data)
    fig.savefig("sketching_parameters.pgf")
    fig.savefig("sketching_parameters.png")


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("group", help="The wandb group to identify runs")

    main(parser.parse_args())
