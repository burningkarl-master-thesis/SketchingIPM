import pathlib

import matplotlib
import pandas as pd
import seaborn as sns
import wandb

from logzero import logger

dataframe_directory = pathlib.Path.cwd()


def set_plot_aesthetics():
    # matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )
    sns.set_theme("paper", "darkgrid")


def load_data(group):
    all_data_filename = dataframe_directory / (group + "_all_data.pkl")
    summary_data_filename = dataframe_directory / (group + "_summary_data.pkl")
    if all_data_filename.exists() and summary_data_filename.exists():
        logger.info(f"{all_data_filename} and {summary_data_filename} found")
        all_data = pd.read_pickle(all_data_filename)
        summary_data = pd.read_pickle(summary_data_filename)
    else:
        logger.info(f"Downloading and processing the data...")
        api = wandb.Api()
        runs = api.runs(
            "karl-welzel/sketching-ipm-condition-number",
            filters={"group": group},
        )
        all_data = pd.concat(
            [run.history().assign(name=run.name, **run.config) for run in runs]
        )
        summary_data = pd.DataFrame(
            [{**run.summary, **run.config} for run in runs],
            index=[run.name for run in runs],
        )
        all_data.to_pickle(all_data_filename)
        summary_data.to_pickle(summary_data_filename)
        logger.info(f"Saved to {all_data_filename} and {summary_data_filename}")
    logger.info("Done loading.")
    return all_data, summary_data
