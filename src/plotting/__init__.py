from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def read_results_data(output_dir):
    """
    Read and combine results data from CSV files.

    Parameters
    ----------
    output_dir : Path
        Directory containing the result CSV files

    Returns
    -------
    pd.DataFrame
        Combined dataframe with properly formatted indices and columns
    """
    df_list = []
    for i in range(0, 10):
        try:
            df_results = pd.read_csv(output_dir / Path(f"no_{i}.csv"), header=[0, 1], index_col=[0, 1])
            df_results.columns = pd.MultiIndex.from_tuples(
                [(method, int(dim.split(": ")[1])) for method, dim in df_results.columns],
                names=["method", "dim"],
            )
            df_results.index = pd.MultiIndex.from_tuples([(heu, int(p.split(": ")[1])) for p, heu in df_results.index])
            df_list.append(df_results)
        except FileNotFoundError:
            continue

    return pd.concat(df_list)


def get_subplot_layout(num_metrics):
    """
    Determine the appropriate subplot layout based on the number of metrics.

    Parameters
    ----------
    num_metrics : int
        Number of metrics to plot

    Returns
    -------
    Tuple[int, int]
        Number of rows and columns for subplots
    """
    if num_metrics == 1:
        return 1, 1
    elif num_metrics == 2:
        return 1, 2
    elif num_metrics == 3:
        return 1, 3
    else:  # 4 metrics
        return 2, 2


def create_line_plots(df_results, methods, metrics, n_preferences, output_dir):
    """
    Create line plots for each method showing metrics vs components with different points as lines.

    Parameters
    ----------
    df_results : pd.DataFrame
        Dataframe containing the results data
    methods : List
        List of methods to create plots for
    metrics : List[str]
        List of metrics to include in the plots
    n_preferences : int
        Number of preferences used
    output_dir : Path
        Directory to save the plots to
    """
    num_metrics = len(metrics)
    num_rows, num_cols = get_subplot_layout(num_metrics)

    for method in methods:
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(7.5 * num_cols, 5 * num_rows))
        if num_metrics == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        for i, heuristic in enumerate(metrics):
            if heuristic in df_results.index.get_level_values(0):
                df_r = (
                    df_results[method]
                    .loc[heuristic]
                    .reset_index(names="points")
                    .melt(id_vars=["points"], var_name="components", value_name="value")
                )
                df_r.components = df_r.components.astype(np.float64)

                sns.lineplot(x="components", y="value", hue="points", data=df_r, ax=axes[i], palette="magma")

                axes[i].set_xticks(df_r["components"].unique())
                axes[i].set_ylabel(heuristic)
                handles, _ = axes[i].get_legend_handles_labels()
                axes[i].legend(title="points", handles=handles, loc="upper left", bbox_to_anchor=(1, 1), markerscale=5)

        plt.suptitle(f"Mean heuristics for {method}, {n_preferences} preferences", fontsize=10 + num_metrics * 2.5)
        plt.tight_layout()
        plt.savefig(output_dir / f"plots/lineplot/{method}.png")
        plt.close(fig)


def create_heatmaps(df_results, methods, metrics, n_preferences, output_dir):
    """
    Create heatmaps for each method showing metrics as a function of points (y) and components (x).

    Parameters
    ----------
    df_results : pd.DataFrame
        Dataframe containing the results data
    methods : List
        List of methods to create plots for
    metrics : List[str]
        List of metrics to include in the plots
    n_preferences : int
        Number of preferences used
    output_dir : Path
        Directory to save the plots to
    """
    num_metrics = len(metrics)
    num_rows, num_cols = get_subplot_layout(num_metrics)

    for method in methods:
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
        if num_metrics == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        for i, heuristic in enumerate(metrics):
            if heuristic in df_results.index.get_level_values(0):
                df_r = df_results[method].loc[heuristic].reset_index(names="points").groupby("points").agg("mean")
                sns.heatmap(df_r, ax=axes[i], annot=True, fmt=".2f", cmap="crest")
                axes[i].set_title(heuristic, fontsize=14)
                axes[i].set_ylabel("points", fontsize=10)
                axes[i].set_xlabel("components", fontsize=10)
        plt.suptitle(f"Mean heuristics for {method}, {n_preferences} preferences", fontsize=10 + num_metrics * 2.5)
        plt.tight_layout()
        plt.savefig(output_dir / f"plots/heatmap/{method}.png")
        plt.close(fig)
