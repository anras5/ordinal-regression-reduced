import argparse
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mcda.dataset import MCDADataset
from mcda.report import calculate_heuristics
from mcda.uta import Criterion, check_uta_feasibility
from methods.autoencoder import DominanceAutoEncoder
from methods.mvu import MaximumVarianceUnfolding
from plotting import (
    create_heatmaps,
    create_heatmaps_separate,
    create_line_plots,
    create_line_plots_separate,
    read_results_data,
)

SETS_OF_PREFERENCES = 10


def get_methods(n: int) -> dict:
    """
    Get a dictionary of methods for dimensionality reduction.

    Parameters
    ----------
    n (int): Number of components for the methods.

    Returns
    -------
    dict: Dictionary of methods.

    """
    return {
        "PCA": Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=n, random_state=42))]),
        "KernelPCA": Pipeline(
            [("scaler", StandardScaler()), ("kpca", KernelPCA(n_components=n, random_state=42, kernel="sigmoid"))]
        ),
        "Isomap": Pipeline([("scaler", StandardScaler()), ("isomap", Isomap(n_components=n))]),
        "MVU": Pipeline([("scaler", StandardScaler()), ("mvu", MaximumVarianceUnfolding(n_components=n, seed=42))]),
        "DAE": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("dae", DominanceAutoEncoder(latent_dim=n, num_epochs=1000, random_state=42, verbose=False)),
            ]
        ),
    }


def is_dominating(a_values: List[float], b_values: List[float]) -> bool:
    """
    Check if a is dominating b

    Parameters
    ----------
    a_values (List[float, ...]): List of performances for a
    b_values (List[float, ...]): List of performances for b

    Returns
    -------
    bool: True if a is dominating b, False otherwise

    """
    return all(a >= b for a, b in zip(a_values, b_values)) and any(a > b for a, b in zip(a_values, b_values))


def get_domination_df(dataset: MCDADataset, n_components: List[int]) -> pd.DataFrame:
    """
    Get a DataFrame with domination relations for each method and original dataset.

    Parameters
    ----------
    dataset (MCDADataset): Dataset to check domination relations
    n_components (List[int]): List of number of components for methods

    Returns
    -------
    pd.DataFrame: DataFrame with domination relations for each method and original dataset. Indices are tuples that can be passed to `preferences`.
    """

    domination = defaultdict(dict)
    for n in n_components:
        # check for each method
        for method_name, method in get_methods(n).items():
            df_m = (
                pd.DataFrame(method.fit_transform(dataset.data), index=dataset.data.index, columns=range(n))
                .map(lambda x: f"{x:.4f}")
                .astype(np.float64)
            )
            for alt_1 in df_m.index:
                for alt_2 in df_m.index:
                    if alt_1 == alt_2:
                        continue
                    domination[(method_name, n)][(alt_2, alt_1)] = is_dominating(
                        df_m.loc[alt_1, :].tolist(), df_m.loc[alt_2, :].tolist()
                    )

        # check for original dataset
        for alt_1 in dataset.data.index:
            for alt_2 in dataset.data.index:
                if alt_1 == alt_2:
                    continue
                domination[("original", n)][(alt_2, alt_1)] = is_dominating(
                    dataset.data.loc[alt_1, :].tolist(), dataset.data.loc[alt_2, :].tolist()
                )

    df_domination = pd.DataFrame(domination)
    return df_domination[df_domination.eq(False).all(axis=1)]


def get_possible_preferences(dataset: MCDADataset, components, n_preferences, points) -> List[Tuple[str, str]]:
    """
    Get a list of possible preferences for the dataset.
    Returns random preferences that are feasible for all components, all methods and all number of points.

    Parameters
    ----------
    dataset (MCDADataset): Dataset to get preferences from.
    components (List[int]): List of number of components for the methods.
    n_preferences (int): Number of preferences to generate.
    points (List[int]): List of number of points for the criteria.

    Returns
    -------
    List[Tuple[str, str]]: List of possible preferences for the dataset.
    """
    preferences_list = []
    tries = 0  # holds the number of tries to generate preferences (used for random state)
    possible_pairs = get_domination_df(dataset, components).index.to_series()

    dataframes = []
    for n in components:
        # check for each method
        for method_name, method in get_methods(n).items():
            df_m = (
                pd.DataFrame(method.fit_transform(dataset.data), index=dataset.data.index, columns=range(n))
                .map(lambda x: f"{x:.4f}")
                .astype(np.float64)
            )
            dataframes.append(df_m)

    while len(preferences_list) < SETS_OF_PREFERENCES:
        # preferences contains tuples (A, B) where A should be preferred to B
        # - the number of tuples is equal to n_preferences
        # - each preference is possible in every space and for every method
        preferences = possible_pairs.sample(n_preferences, random_state=tries).tolist()
        possible = True

        for df_m in dataframes:
            if not possible:
                break
            for number_of_points in points:
                criteria = [Criterion(name, points=number_of_points) for name in df_m.columns]
                try:
                    status = check_uta_feasibility(df_m, preferences, criteria)
                except:
                    status = -1000
                if status != 1:
                    possible = False
                    break
        if possible:
            preferences_list.append(preferences)
        tries += 1
        print(f"current preferences: {len(preferences_list)},\ttries:\t{tries}")

    return preferences_list


def process_preferences(preferences, components, df, available_points, output, _input, metrics, i):
    results = defaultdict(dict)
    for n in components:
        print(f"i: {i}, n: {n}")
        methods = get_methods(n)
        for method_name, method in methods.items():
            df_m = (
                pd.DataFrame(method.fit_transform(df), index=df.index, columns=range(n))
                .map(lambda x: f"{x:.4f}")
                .astype(np.float64)
            )
            for points in available_points:
                criteria = [Criterion(name, points=points) for name in df_m.columns]
                try:
                    f_nec, f_era, f_pwi, f_rai = calculate_heuristics(
                        df_m,
                        preferences,
                        criteria,
                        nec="nec" in metrics,
                        era="era" in metrics,
                        pwi="pwi" in metrics,
                        rai="rai" in metrics,
                    )
                    results[(method_name, f"dims: {n}")][(f"points: {points}", "f_nec")] = f_nec
                    results[(method_name, f"dims: {n}")][(f"points: {points}", "f_era")] = f_era
                    results[(method_name, f"dims: {n}")][(f"points: {points}", "f_pwi")] = f_pwi
                    results[(method_name, f"dims: {n}")][(f"points: {points}", "f_rai")] = f_rai
                except:
                    results[(method_name, f"dims: {n}")][(f"points: {points}", "f_nec")] = None
                    results[(method_name, f"dims: {n}")][(f"points: {points}", "f_era")] = None
                    results[(method_name, f"dims: {n}")][(f"points: {points}", "f_pwi")] = None
                    results[(method_name, f"dims: {n}")][(f"points: {points}", "f_rai")] = None
                    print(f"{i=} {n=} {method_name=} {points=} failed!")

    file_path = output / Path(f"no_{i}.csv")
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_index(axis=1, level=[0, 1])
    if file_path.exists():
        df_old = pd.read_csv(file_path, header=[0, 1], index_col=[0, 1])
        df_new = df_old.join(df_results, how="outer")
        df_new = df_new.sort_index(axis=1, level=[0, 1])
    else:
        df_new = df_results
    df_new.to_csv(file_path)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="Path to the input file")
    parser.add_argument(
        "--output_dir", type=Path, help="Path to the output directory (will contain csv files with results and plots)"
    )
    parser.add_argument("--n_preferences", type=int, default=1, help="Number of preferences to generate")
    parser.add_argument(
        "--components", type=int, nargs="+", default=[2, 3, 4, 5, 6], help="Number of components for the methods"
    )
    parser.add_argument("--points", type=int, nargs="+", default=[2, 4, 6], help="Number of points for the criteria")
    parser.add_argument("--cores", type=int, default=10, help="Number of cores to use for parallel processing")
    parser.add_argument(
        "--metrics",
        type=str,
        default=["nec", "era", "pwi", "rai"],
        choices=["nec", "era", "pwi", "rai"],
        nargs="+",
        help="Metrics to calculate",
    )
    parser.add_argument(
        "--skip_calculations",
        action="store_true",
        help="Skip calculations step",
    )
    parser.add_argument(
        "--plots_type",
        type=str,
        default="separate",
        choices=["separate", "combined"],
        help="Type of plots to create",
    )
    args = parser.parse_args()

    # Read the dataset
    dataset: MCDADataset = MCDADataset.read_csv(args.input)

    if not args.skip_calculations:
        preferences_list = get_possible_preferences(dataset, args.components, args.n_preferences, args.points)
        print(preferences_list)
        # Calculate for each method
        Parallel(n_jobs=args.cores)(
            delayed(process_preferences)(
                preferences, args.components, dataset.data, args.points, args.output_dir, args.input, args.metrics, i
            )
            for i, preferences in enumerate(preferences_list)
        )

    # Plotting
    df_results = read_results_data(args.output_dir)
    metrics = [f"f_{metric}" for metric in args.metrics]
    methods = df_results.columns.get_level_values(0).unique()
    if args.plots_type == "combined":
        create_line_plots(df_results, methods, metrics, args.n_preferences, args.output_dir)
        create_heatmaps(df_results, methods, metrics, args.n_preferences, args.output_dir)
    else:
        create_line_plots_separate(df_results, methods, metrics, args.output_dir)
        metrics_reverse_colormap = {f"f_{metric}": False if metric == "nec" else True for metric in args.metrics}
        create_heatmaps_separate(df_results, methods, metrics_reverse_colormap, args.output_dir)
