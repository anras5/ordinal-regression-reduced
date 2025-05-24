import argparse
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mcda.dataset import MCDADataset
from mcda.uta import Criterion, check_uta_feasibility
from methods.mvu import MaximumVarianceUnfolding

# number of checked sets of preferences is PROCESSES * SAMPLES
PROCESSES = 10
SAMPLES = 100  # number of sets of preferences to be checked by one process


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
        "KernelPCA": Pipeline([("scaler", StandardScaler()), ("kpca", KernelPCA(n_components=n, random_state=42))]),
        "LLE": Pipeline(
            [("scaler", StandardScaler()), ("lle", LocallyLinearEmbedding(n_components=n, random_state=42))]
        ),
        "Isomap": Pipeline([("scaler", StandardScaler()), ("isomap", Isomap(n_components=n))]),
        "MVU": Pipeline([("scaler", StandardScaler()), ("mvu", MaximumVarianceUnfolding(n_components=n, seed=42))]),
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


def get_possible_preferences(
    dataset: MCDADataset, components, n_preferences, points, start_state=0, samples=SAMPLES
) -> List[Tuple[str, str]]:
    """
    Get a list of possible preferences for the dataset.
    Returns random preferences that are feasible for all components, all methods and all number of points.

    Parameters
    ----------
    dataset (MCDADataset): Dataset to get preferences from.
    components (List[int]): List of number of components for the methods.
    n_preferences (int): Number of preferences to generate.
    points (List[int]): List of number of points for the criteria.
    start_state (int): Number used for random state. Goes from `start_state` to `start_state + samples`.
    samples (int): Number of checked preferences.

    Returns
    -------
    List[Tuple[str, str]]: List of possible preferences for the dataset.
    """
    preferences_list = []
    possible_pairs = get_domination_df(dataset, components).index.to_series()
    random_state = start_state
    while random_state < start_state + samples:
        # preferences contains tuples (A, B) where A should be preferred to B
        # - the number of tuples is equal to n_preferences
        # - each preference is possible in every space and for every method
        preferences = possible_pairs.sample(n_preferences, random_state=random_state).tolist()
        possible = True

        # check if preferences are feasible for all components, all methods and all points
        for n in components:
            for method_name, method in get_methods(n).items():
                for number_of_points in points:
                    df_m = (
                        pd.DataFrame(method.fit_transform(dataset.data), index=dataset.data.index, columns=range(n))
                        .map(lambda x: f"{x:.4f}")
                        .astype(np.float64)
                    )
                    criteria = [Criterion(name, points=number_of_points) for name in df_m.columns]
                    status = check_uta_feasibility(df_m, preferences, criteria)
                    if status != 1:
                        possible = False
                        break
        if possible:
            preferences_list.append(preferences)
        random_state += 1

    print(len(preferences_list))
    return preferences_list


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="Path to the input file")
    parser.add_argument("--n_preferences", type=int, default=1, help="Number of preferences to generate")
    parser.add_argument(
        "--components", type=int, nargs="+", default=[2, 3, 4, 5, 6], help="Number of components for the methods"
    )
    parser.add_argument(
        "--points", type=int, nargs="+", default=[2, 3, 4, 5, 6], help="Number of points for the criteria"
    )
    args = parser.parse_args()

    # Read the dataset
    dataset: MCDADataset = MCDADataset.read_csv(args.input)

    Parallel(n_jobs=-3)(
        delayed(get_possible_preferences)(dataset, args.components, args.n_preferences, args.points, SAMPLES * process)
        for process in range(PROCESSES)
    )
