import argparse
import warnings
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, MDS, Isomap, SpectralEmbedding, LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from mcda.dataset import MCDADataset
from mcda.report import calculate_heuristics
from mcda.uta import Criterion

warnings.filterwarnings("ignore")


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
        "PCA": PCA(n_components=n, random_state=42),
        "KernelPCA": KernelPCA(n_components=n, random_state=42),
        "t-SNE": TSNE(n_components=n, perplexity=10, method="exact", random_state=42),
        "MDS": MDS(n_components=n, random_state=42),
        "LLE": LocallyLinearEmbedding(n_components=n, random_state=42),
        "Isomap": Isomap(n_components=n),
        "SpectralEmbedding": SpectralEmbedding(n_components=n, random_state=42),
        "UMAP": UMAP(n_components=n, random_state=42),
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


def get_domination_df(n_components: List[int], dataset: MCDADataset) -> pd.DataFrame:
    """
    Get a DataFrame with domination relations for each method and original dataset.

    Parameters
    ----------
    n_components (List[int]): List of number of components for methods
    dataset (MCDADataset): Dataset to check domination relations

    Returns
    -------
    pd.DataFrame: DataFrame with domination relations for each method and original dataset. Indices are tuples that can be passed to `preferences`.
    """

    # Scaling data for methods
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(dataset.data), columns=dataset.data.columns)

    domination = defaultdict(dict)
    for n in n_components:
        # check for each method
        methods = get_methods(n)
        for method_name, method in methods.items():
            df_m = (
                pd.DataFrame(method.fit_transform(df_scaled), index=dataset.data.index, columns=range(n))
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


def process_preferences(preferences, n_components, df_scaled, available_points, output, _input, i):
    results = defaultdict(dict)
    for n in n_components:
        methods = get_methods(n)
        for method_name, method in methods.items():
            for points in available_points:
                print(f"{i=} {n=} {method_name=} {points=}")
                df_m = (
                    pd.DataFrame(method.fit_transform(df_scaled), index=df_scaled.index, columns=range(n))
                    .map(lambda x: f"{x:.4f}")
                    .astype(np.float64)
                )
                criteria = [Criterion(name, points=points) for name in df_m.columns]
                f_nec, f_era, f_pwi, f_rai = calculate_heuristics(df_m, preferences, criteria)
                results[(method_name, f"dims: {n}")][(f"points: {points}", "f_nec")] = f_nec
                results[(method_name, f"dims: {n}")][(f"points: {points}", "f_era")] = f_era
                results[(method_name, f"dims: {n}")][(f"points: {points}", "f_pwi")] = f_pwi
                results[(method_name, f"dims: {n}")][(f"points: {points}", "f_rai")] = f_rai
                print(f"{i=} {n=} {method_name=} {points=} calculated!")

    file_path = Path(f"./data/output/{_input.stem}_{len(preferences)}_{i}.csv")
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
    parser.add_argument("--output_dir", type=Path, help="Path to the output directory")
    parser.add_argument("--n_preferences", type=int, default=1, help="Number of preferences to generate")
    args = parser.parse_args()

    # Read the dataset
    dataset: MCDADataset = MCDADataset.read_csv(args.input)

    # Define the number of components
    available_points = [2, 3, 4, 5, 6]
    n_components = [2, 3, 4, 5, 6]

    # Scale the data
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(dataset.data), columns=dataset.data.columns)

    # Define possible preferences
    df_domination = get_domination_df(n_components, dataset)
    preferences_list = [
        df_domination.index.to_series().sample(args.n_preferences, random_state=i).tolist() for i in range(10)
    ]
    print(preferences_list)

    # Calculate original dataset
    # results_original = defaultdict(dict)
    # for points in available_points:
    #     print(f"points: {points}, method: original")
    #     criteria = [Criterion(name, points=points) for name in dataset.data.columns]
    #     f_nec, f_era, f_pwi, f_rai = calculate_heuristics(dataset.data, PREFERENCES, criteria, 1000)
    #     results_original["original"][(f"points: {points}", "f_nec")] = f_nec
    #     results_original["original"][(f"points: {points}", "f_era")] = f_era
    #     results_original["original"][(f"points: {points}", "f_pwi")] = f_pwi
    #     results_original["original"][(f"points: {points}", "f_rai")] = f_rai
    # df_results_original = pd.DataFrame(results_original)
    # print(df_results_original)
    # df_results_original.to_csv(args.output)

    # Calculate for each method
    Parallel(n_jobs=-3)(
        delayed(process_preferences)(
            preferences, n_components, df_scaled, available_points, args.output_dir, args.input, i
        )
        for i, preferences in enumerate(preferences_list)
    )
