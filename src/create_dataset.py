import argparse

import numpy as np
import pandas as pd

from mcda.dataset import MCDADataset


def create_synthetic_dataset(original_dataset_path, output_path, size):
    original_data = MCDADataset.read_csv(original_dataset_path, convert_to_gain=False)
    df = original_data.data

    covariance_matrix = df.cov()
    means = df.mean()

    rng = np.random.default_rng(42)
    synthetic_data = rng.multivariate_normal(mean=means, cov=covariance_matrix, size=size)

    synthetic_data_df = pd.DataFrame(synthetic_data, columns=df.columns).astype(np.float64)

    synthetic_dataset = MCDADataset(synthetic_data_df, original_data.criteria)
    synthetic_dataset.write_csv(output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset based on an original dataset.")
    parser.add_argument("--original", required=True, help="Path to the original dataset (CSV file).")
    parser.add_argument("--output", required=True, help="Path to save the synthetic dataset (CSV file).")
    parser.add_argument("--size", type=int, required=True, help="Number of samples in the synthetic dataset.")

    args = parser.parse_args()

    create_synthetic_dataset(args.original, args.output, args.size)


if __name__ == "__main__":
    main()
