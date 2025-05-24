import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from mcda.dataset import MCDADataset


def create_correlation_matrix(df, output_path):
    corr_matrix = df.corr()
    n_cols = len(corr_matrix.columns)
    font_size = max(10, min(18, 20 - 0.7 * n_cols))
    fig_size = max(8, n_cols * 0.6)
    sns.set_context("notebook")
    plt.figure(figsize=(fig_size, fig_size * 0.8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="crest",
        annot_kws={"size": font_size}
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate correlation matrix from CSV file")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument(
        "--output_dir", help="Directory to save the output (default: corr in the same directory as input file)"
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent / "corr"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    output_path = output_dir / f"{input_path.stem}_correlation.png"
    dataset = MCDADataset.read_csv(input_path, convert_to_gain=False)
    if dataset.data.empty:
        print("Error: No valid data found in the CSV file.")
        return

    saved_path = create_correlation_matrix(dataset.data, output_path)
    print(f"Correlation matrix saved to: {saved_path}")


if __name__ == "__main__":
    main()
