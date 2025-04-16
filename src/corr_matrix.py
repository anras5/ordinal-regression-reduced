import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from mcda.dataset import MCDADataset


def create_correlation_matrix(df, output_path):
    """Create and save correlation matrix plot."""
    # Calculate correlation matrix
    corr_matrix = df.corr()

    # Create figure
    plt.figure(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="crest")
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate correlation matrix from CSV file')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('--output_dir',
                        help='Directory to save the output (default: corr in the same directory as input file)')

    args = parser.parse_args()

    # Set up the input and output paths
    input_path = Path(args.input_file)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent / "corr"

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    output_path = output_dir / f"{input_path.stem}_correlation.png"

    dataset = MCDADataset.read_csv(input_path, convert_to_gain=False)

    # Check if we have valid data
    if dataset.data.empty:
        print("Error: No valid data found in the CSV file.")
        return

    # Create and save correlation matrix
    saved_path = create_correlation_matrix(dataset.data, output_path)
    print(f"Correlation matrix saved to: {saved_path}")


if __name__ == "__main__":
    main()
