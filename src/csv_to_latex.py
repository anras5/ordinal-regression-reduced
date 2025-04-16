import argparse
import csv
import math
import subprocess


def create_latex_table(input_file, max_cols=7):
    rows = []
    with open(input_file, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            rows.append(row)

    # Skip rows with number of characteristic points
    rows.pop(0)  # Remove the first row

    # Get cost/gain indicators
    cost_gain_row = rows.pop(0)

    # Get column headers (but we'll replace them with g_1, g_2, etc.)
    headers = rows.pop(0)

    # Start building the LaTeX table
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{Data Analysis}\n"
    latex += "\\label{tab:data}\n"

    # Calculate how many columns we need to display (excluding the first column with labels)
    data_cols = len(headers) - 1

    # Calculate how many rows of tables we need
    if data_cols <= max_cols:
        num_table_rows = 1
        cols_per_row = data_cols
    else:
        # Calculate optimal number of table rows for balanced columns
        num_table_rows = math.ceil(data_cols / max_cols)
        cols_per_row = math.ceil(data_cols / num_table_rows)

    # Find the longest first column label for consistent width
    max_label_width = max(len(row[0]) for row in rows if row)

    # Begin the overall table
    latex += "\\begin{tabular}{@{}c@{}}\n"

    # Process each row of tables
    for table_row in range(num_table_rows):
        start_col = table_row * cols_per_row + 1  # +1 for 1-indexing of g_i
        end_col = min((table_row + 1) * cols_per_row, data_cols) + 1  # +1 for headers indexing

        # Calculate the number of data columns in this row
        current_row_cols = end_col - start_col

        # Begin the subtable with fixed widths
        # Using p{} for fixed width columns
        latex += "\\begin{tabular}{|p{1cm}|" + ">{\\raggedleft\\arraybackslash}p{1.75cm}|" * current_row_cols + "}\n"
        latex += "\\hline\n"

        # Add column headers with g_i notation and arrows for cost/gain
        header_row = []
        for i in range(start_col, end_col):
            if cost_gain_row[i].lower() == 'cost':
                arrow = "$\\downarrow$"  # Downward arrow for cost
            elif cost_gain_row[i].lower() == 'gain':
                arrow = "$\\uparrow$"  # Upward arrow for gain
            else:
                arrow = ""

            header_row.append(f"$g_{{{i}}}$ {arrow}")

        latex += " & " + " & ".join(header_row) + " \\\\\n"
        latex += "\\hline\n"

        # Add data rows for this section
        for row in rows:

            # Format numerical values in scientific notation
            formatted_row = [row[0]]  # First column (label)
            for i in range(start_col, end_col):
                formatted_row.append(f"${float(row[i]):.2e}$")

            latex += " & ".join(formatted_row) + " \\\\\n"

        # End this subtable
        latex += "\\hline\n"
        latex += "\\end{tabular}\n"

        # Add line break between table rows except after the last row
        if table_row < num_table_rows - 1:
            latex += "\\\\ [2ex]\n"

    # End the overall table
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex


def main():
    parser = argparse.ArgumentParser(description='Convert CSV to LaTeX table')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('--output', '-o', help='Path to output LaTeX file (optional)')
    parser.add_argument('--max_cols', type=int, default=7,
                        help='Maximum number of columns per table row (default: 7)')
    parser.add_argument('--clipboard', '-c', action='store_true',
                        help='Copy the LaTeX table to clipboard (works only on macOS)')

    args = parser.parse_args()

    latex_table = create_latex_table(args.input_file, args.max_cols)

    if args.output:
        with open(args.output, 'w') as out_file:
            out_file.write(latex_table)
        print(f"LaTeX table written to {args.output}")

    if args.clipboard:
        subprocess.run("pbcopy", text=True, input=latex_table)
        print("LaTeX table copied to clipboard")

    if not args.output and not args.clipboard:
        print(latex_table)


if __name__ == "__main__":
    main()
