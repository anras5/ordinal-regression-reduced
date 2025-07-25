import argparse
import csv
import math
import subprocess


def create_latex_table(input_file, max_cols=7, column_width=1.75):
    rows = []
    with open(input_file, "r") as file:
        reader = csv.reader(file, delimiter=";")
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

    # Begin the overall table
    latex += "\\begin{tabular}{@{}c@{}}\n"

    # Process each row of tables
    for table_row in range(num_table_rows):
        start_col = table_row * cols_per_row + 1  # +1 for 1-indexing of g_i
        end_col = min((table_row + 1) * cols_per_row, data_cols) + 1  # +1 for headers indexing

        # Calculate the number of data columns in this row
        current_row_cols = end_col - start_col

        # Begin the subtable with fixed widths
        latex += "\\begin{tabular}{|p{1cm}|" + (">{\\raggedleft\\arraybackslash}p{" + f"{column_width}cm}}|") * current_row_cols + "}\n"
        latex += "\\hline\n"

        # Find the smallest power of 10 for each column
        min_exponents = {}
        for col_idx in range(start_col, end_col):
            exponents = []
            for row in rows:
                if row[col_idx].strip():  # Skip empty cells
                    try:
                        value = float(row[col_idx])
                        # Extract exponent from scientific notation
                        if value != 0:  # Avoid log(0)
                            exp = int(math.floor(math.log10(abs(value))))
                            exponents.append(exp)
                    except ValueError:
                        continue
            if exponents:
                min_exponents[col_idx] = min(exponents)
            else:
                min_exponents[col_idx] = 0

        # Add column headers with g_i notation and arrows for cost/gain
        header_row = [""]  # First column is empty
        for i in range(start_col, end_col):
            if cost_gain_row[i].lower() == "cost":
                arrow = "$\\downarrow$"  # Downward arrow for cost
            elif cost_gain_row[i].lower() == "gain":
                arrow = "$\\uparrow$"  # Upward arrow for gain
            else:
                arrow = ""

            header_row.append(f"$g_{{{i}}}$ {arrow}")

        latex += " & ".join(header_row) + " \\\\\n"
        latex += "\\hline\n"

        # Add exponent headers
        exponent_row = [""]  # First column is empty
        for i in range(start_col, end_col):
            exp = min_exponents.get(i, 0)
            if exp != 0:
                exponent_row.append(f"$\\times 10^{{{exp}}}$")
            else:
                exponent_row.append(f" ")

        latex += " & ".join(exponent_row) + " \\\\\n"
        latex += f"\\hhline{'{' + '|='*len(exponent_row) + '|}'}\n"

        # Add data rows for this section
        for row_i, row in enumerate(rows, start=1):
            formatted_row = [f"$a_{{{row_i}}}$"]  # First column (label)
            for i in range(start_col, end_col):
                if row[i].strip():  # Skip empty cells
                    try:
                        value = float(row[i])
                        # Scale the value based on the minimum exponent
                        scaled_value = value / (10 ** min_exponents[i])
                        formatted_row.append(f"${scaled_value:.2f}$")
                    except ValueError:
                        formatted_row.append(row[i])
                else:
                    formatted_row.append("")

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
    parser = argparse.ArgumentParser(description="Convert CSV to LaTeX table")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("--output", "-o", help="Path to output LaTeX file (optional)")
    parser.add_argument("--max_cols", type=int, default=7, help="Maximum number of columns per table row (default: 7)")
    parser.add_argument("--column_width", type=float, default=1.75, help="Width of each column in cm (default: 1.75)")
    parser.add_argument(
        "--clipboard", "-c", action="store_true", help="Copy the LaTeX table to clipboard (works only on macOS)"
    )

    args = parser.parse_args()

    latex_table = create_latex_table(args.input_file, args.max_cols, args.column_width)

    if args.output:
        with open(args.output, "w") as out_file:
            out_file.write(latex_table)
        print(f"LaTeX table written to {args.output}")

    if args.clipboard:
        subprocess.run("pbcopy", text=True, input=latex_table)
        print("LaTeX table copied to clipboard")

    if not args.output and not args.clipboard:
        print(latex_table)


if __name__ == "__main__":
    main()