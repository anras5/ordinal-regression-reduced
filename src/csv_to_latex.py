import argparse
import csv
import re


def create_latex_table(input_file):
    rows = []
    with open(input_file, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            rows.append(row)

    # Skip rows with '2's
    if all(cell == '2' for cell in rows[0][1:]):
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

    # Calculate number of columns for the tabular environment
    num_cols = len(headers)
    latex += "\\begin{tabular}{|l|" + "c|" * (num_cols - 1) + "}\n"
    latex += "\\hline\n"

    # Add column headers with g_i notation and arrows for cost/gain
    header_row = []
    for i in range(1, len(headers)):
        if cost_gain_row[i].lower() == 'cost':
            arrow = "$\\downarrow$"  # Downward arrow for cost
        elif cost_gain_row[i].lower() == 'gain':
            arrow = "$\\uparrow$"  # Upward arrow for gain
        else:
            arrow = ""

        header_row.append(f"$g_{{{i}}}$ {arrow}")

    latex += " & " + " & ".join(header_row) + " \\\\\n"
    latex += "\\hline\n"

    # Add data rows
    for row in rows:
        # Skip empty rows or rows with only 2's
        if not row or all(cell == '2' for cell in row[1:]):
            continue

        # Format numerical values in scientific notation
        formatted_row = [row[0]]  # First column (label)
        for cell in row[1:]:
            if cell and re.match(r'^-?\d+\.?\d*$', cell):
                # Format numbers in scientific notation
                formatted_row.append(f"${float(cell):.2e}$")
            else:
                formatted_row.append(cell)

        latex += " & ".join(formatted_row) + " \\\\\n"

    # Finish the table
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex


def main():
    parser = argparse.ArgumentParser(description='Convert CSV to LaTeX table')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('--output', '-o', help='Path to output LaTeX file (optional)')

    args = parser.parse_args()

    latex_table = create_latex_table(args.input_file)

    if args.output:
        with open(args.output, 'w') as out_file:
            out_file.write(latex_table)
        print(f"LaTeX table written to {args.output}")
    else:
        print(latex_table)


if __name__ == "__main__":
    main()