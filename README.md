# Dimensionality reduction techniques in robust Multiple Criteria Decision Aiding

Dimensionally reduced robust and stochastic ordinal regression.


## Run the app
```
docker compose up
```
Opens up `jupyter lab` enviroment on port 8888.

## Directories:
```
└── src
    ...
    ├── data
    │   ├── building                         # defines the dataset type
    │   │   ├── dataset.csv                  # defines the dataset (could be an original dataset or synethetic)
    │   │   ├── corr                         # directory with correlation matrix
    │   │   │   └── dataset_correlation.png  # correlation matrix of the dataset
    │   │   └── output-dataset               # output directory with results (for dataset.csv)
    │   │       └── preferences_1            # results for "one" preference
    │   │           └── plots                # plots for the results (contains heatmap and lineplot)
    │   ├── ceiling-structures
    ...
```

## Scripts

- `src/check_preferences.py`: Checks how many sets of preferences are satisfiable for a given dataset.
- `src/corr_matrix.py`: Plots the correlation matrix of the dataset.
- `src/create_dataset.py`: Creates a synthetic dataset with a given number of alternatives.
- `src/csv_to_latex.py`: Converts a CSV (with MCDADataset) file to a LaTeX table. (--clipboard option does not work through docker)
- `src/main.py`: Main script for calculations - should be run with `run.sh`.

## Notebooks

- `src/notebooks/case-study.ipynb`: Code for the case study. (chapter 4 in the thesis)
- `src/notebooks/datasets-comparison`: Code for comparing datasets. (number of alternatives and criteria)
