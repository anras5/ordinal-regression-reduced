# uta-gms-reduced

Dimensionally reduced robust and stochastic ordinal regression.

```
docker compose up
```

Output directories:
```
└── src
    ...
    ├── data
    │   ├── building                 # defines the dataset type
    │   │   ├── dataset.csv          # defines the dataset (could be an original dataset or synethetic)
    │   │   └── output               # output directory with results
    │   │       └── preferences_1    # results for "one" preference
    │   │           └── plots        # plots for the results (contains heatmap and lineplot)
    │   ├── ceiling-structures
    ...
```