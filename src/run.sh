#!/usr/bin/bash

DIR="ceiling-structures"


for N in {1..6}; do
  mkdir -p "./data/$DIR/output/preferences_$N/plots/heatmap"
  mkdir -p "./data/$DIR/output/preferences_$N/plots/lineplot"
  python3 main.py \
    --input "./data/$DIR/dataset.csv" \
    --output_dir "./data/$DIR/output/preferences_$N" \
    --n_preferences "$N" \
    --components 2 3 4 5 6 \
    --points 2 3 4 5 6
done