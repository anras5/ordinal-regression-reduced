#!/usr/bin/bash

N=3
DIR="ceiling-structures"


mkdir -p "./data/$DIR/output/preferences_$N/plots/heatmap"
mkdir -p "./data/$DIR/output/preferences_$N/plots/lineplot"
python3 main.py \
  --input "./data/$DIR/dataset.csv" \
  --output_dir "./data/$DIR/output/preferences_$N" \
  --n_preferences "$N"