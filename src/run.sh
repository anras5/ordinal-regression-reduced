#!/usr/bin/bash

N=5
DIR="ceiling-structures"


mkdir -p "./data/$DIR/output/preferences_$N/plots/heatmap"
mkdir -p "./data/$DIR/output/preferences_$N/plots/lineplot"
python3 main.py \
  --input "./data/$DIR/dataset.csv" \
  --output_dir "./data/$DIR/output/preferences_$N" \
  --n_preferences "$N" \
  --components 2 3 4 5 6 \
  --points 2 4 6