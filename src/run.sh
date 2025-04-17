#!/usr/bin/bash

DIR="insulating-materials"
DATASET_FILENAME="dataset30.csv"
BASENAME="${DATASET_FILENAME%.*}"

for N in {1..6}; do
  mkdir -p "./data/$DIR/output-$BASENAME/preferences_$N/plots/heatmap"
  mkdir -p "./data/$DIR/output-$BASENAME/preferences_$N/plots/lineplot"
  python3 main.py \
    --input "./data/$DIR/$DATASET_FILENAME" \
    --output_dir "./data/$DIR/output-$BASENAME/preferences_$N" \
    --n_preferences "$N" \
    --components 2 3 4 5 6 \
    --points 2 3 4 5 6
done
