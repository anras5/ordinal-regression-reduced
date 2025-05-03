#!/usr/bin/bash

# configure these variables:
DIR="insulating-materials"  # folder located in the data directory, indicating the dataset
DATASET_FILENAME="dataset.csv"  # filename of the dataset located in DIR (could be original dataset.csv or synthetic)

# script:
BASENAME="${DATASET_FILENAME%.*}"
for N in {1..6}; do
  mkdir -p "./data/$DIR/output-$BASENAME/preferences_$N/plots/heatmap"
  mkdir -p "./data/$DIR/output-$BASENAME/preferences_$N/plots/lineplot"
  python3 main.py \
    --input "./data/$DIR/$DATASET_FILENAME" \
    --output_dir "./data/$DIR/output-$BASENAME/preferences_$N" \
    --n_preferences "$N" \
    --components 2 3 4 5 6 \
    --points 2 3 4 5 6 \
    --skip_calculations \
    --plots_type "separate"
done
