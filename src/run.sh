#!/usr/bin/bash

mkdir -p ./data/building/output/plots/
python3 main.py --input ./data/building/dataset.csv --output_dir ./data/building/output/ --n_preferences 1