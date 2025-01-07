#!/bin/bash

total_fold=$(($(ls kf | wc -l) / 2))

for fold in $(seq 1 $total_fold); do
    echo "Running: python train.py --fold $fold"
    python train.py --fold $fold
done