#!/bin/bash

datasets=("sift-128-euclidean" "glove-100-angular" "deep-image-96-angular" "redcaps-512-angular" "adversarial-100-angular")
memories=("postfiltering" "vamana-tree" "super-postfiltering")

for dataset in "${datasets[@]}"
do
    for memory in "${memories[@]}"
    do
        python3 all_memories.py --dataset "$dataset" --index_type "$memory"
    done
done
