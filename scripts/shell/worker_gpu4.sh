#!/bin/bash
DATA_DIR="data/processed_sepsis_full"
tasks=("los")
subsets=("full" "no_treatments" "core" "emergency")
variants=("vanilla" "deep_dki")

for task in "${tasks[@]}"; do
    for subset in "${subsets[@]}"; do
        for variant in "${variants[@]}"; do
            echo "Starting $task $subset $variant on GPU 4"
            cmd="python -u scripts/train_sepsis_benchmarks.py --task $task --model transformer --epochs 30 --batch_size 32 --data_dir $DATA_DIR --feature_subset $subset"
            if [ "$variant" == "deep_dki" ]; then cmd="$cmd --use_kgi"; fi
            $cmd > logs/master_benchmark/${task}_${subset}_${variant}.log 2>&1
        done
    done
done
