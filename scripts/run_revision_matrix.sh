#!/bin/bash
# Sepsis Revision Matrix - DGI v2 (Mask-Aware)
# Targets GPU 4-7

TASKS=("ihm" "ss" "vr" "los")
SUBSETS=("full" "no_treatments" "core" "emergency")

# 1. SAITS DGI v2 (GPU 4 & 5)
echo "Starting SAITS DGI v2 runs..."
for task in "${TASKS[@]}"; do
    for subset in "${SUBSETS[@]}"; do
        # Skip if already running (GPU 4/5 are busy now, they will start after current finishes)
        gpu=$((4 + (RANDOM % 2)))
        echo "Queuing SAITS DGI v2: Task=$task, Subset=$subset on GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu python train.py model=joint model.task=$task \
            data.feature_subset=$subset data.processed_dir=data/processed_sepsis_full \
            model.imputator_kwargs.use_kgi=true +model.imputator_kwargs.kgi_mode=dgi_mask \
            trainer.max_epochs=30
        sleep 5
    done
done

# 2. Transformer DGI v2 (GPU 6 & 7)
# These will start after SSSD finishes on these GPUs
echo "Starting Transformer DGI v2 runs..."
for task in "${TASKS[@]}"; do
    for subset in "${SUBSETS[@]}"; do
        gpu=$((6 + (RANDOM % 2)))
        echo "Queuing Transformer DGI v2: Task=$task, Subset=$subset on GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu python scripts/train_sepsis_benchmarks.py \
            --task $task --model transformer --feature_subset $subset \
            --epochs 30 --batch_size 64 --use_kgi --kgi_mode dgi_mask
        sleep 5
    done
done
