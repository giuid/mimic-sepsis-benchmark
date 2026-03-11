#!/bin/bash
# scripts/shell/run_sepsis_ablation.sh

DATA_DIR="data/processed_sepsis_full"
EPOCHS=30
BS=32

mkdir -p logs/ablation

run_group() {
    local gpu=$1
    local subset=$2
    local session_name="ablation_${subset}"
    
    local cmd_vanilla="python -u scripts/train_sepsis_benchmarks.py --task ihm --model transformer --epochs $EPOCHS --batch_size $BS --data_dir $DATA_DIR --feature_subset $subset > logs/ablation/${subset}_vanilla.log 2>&1"
    local cmd_fixed="python -u scripts/train_sepsis_benchmarks.py --task ihm --model transformer --epochs $EPOCHS --batch_size $BS --data_dir $DATA_DIR --feature_subset $subset --use_kgi --kgi_alpha 1.0 --kgi_alpha_fixed > logs/ablation/${subset}_fixed.log 2>&1"
    local cmd_adaptive="python -u scripts/train_sepsis_benchmarks.py --task ihm --model transformer --epochs $EPOCHS --batch_size $BS --data_dir $DATA_DIR --feature_subset $subset --use_kgi --kgi_alpha 1.0 > logs/ablation/${subset}_adaptive.log 2>&1"
    
    echo "Launching group $subset on GPU $gpu..."
    tmux new-session -d -s "$session_name" "CUDA_VISIBLE_DEVICES=$gpu bash -c '$cmd_vanilla && $cmd_fixed && $cmd_adaptive'"
}

# Lancio i 4 gruppi in parallelo sulle 4 GPU
run_group 4 full
run_group 5 no_treatments
run_group 6 core
run_group 7 emergency

echo "All ablation groups launched. Each GPU is running 3 variants in sequence."
