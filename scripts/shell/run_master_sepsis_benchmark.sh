#!/bin/bash
# scripts/shell/run_master_sepsis_benchmark.sh
# Completes the sepsis benchmark for LOS, SS, VR across all subsets.
# RESTRICTED TO GPU 4 and 5 (GPU 6 and 7 are busy with SAITS Joint)

DATA_DIR="data/processed_sepsis_full"
EPOCHS=30
BS=32

mkdir -p logs/master_benchmark

run_task_list() {
    local gpu=$1
    local task_list=($2)
    local session_name="master_gpu_${gpu}"
    
    local subsets=("full" "no_treatments" "core" "emergency")
    local variants=("vanilla" "deep_dki")
    
    local full_cmd=""
    
    for task in "${task_list[@]}"; do
        for subset in "${subsets[@]}"; do
            for variant in "${variants[@]}"; do
                local log_file="logs/master_benchmark/${task}_${subset}_${variant}.log"
                local cmd="python -u scripts/train_sepsis_benchmarks.py --task $task --model transformer --epochs $EPOCHS --batch_size $BS --data_dir $DATA_DIR --feature_subset $subset"
                
                if [ "$variant" == "deep_dki" ]; then
                    cmd="$cmd --use_kgi"
                fi
                
                if [ -z "$full_cmd" ]; then
                    full_cmd="echo 'Starting ${task} ${subset} ${variant}' && $cmd > $log_file 2>&1"
                else
                    full_cmd="$full_cmd && echo 'Starting ${task} ${subset} ${variant}' && $cmd > $log_file 2>&1"
                fi
            done
        done
    done
    
    echo "Launching tasks on GPU $gpu..."
    tmux new-session -d -s "$session_name" "CUDA_VISIBLE_DEVICES=$gpu bash -c '$full_cmd'"
}

# GPU 4: LOS and half of SS
run_task_list 4 "los"

# GPU 5: SS and VR
run_task_list 5 "ss vr"

echo "All master benchmark tasks launched on GPU 4 and 5. Monitor logs in logs/master_benchmark/"
