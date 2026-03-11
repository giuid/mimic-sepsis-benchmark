#!/bin/bash
# run_sepsis_transformer.sh
# Dispatches original Sepsis benchmark Transformer on downstream tasks across GPUs

SESSION="sepsis_transformer"
tmux new-session -d -s $SESSION

# Only testing the Transformer baseline
MODELS=("transformer")
# 4 Benchmark tasks
TASKS=("ihm" "los" "vr" "ss")

# Available GPUs
GPUs=(4 5 6 7)
gpu_idx=0

# Create windows and dispatch jobs
for TASK in "${TASKS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for kgi_flag in "" "--use_kgi"; do
            
            # Select GPU
            GPU=${GPUs[$gpu_idx]}
            
            # Window Name
            kgi_suffix="van"
            if [ "$kgi_flag" == "--use_kgi" ]; then
                kgi_suffix="kgi"
            fi
            WNAME="${TASK}_${MODEL}_${kgi_suffix}"
            
            tmux new-window -t $SESSION -n $WNAME
            
            CMD="CUDA_VISIBLE_DEVICES=$GPU python scripts/train_sepsis_benchmarks.py --model $MODEL --task $TASK --epochs 100 $kgi_flag"
            echo "Dispatching on physical GPU $GPU: $CMD"
            
            tmux send-keys -t $SESSION:$WNAME "$CMD" C-m
            
            # Rotate GPU index
            gpu_idx=$(( (gpu_idx + 1) % ${#GPUs[@]} ))
            
            sleep 1
        done
    done
done

# Kill the default empty window
tmux kill-window -t $SESSION:0

echo "All 8 jobs dispatched (4 tasks * 1 model * 2 variants). Use 'tmux attach -t sepsis_transformer' to monitor."
