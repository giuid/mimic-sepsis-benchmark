#!/bin/bash
# run_sepsis_pypots.sh
# Dispatches PyPOTS models on MIMIC-Sepsis downstream tasks across multiple GPUs

SESSION="sepsis_pypots"
tmux new-session -d -s $SESSION

# Models to evaluate
MODELS=("saits" "mrnn" "brits")
# 4 Benchmark tasks
TASKS=("ihm" "los" "vr" "ss")

# Available GPUs
GPUs=(4 5 6 7)
gpu_idx=0

# Create windows and dispatch jobs
for TASK in "${TASKS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for kgi_flag in "" "--kgi"; do
            
            # Select GPU
            GPU=${GPUs[$gpu_idx]}
            
            # Window Name
            kgi_suffix="van"
            if [ "$kgi_flag" == "--kgi" ]; then
                kgi_suffix="kgi"
            fi
            WNAME="${TASK}_${MODEL}_${kgi_suffix}"
            
            tmux new-window -t $SESSION -n $WNAME
            
            CMD="CUDA_VISIBLE_DEVICES=$GPU python scripts/train_sepsis_pypots.py --model $MODEL --task $TASK --gpu 0 $kgi_flag --epochs 10"
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

echo "All 24 jobs dispatched (4 tasks * 3 models * 2 variants). Use 'tmux attach -t sepsis_pypots' to monitor."
