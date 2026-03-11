#!/bin/bash
# scripts/shell/relaunch_dki_ablation.sh

mkdir -p logs/ablation_joint

SESSION="saits_ablation_dki"
tmux new-session -d -s $SESSION

TASKS=("ihm" "los" "vr" "ss")
SUBSETS=("full" "core" "emergency" "no_treatments")
GPUs=(4 5 6 7)
gpu_idx=0

for TASK in "${TASKS[@]}"; do
    for SUBSET in "${SUBSETS[@]}"; do
        GPU=${GPUs[$gpu_idx]}
        WNAME="dki_${TASK}_${SUBSET}"
        
        tmux new-window -t $SESSION -n $WNAME
        
        CMD="CUDA_VISIBLE_DEVICES=$GPU python -u scripts/train_sepsis_benchmarks.py --task $TASK --model transformer --epochs 30 --batch_size 32 --data_dir data/processed_sepsis_full --feature_subset $SUBSET --use_kgi --kgi_alpha 1.0 > logs/ablation_joint/${TASK}_${SUBSET}_dki.log 2>&1"
        
        echo "Dispatching $TASK $SUBSET on GPU $GPU"
        tmux send-keys -t $SESSION:$WNAME "$CMD" C-m
        
        # Rotate GPU index
        gpu_idx=$(( (gpu_idx + 1) % ${#GPUs[@]} ))
        sleep 1
    done
done

tmux kill-window -t $SESSION:0
echo "All 16 SAITS DKI ablation jobs dispatched. Use 'tmux attach -t $SESSION' to monitor."
