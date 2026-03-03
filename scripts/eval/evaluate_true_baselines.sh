#!/bin/bash

# Kill any existing evaluation processes
pkill -f "python3 evaluate.py" || true

# Session Name
SESSION="saits_true_eval"

# Check if session exists, kill it if so
tmux kill-session -t $SESSION 2>/dev/null

# Window 0: True Vanilla
tmux new-session -d -s $SESSION -n "eval_vanilla"
tmux send-keys -t $SESSION:eval_vanilla "export CUDA_VISIBLE_DEVICES=4" C-m
tmux send-keys -t $SESSION:eval_vanilla "python3 evaluate.py \
    --model saits \
    --checkpoint 'outputs/saits/random/2026-02-18_15-40-17/checkpoints/best-epoch=95-val/loss=0.3362.ckpt' \
    --masking random \
    --masking_p 0.3 \
    --data_dir data/processed \
    --output_dir results/true_vanilla_eval" C-m

# Window 1: True Prior Nullo
tmux new-window -t $SESSION -n "eval_nullo"
tmux send-keys -t $SESSION:eval_nullo "export CUDA_VISIBLE_DEVICES=5" C-m
# Note: Using the checkpoint path from the log output for Prior Nullo
tmux send-keys -t $SESSION:eval_nullo "python3 evaluate.py \
    --model saits \
    --checkpoint 'outputs/saits/random/2026-02-18_15-40-17/checkpoints/best-epoch=97-val/loss=0.3233.ckpt' \
    --masking random \
    --masking_p 0.3 \
    --data_dir data/processed \
    --output_dir results/true_nullo_eval" C-m

echo "Tmux session '$SESSION' created."
echo "Use 'tmux attach -t $SESSION' to view."
