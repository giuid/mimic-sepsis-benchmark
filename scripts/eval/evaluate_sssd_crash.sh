#!/bin/bash

# Kill any existing evaluation processes
pkill -f "python3 evaluate.py" || true

# Session Name
SESSION="sssd_eval"

# Check if session exists, kill it if so
tmux kill-session -t $SESSION 2>/dev/null

# Create new session - Window 0: SSSD Eval (GPU 7)
tmux new-session -d -s $SESSION -n "eval_sssd"
tmux send-keys -t $SESSION:eval_sssd "export CUDA_VISIBLE_DEVICES=7" C-m
tmux send-keys -t $SESSION:eval_sssd "python3 evaluate.py \
    --model sssd \
    --checkpoint 'outputs/sssd/random/2026-02-18_12-20-48/checkpoints/best-epoch=49-val/loss=0.1457.ckpt' \
    --masking random \
    --masking_p 0.3 \
    --data_dir data/processed \
    --output_dir results/sssd_crash_eval" C-m

echo "Tmux session '$SESSION' created."
echo "Use 'tmux attach -t $SESSION' to view."
