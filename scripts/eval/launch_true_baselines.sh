#!/bin/bash

# Kill any existing training processes to free GPUs
pkill -f "python3 train.py" || true

# Session Name
SESSION="true_baselines"

# Check if session exists, kill it if so
tmux kill-session -t $SESSION 2>/dev/null

# 1. True Vanilla SAITS (No Graph Layer at all)
# GPU 4
tmux new-session -d -s $SESSION -n "true_vanilla"
tmux send-keys -t $SESSION:true_vanilla "export CUDA_VISIBLE_DEVICES=4" C-m
tmux send-keys -t $SESSION:true_vanilla "python3 train.py \
    model=saits \
    +model.use_graph_layer=false \
    data.batch_size=256 \
    trainer.max_epochs=100 \
    hydra.run.dir=outputs/benchmark/true_vanilla_saits" C-m

# 2. True Prior Nullo (Parallel Graph Layer, but Random Init and Zero Loss)
# GPU 5
tmux new-window -t $SESSION -n "true_prior_nullo"
tmux send-keys -t $SESSION:true_prior_nullo "export CUDA_VISIBLE_DEVICES=5" C-m
tmux send-keys -t $SESSION:true_prior_nullo "python3 train.py \
    model=saits \
    +model.use_graph_layer=true \
    +model.parallel_attention=true \
    model.use_prior_init=false \
    model.graph_loss_weight=0.0 \
    data.batch_size=256 \
    trainer.max_epochs=100 \
    hydra.run.dir=outputs/benchmark/true_prior_nullo_parallel" C-m

echo "Tmux session '$SESSION' created with 2 training windows."
echo "Use 'tmux attach -t $SESSION' to view."
