#!/bin/bash

# Kill any existing training processes to free GPUs
pkill -f "python3 train.py" || true

# Session Name
SESSION="benchmark"

# Check if session exists, kill it if so
tmux kill-session -t $SESSION 2>/dev/null

# Create new session - Window 0: Prior Nullo (GPU 4)
tmux new-session -d -s $SESSION -n "prior_nullo"
tmux send-keys -t $SESSION:prior_nullo "export CUDA_VISIBLE_DEVICES=4" C-m
tmux send-keys -t $SESSION:prior_nullo "python3 train.py \
    model=saits \
    +model.use_graph_layer=true \
    +model.graph_type=nullo \
    data.batch_size=256 \
    trainer.max_epochs=100 \
    hydra.run.dir=outputs/benchmark/prior_nullo_enriched" C-m

# Window 1: SapBERT + CI-GNN (GPU 5)
tmux new-window -t $SESSION -n "sapbert_cignn"
tmux send-keys -t $SESSION:sapbert_cignn "export CUDA_VISIBLE_DEVICES=5" C-m
tmux send-keys -t $SESSION:sapbert_cignn "python3 train.py \
    model=saits \
    +model.use_graph_layer=true \
    +model.graph_type=ci_gnn \
    data.batch_size=256 \
    trainer.max_epochs=100 \
    hydra.run.dir=outputs/benchmark/sapbert_cignn_enriched" C-m

# Window 2: Vanilla SAITS (GPU 6)
tmux new-window -t $SESSION -n "vanilla_saits"
tmux send-keys -t $SESSION:vanilla_saits "export CUDA_VISIBLE_DEVICES=6" C-m
tmux send-keys -t $SESSION:vanilla_saits "python3 train.py \
    model=saits \
    data.batch_size=256 \
    trainer.max_epochs=100 \
    hydra.run.dir=outputs/benchmark/vanilla_saits_enriched" C-m

# Window 3: SSSD (GPU 7)
tmux new-window -t $SESSION -n "sssd"
tmux send-keys -t $SESSION:sssd "export CUDA_VISIBLE_DEVICES=7" C-m
tmux send-keys -t $SESSION:sssd "python3 train.py \
    model=sssd \
    data.batch_size=32 \
    trainer.max_epochs=100 \
    trainer.gradient_clip_val=1.0 \
    hydra.run.dir=outputs/benchmark/sssd_enriched" C-m

echo "Tmux session '$SESSION' created with 4 training windows."
echo "Use 'tmux attach -t $SESSION' to view."
