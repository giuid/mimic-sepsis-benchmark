#!/bin/bash
# SSSD Vanilla (No-Graph) Ablation Training
# Uses GPUs 6,7 with DDP
# Strategy: Same as v3-Stable but with use_graph_prior=False

set -e

echo "Starting SSSD Vanilla (No-Graph) Training on GPUs 6,7..."

# Clean up any leftover processes
pkill -f "python3 train.py.*sssd" || true
sleep 2

CUDA_VISIBLE_DEVICES=6,7 python3 train.py \
    model=sssd \
    masking=block \
    trainer.devices=2 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    data.batch_size=1024 \
    data.num_workers=8 \
    model.optimizer.lr=0.0004 \
    model.use_graph_prior=false \
    output_dir=outputs/benchmark/sssd_vanilla_ablation \
    +logging.name=sssd_vanilla_ablation

echo "SSSD Vanilla Training Complete."
