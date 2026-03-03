#!/bin/bash
# SSSD v3-Stable Training Script
# Uses GPUs 6,7 with DDP
# Strategy: Full Sequence Loss + Strongly Contractive S4
# Optimized for A100: BS=1024, Workers=8, LR=4e-4

set -e

echo "Starting SSSD v3-Stable Accelerated Training on GPUs 6,7..."

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
    output_dir=outputs/benchmark/sssd_v3_stable \
    +logging.name=sssd_v3_stable

echo "SSSD v3-Stable Training Complete."
