#!/bin/bash
# SSSD Refined Training Script (v2 Accelerated)
# Uses GPUs 6,7 with DDP
# Optimized for A100 (40GB): BS=1024, Workers=8, LR=4e-4

set -e

echo "Starting SSSD Refined v2 Accelerated Training on GPUs 6,7..."

CUDA_VISIBLE_DEVICES=6,7 python3 train.py \
    model=sssd \
    masking=block \
    trainer.devices=2 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    data.batch_size=1024 \
    data.num_workers=8 \
    model.optimizer.lr=0.0004 \
    output_dir=outputs/benchmark/sssd_refined_v2 \
    +logging.name=sssd_refined_v2

echo "SSSD Refined v2 Training Complete."
