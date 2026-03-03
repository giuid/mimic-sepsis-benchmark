#!/bin/bash
# SSSD Refined Training Script
# Uses GPUs 6,7 with DDP
# Changes: 6 layers, 128 channels, S4 stable init. NO BatchNorm/LayerNorm.

set -e

echo "Starting SSSD Refined Training on GPUs 6,7..."

CUDA_VISIBLE_DEVICES=6,7 python3 train.py \
    model=sssd \
    masking=block \
    trainer.devices=2 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    data.batch_size=256 \
    output_dir=outputs/benchmark/sssd_refined \
    +logging.name=sssd_refined

echo "SSSD Refined Training Complete."
