#!/bin/bash
# SSSD Stable Training Script
# Uses GPUs 6,7 with DDP for maximum utilization
# Changes from original: 6 layers, 128 channels, BatchNorm + LayerNorm

set -e

echo "Starting SSSD Stable Training on GPUs 6,7..."

CUDA_VISIBLE_DEVICES=6,7 python3 train.py \
    model=sssd \
    masking=block \
    trainer.devices=2 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    data.batch_size=256 \
    output_dir=outputs/benchmark/sssd_stable \
    +logging.name=sssd_stable

echo "SSSD Stable Training Complete."

