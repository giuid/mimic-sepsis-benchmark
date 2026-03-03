#!/bin/bash
# SSSD Training Script
# GPU: 7

# Ensure script stops on error
set -e

echo "Starting SSSD Training on GPU 7..."

CUDA_VISIBLE_DEVICES=7 python3 train.py \
    model=sssd \
    masking=block \
    trainer.devices=1 \
    data.batch_size=256 \
    output_dir=outputs/benchmark/sssd \
    +logging.name=sssd_full

echo "SSSD Training Complete."
