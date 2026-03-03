#!/bin/bash
# Launch PyPOTS Baselines on SOTA Data (data/processed_sota)
# Using GPUs 4, 5, 6 alongside existing experiments

echo "Starting PyPOTS Baselines on SOTA Data..."

# BRITS on GPU 4
echo "Running: BRITS on GPU 4 (SOTA Data)"
CUDA_VISIBLE_DEVICES=4 python train.py \
    model=brits \
    data=mimic4 \
    data.processed_dir=data/processed_sota \
    data.batch_size=512 \
    trainer.max_epochs=50 \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.precision=16-mixed \
    > logs/brits_sota.log 2>&1 &
disown

# MRNN on GPU 5
echo "Running: MRNN on GPU 5 (SOTA Data)"
CUDA_VISIBLE_DEVICES=5 python train.py \
    model=mrnn \
    data=mimic4 \
    data.processed_dir=data/processed_sota \
    data.batch_size=512 \
    trainer.max_epochs=50 \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.precision=16-mixed \
    > logs/mrnn_sota.log 2>&1 &
disown

# GP-VAE on GPU 6
echo "Running: GP-VAE on GPU 6 (SOTA Data)"
CUDA_VISIBLE_DEVICES=6 python train.py \
    model=gpvae \
    data=mimic4 \
    data.processed_dir=data/processed_sota \
    data.batch_size=512 \
    trainer.max_epochs=50 \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.precision=16-mixed \
    > logs/gpvae_sota.log 2>&1 &
disown

echo "Baseline SOTA experiments launched on GPUs 4-6."
