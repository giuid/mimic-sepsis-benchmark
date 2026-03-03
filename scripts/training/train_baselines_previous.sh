#!/bin/bash
# Train PyPOTS baselines on the previous 17-variable dataset (<30% missingness)
# Distributed across GPUs 4-6 with optimized batch size.

# Ensure output directory for logs is clear
mkdir -p logs

echo "Starting PyPOTS Baselines on Previous Data (Distributed)..."

# 1. BRITS
echo "Running Experiment: BRITS on GPU 4"
CUDA_VISIBLE_DEVICES=4 python train.py \
    model=brits \
    data=mimic4 \
    data.processed_dir=data/processed \
    data.batch_size=1024 \
    trainer.max_epochs=50 \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.precision=16-mixed \
    > logs/brits_previous.log 2>&1 &

# 2. MRNN
echo "Running Experiment: MRNN on GPU 5"
CUDA_VISIBLE_DEVICES=5 python train.py \
    model=mrnn \
    data=mimic4 \
    data.processed_dir=data/processed \
    data.batch_size=1024 \
    trainer.max_epochs=50 \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.precision=16-mixed \
    > logs/mrnn_previous.log 2>&1 &

# 3. GP-VAE
echo "Running Experiment: GP-VAE on GPU 6"
CUDA_VISIBLE_DEVICES=6 python train.py \
    model=gpvae \
    data=mimic4 \
    data.processed_dir=data/processed \
    data.batch_size=1024 \
    trainer.max_epochs=50 \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.precision=16-mixed \
    > logs/gpvae_previous.log 2>&1 &

echo "PyPOTS baselines launched on GPUs 4-6."
