#!/bin/bash
# Run experiments for PyPOTS baselines

# 1. BRITS
echo "Running Experiment: BRITS"
python train.py \
    model=brits \
    config=configs/data/mimic4.yaml \
    trainer.max_epochs=50

# 2. MRNN
echo "Running Experiment: MRNN"
python train.py \
    model=mrnn \
    config=configs/data/mimic4.yaml \
    trainer.max_epochs=50

# 3. GP-VAE
echo "Running Experiment: GP-VAE"
python train.py \
    model=gpvae \
    config=configs/data/mimic4.yaml \
    trainer.max_epochs=50
