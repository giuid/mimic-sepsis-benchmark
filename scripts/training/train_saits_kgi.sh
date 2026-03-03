#!/bin/bash
# KGI Imputation SAITS training on MIMIC-IV 3.1 
# GPUs 4-7 exclusively

# Output directory for logs
mkdir -p logs

echo "Starting SAITS KGI (Knowledge Guided Imputation) on SOTA Data..."

# 1. KGI SAITS (Using SapBERT nodes + MedBERT triplets)
echo "Running: SAITS - KGI on GPU 6"
CUDA_VISIBLE_DEVICES=6 python train.py \
    model=saits \
    model.embedding_type=vanilla \
    +model.use_kgi=True \
    data=mimic4 \
    data.processed_dir=data/sota \
    data.batch_size=1024 \
    trainer.max_epochs=50 \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.precision=16-mixed \
    > logs/saits_kgi_sota.log 2>&1 &

# We also launch the Vanilla SAITS on GPU 7 to have a direct comparison on the same run
echo "Running: SAITS - Vanilla comparison on GPU 7"
CUDA_VISIBLE_DEVICES=7 python train.py \
    model=saits \
    model.embedding_type=vanilla \
    +model.use_kgi=False \
    data=mimic4 \
    data.processed_dir=data/sota \
    data.batch_size=1024 \
    trainer.max_epochs=50 \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.precision=16-mixed \
    > logs/saits_vanilla_baseline_sota.log 2>&1 &

echo "KGI experiment launched on GPU 4. Vanilla baseline running on GPU 5."
echo "Monitor logs/saits_kgi_sota.log"
