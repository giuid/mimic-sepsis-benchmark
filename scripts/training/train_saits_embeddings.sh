#!/bin/bash
# Compare SAITS embedding initialization strategies on SOTA Data (D=17, T=48)
# Distributed across GPUs 0-3 with optimized batch size.

# Output directory for logs
mkdir -p logs

echo "Starting SAITS Embedding Experiments on SOTA Data (Distributed)..."

# 1. Vanilla SAITS (Default)
echo "Running: SAITS - Vanilla on GPU 0"
CUDA_VISIBLE_DEVICES=0 python train.py \
    model=saits \
    model.embedding_type=vanilla \
    data=mimic4 \
    data.processed_dir=data/processed_sota \
    data.batch_size=1024 \
    trainer.max_epochs=50 \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.precision=16-mixed \
    > logs/saits_vanilla_sota.log 2>&1 &

# 2. SapBERT Embeddings
echo "Running: SAITS - SapBERT on GPU 1"
CUDA_VISIBLE_DEVICES=1 python train.py \
    model=saits \
    model.embedding_type=sapbert \
    data=mimic4 \
    data.processed_dir=data/processed_sota \
    data.batch_size=1024 \
    trainer.max_epochs=50 \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.precision=16-mixed \
    > logs/saits_sapbert_sota.log 2>&1 &

# 3. Generic Embeddings (BERT)
echo "Running: SAITS - Generic (BERT) on GPU 2"
CUDA_VISIBLE_DEVICES=2 python train.py \
    model=saits \
    model.embedding_type=bert \
    data=mimic4 \
    data.processed_dir=data/processed_sota \
    data.batch_size=1024 \
    trainer.max_epochs=50 \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.precision=16-mixed \
    > logs/saits_generic_sota.log 2>&1 &

# 4. Random Embeddings
echo "Running: SAITS - Random on GPU 3"
CUDA_VISIBLE_DEVICES=3 python train.py \
    model=saits \
    model.embedding_type=random \
    data=mimic4 \
    data.processed_dir=data/processed_sota \
    data.batch_size=1024 \
    trainer.max_epochs=50 \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.precision=16-mixed \
    > logs/saits_random_sota.log 2>&1 &

echo "SAITS experiments launched on GPUs 0-3."
