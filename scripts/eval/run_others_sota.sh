#!/bin/bash
# Evaluating Remaining SOTA Models

# MRNN
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_downstream.py \
    --model_name mrnn \
    --checkpoint outputs/mrnn/random/2026-02-19_15-22-28/checkpoints/best-epoch=49-val/loss=0.0999.ckpt \
    --data_dir data/processed_sota \
    --device cuda \
    --batch_size 32 > logs_others_mrnn.txt 2>&1 &

# GP-VAE
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_downstream.py \
    --model_name gpvae \
    --checkpoint outputs/gpvae/random/2026-02-19_15-17-02/checkpoints/best-epoch=38-val/loss=39343.2227.ckpt \
    --data_dir data/processed_sota \
    --device cuda \
    --batch_size 32 > logs_others_gpvae.txt 2>&1 &

wait
