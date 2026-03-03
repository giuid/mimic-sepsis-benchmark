#!/bin/bash
# Evaluating Remaining PREV Models

# MRNN Prev
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_downstream.py \
    --model_name mrnn \
    --checkpoint outputs/mrnn/random/2026-02-19_15-22-26/checkpoints/best-epoch=49-val/loss=0.1066.ckpt \
    --data_dir data/processed \
    --device cuda \
    --batch_size 32 > logs_prev_mrnn.txt 2>&1 &

# GP-VAE Prev
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_downstream.py \
    --model_name gpvae \
    --checkpoint outputs/gpvae/random/2026-02-19_15-17-00/checkpoints/best-epoch=37-val/loss=39336.9219.ckpt \
    --data_dir data/processed \
    --device cuda \
    --batch_size 32 > logs_prev_gpvae.txt 2>&1 &

# SSSD Prev (Stable v3)
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_downstream.py \
    --model_name sssd \
    --checkpoint outputs/benchmark/sssd_v3_stable/checkpoints/best-epoch=72-val/loss=0.2455.ckpt \
    --data_dir data/processed \
    --device cuda \
    --batch_size 16 > logs_prev_sssd.txt 2>&1 &

wait
