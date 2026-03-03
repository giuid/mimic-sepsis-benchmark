#!/bin/bash
set -e

# Vanilla SAITS previous
CUDA_VISIBLE_DEVICES=5 python scripts/evaluate_downstream.py --batch_size 32 --model_name saits --data_dir data/processed --checkpoint outputs/saits/random/2026-02-18_15-40-17/checkpoints/best-epoch=97-val/loss=0.3233.ckpt > logs_prev_vanilla.txt 2>&1 &
echo $! > pid_prev_vanilla.txt

# SapBERT SAITS previous
CUDA_VISIBLE_DEVICES=6 python scripts/evaluate_downstream.py --batch_size 32 --model_name saits --data_dir data/processed --checkpoint outputs/saits/random/2026-02-18_14-16-23/checkpoints/best-epoch=96-val/loss=0.3244.ckpt > logs_prev_sapbert.txt 2>&1 &
echo $! > pid_prev_sapbert.txt

# SapBERT Prior Init SAITS previous
CUDA_VISIBLE_DEVICES=7 python scripts/evaluate_downstream.py --batch_size 32 --model_name saits --data_dir data/processed --checkpoint outputs/saits/random/2026-02-18_14-16-22/checkpoints/best-epoch=93-val/loss=0.3246.ckpt > logs_prev_prior_init.txt 2>&1 &
echo $! > pid_prev_prior_init.txt
