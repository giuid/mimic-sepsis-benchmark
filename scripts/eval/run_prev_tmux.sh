#!/bin/bash
set -e

echo "Starting downstream evaluation for previously selected models (11 features)"

# Vanilla SAITS previous -> GPU 1
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_downstream.py --batch_size 32 --model_name saits --data_dir data/processed --checkpoint outputs/saits/random/2026-02-18_15-40-17/checkpoints/best-epoch=97-val/loss=0.3233.ckpt > logs_prev_vanilla.txt 2>&1 &
PID1=$!

# SapBERT SAITS previous -> GPU 2
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_downstream.py --batch_size 32 --model_name saits --data_dir data/processed --checkpoint outputs/saits/random/2026-02-18_14-16-23/checkpoints/best-epoch=96-val/loss=0.3244.ckpt > logs_prev_sapbert.txt 2>&1 &
PID2=$!

# SapBERT Prior Init SAITS previous -> GPU 3
CUDA_VISIBLE_DEVICES=3 python scripts/evaluate_downstream.py --batch_size 32 --model_name saits --data_dir data/processed --checkpoint outputs/saits/random/2026-02-18_14-16-22/checkpoints/best-epoch=93-val/loss=0.3246.ckpt > logs_prev_prior_init.txt 2>&1 &
PID3=$!

echo "Evaluations started on GPUs 1, 2, and 3."
echo "Waiting for all processes to finish..."

wait $PID1
wait $PID2
wait $PID3

echo "All evaluations finished! Results appended to results_downstream.jsonl"
