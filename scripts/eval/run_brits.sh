#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=2
echo "Evaluating BRITS"
python scripts/evaluate_downstream.py --batch_size 32 --model_name brits --data_dir data/processed_sota --checkpoint outputs/brits/random/2026-02-19_15-22-24/checkpoints/best-epoch=49-val/loss=0.1612.ckpt
