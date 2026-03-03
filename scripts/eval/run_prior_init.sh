#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=1
echo "Evaluating SAITS SapBERT (Prior Init)"
python scripts/evaluate_downstream.py --batch_size 32 --model_name saits --data_dir data/processed_sota --checkpoint outputs/saits/random/2026-02-19_16-09-02/checkpoints/best-epoch=99-val/loss=0.3087.ckpt
