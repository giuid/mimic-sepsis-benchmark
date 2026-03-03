#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=4

# SOTA Models
echo "Evaluating Vanilla SAITS"
python scripts/evaluate_downstream.py --batch_size 32 --model_name saits --data_dir data/processed_sota --checkpoint outputs/saits/random/2026-02-19_15-11-55/checkpoints/best-epoch=48-val/loss=0.2752.ckpt

echo "Evaluating SAITS SapBERT"
python scripts/evaluate_downstream.py --batch_size 32 --model_name saits --data_dir data/processed_sota --checkpoint outputs/saits/random/2026-02-19_14-51-00/checkpoints/best-epoch=49-val/loss=0.3108.ckpt

echo "Evaluating SAITS SapBERT (Prior Init)"
python scripts/evaluate_downstream.py --batch_size 32 --model_name saits --data_dir data/processed_sota --checkpoint outputs/saits/random/2026-02-19_16-09-02/checkpoints/best-epoch=99-val/loss=0.3087.ckpt

# BRITS
echo "Evaluating BRITS"
python scripts/evaluate_downstream.py --batch_size 32 --model_name brits --data_dir data/processed_sota --checkpoint outputs/brits/random/2026-02-19_15-22-24/checkpoints/best-epoch=49-val/loss=0.1612.ckpt

echo "All SOTA deep evaluations finished! Results appended to results_downstream.jsonl"
