#!/bin/bash
# evaluate_sepsis_downstream.sh

DATA_DIR="data/processed_sepsis"

echo "Starting Downstream Evaluation (LGBM & GRU) for Sepsis Task..."

# 1. SAITS KGI
echo "Evaluating SAITS KGI Downstream..."
python scripts/eval/evaluate_downstream.py --task sepsis --model_name saits_kgi --setup handpicked --checkpoint outputs/mimic4_sepsis/saits/random/2026-03-02_15-38-15/checkpoints/vanilla_KGI/best-epoch=39-val/loss=0.7614.ckpt --data_dir $DATA_DIR

# 2. SAITS Vanilla
echo "Evaluating SAITS Vanilla Downstream..."
python scripts/eval/evaluate_downstream.py --task sepsis --model_name saits_vanilla --setup handpicked --checkpoint outputs/mimic4_sepsis/saits/random/2026-03-02_15-40-05/checkpoints/vanilla/best-epoch=01-val/loss=0.8074.ckpt --data_dir $DATA_DIR

# 3. BRITS KGI
echo "Evaluating BRITS KGI Downstream..."
python scripts/eval/evaluate_downstream.py --task sepsis --model_name brits_kgi --setup handpicked --checkpoint outputs/mimic4_sepsis/brits/random/2026-03-02_16-05-30/checkpoints/default_KGI/best-epoch=49-val/loss=15.4726.ckpt --data_dir $DATA_DIR

# 4. BRITS Vanilla
echo "Evaluating BRITS Vanilla Downstream..."
python scripts/eval/evaluate_downstream.py --task sepsis --model_name brits_vanilla --setup handpicked --checkpoint outputs/mimic4_sepsis/brits/random/2026-03-02_16-05-48/checkpoints/default/best-epoch=49-val/loss=15.5532.ckpt --data_dir $DATA_DIR

echo "Downstream evaluation complete. Results in results/mortality_master_benchmark.csv"
