#!/bin/bash
# evaluate_sepsis_best.sh

DATA_DIR="data/processed_sepsis"
MASK_P=0.3

echo "Starting Sepsis Benchmark Evaluation..."

# 1. SAITS KGI
echo "Evaluating SAITS KGI..."
python evaluate.py --model saits --checkpoint outputs/mimic4_sepsis/saits/random/2026-03-02_15-38-15/checkpoints/vanilla_KGI/best-epoch=39-val/loss=0.7614.ckpt --data_dir $DATA_DIR --masking random --masking_p $MASK_P

# 2. SAITS Vanilla
echo "Evaluating SAITS Vanilla..."
python evaluate.py --model saits --checkpoint outputs/mimic4_sepsis/saits/random/2026-03-02_15-40-05/checkpoints/vanilla/best-epoch=01-val/loss=0.8074.ckpt --data_dir $DATA_DIR --masking random --masking_p $MASK_P

# 3. MRNN KGI
echo "Evaluating MRNN KGI..."
python evaluate.py --model mrnn --checkpoint outputs/mimic4_sepsis/mrnn/random/2026-03-02_16-08-43/checkpoints/default_KGI/best-epoch=40-val/loss=0.1485.ckpt --data_dir $DATA_DIR --masking random --masking_p $MASK_P

# 4. MRNN Vanilla
echo "Evaluating MRNN Vanilla..."
python evaluate.py --model mrnn --checkpoint outputs/mimic4_sepsis/mrnn/random/2026-03-02_16-09-18/checkpoints/default/best-epoch=40-val/loss=0.1485.ckpt --data_dir $DATA_DIR --masking random --masking_p $MASK_P

# 5. BRITS KGI
echo "Evaluating BRITS KGI..."
python evaluate.py --model brits --checkpoint outputs/mimic4_sepsis/brits/random/2026-03-02_16-05-30/checkpoints/default_KGI/best-epoch=49-val/loss=15.4726.ckpt --data_dir $DATA_DIR --masking random --masking_p $MASK_P

# 6. BRITS Vanilla
echo "Evaluating BRITS Vanilla..."
python evaluate.py --model brits --checkpoint outputs/mimic4_sepsis/brits/random/2026-03-02_16-05-48/checkpoints/default/best-epoch=49-val/loss=15.5532.ckpt --data_dir $DATA_DIR --masking random --masking_p $MASK_P

echo "All evaluations complete. Results saved in results/master_benchmark.csv"
