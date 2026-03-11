#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export OMP_NUM_THREADS=4

device="cuda:0" # because of CUDA_VISIBLE_DEVICES=7

echo "Running mean on sota"
python scripts/evaluate_downstream.py --model_name mean --setup sota --data_dir data/sota --device $device --batch_size 256 || echo 'Failed mean on sota'

echo "Running locf on sota"
python scripts/evaluate_downstream.py --model_name locf --setup sota --data_dir data/sota --device $device --batch_size 256 || echo 'Failed locf on sota'

echo "Running linear_interp on sota"
python scripts/evaluate_downstream.py --model_name linear_interp --setup sota --data_dir data/sota --device $device --batch_size 256 || echo 'Failed linear_interp on sota'

echo "[sota][saits] Checkpoint NOT FOUND!"

echo "[sota][brits] Checkpoint NOT FOUND!"

echo "[sota][gpvae] Checkpoint NOT FOUND!"

echo "[sota][mrnn] Checkpoint NOT FOUND!"

echo "[sota][sssd] Checkpoint NOT FOUND!"

echo "[sota][timesfm] Checkpoint NOT FOUND!"

echo "[sota][timesfm_sapbert] Checkpoint NOT FOUND!"

echo "Running mean on handpicked"
python scripts/evaluate_downstream.py --model_name mean --setup handpicked --data_dir data/handpicked --device $device --batch_size 256 || echo 'Failed mean on handpicked'

echo "Running locf on handpicked"
python scripts/evaluate_downstream.py --model_name locf --setup handpicked --data_dir data/handpicked --device $device --batch_size 256 || echo 'Failed locf on handpicked'

echo "Running linear_interp on handpicked"
python scripts/evaluate_downstream.py --model_name linear_interp --setup handpicked --data_dir data/handpicked --device $device --batch_size 256 || echo 'Failed linear_interp on handpicked'

echo "[handpicked][saits] Checkpoint NOT FOUND!"

echo "[handpicked][brits] Checkpoint NOT FOUND!"

echo "[handpicked][gpvae] Checkpoint NOT FOUND!"

echo "[handpicked][mrnn] Checkpoint NOT FOUND!"

echo "[handpicked][sssd] Checkpoint NOT FOUND!"

echo "[handpicked][timesfm] Checkpoint NOT FOUND!"

echo "[handpicked][timesfm_sapbert] Checkpoint NOT FOUND!"

echo "All evaluations complete!"
