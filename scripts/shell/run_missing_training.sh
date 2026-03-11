#!/bin/bash
# =============================================================================
# Missing Training Runs — 2026-03-06
#
# BLOCK 1: Transformer 96h Vanilla — 4 subsets × 4 tasks = 16 runs
# BLOCK 2: SAITS Joint + DKI — 4 subsets × 4 tasks = 16 runs
#
# GPU rotation: jobs chained sequentially per GPU window (4,5,6,7)
# =============================================================================

SESSION="missing_runs"
tmux kill-session -t $SESSION 2>/dev/null || true
tmux new-session -d -s $SESSION -n "gpu4"
tmux new-window -t $SESSION -n "gpu5"
tmux new-window -t $SESSION -n "gpu6"
tmux new-window -t $SESSION -n "gpu7"

CD="cd /home/guido/Code/charite/baselines"

# --- GPU 4: Transformer 96h [ihm/full, ihm/no_treatments, ihm/core, ihm/emergency] + SAITS DKI [ihm/full, ihm/no_treatments, ihr/core, ihm/emergency] ---
tmux send-keys -t $SESSION:gpu4 "$CD" C-m
tmux send-keys -t $SESSION:gpu4 "CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset full          --data_dir data/processed_sepsis_full --epochs 30 --batch_size 64 && \
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset no_treatments --data_dir data/processed_sepsis_full --epochs 30 --batch_size 64 && \
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset core         --data_dir data/processed_sepsis_full --epochs 30 --batch_size 64 && \
CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer --feature_subset emergency    --data_dir data/processed_sepsis_full --epochs 30 --batch_size 64 && \
CUDA_VISIBLE_DEVICES=4 python train.py model=joint model.task=ihm model.imputator_kwargs.use_kgi=true data.processed_dir=data/processed_sepsis_full data.feature_subset=full          trainer.max_epochs=30 && \
CUDA_VISIBLE_DEVICES=4 python train.py model=joint model.task=ihm model.imputator_kwargs.use_kgi=true data.processed_dir=data/processed_sepsis_full data.feature_subset=no_treatments trainer.max_epochs=30 && \
CUDA_VISIBLE_DEVICES=4 python train.py model=joint model.task=ihm model.imputator_kwargs.use_kgi=true data.processed_dir=data/processed_sepsis_full data.feature_subset=core          trainer.max_epochs=30 && \
CUDA_VISIBLE_DEVICES=4 python train.py model=joint model.task=ihm model.imputator_kwargs.use_kgi=true data.processed_dir=data/processed_sepsis_full data.feature_subset=emergency    trainer.max_epochs=30 && \
echo 'GPU4 DONE'" C-m

# --- GPU 5: Transformer 96h [los x4] + SAITS DKI [los x4] ---
tmux send-keys -t $SESSION:gpu5 "$CD" C-m
tmux send-keys -t $SESSION:gpu5 "CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset full          --data_dir data/processed_sepsis_full --epochs 30 --batch_size 64 && \
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset no_treatments --data_dir data/processed_sepsis_full --epochs 30 --batch_size 64 && \
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset core         --data_dir data/processed_sepsis_full --epochs 30 --batch_size 64 && \
CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --task los --model transformer --feature_subset emergency    --data_dir data/processed_sepsis_full --epochs 30 --batch_size 64 && \
CUDA_VISIBLE_DEVICES=5 python train.py model=joint model.task=los model.imputator_kwargs.use_kgi=true data.processed_dir=data/processed_sepsis_full data.feature_subset=full          trainer.max_epochs=30 && \
CUDA_VISIBLE_DEVICES=5 python train.py model=joint model.task=los model.imputator_kwargs.use_kgi=true data.processed_dir=data/processed_sepsis_full data.feature_subset=no_treatments trainer.max_epochs=30 && \
CUDA_VISIBLE_DEVICES=5 python train.py model=joint model.task=los model.imputator_kwargs.use_kgi=true data.processed_dir=data/processed_sepsis_full data.feature_subset=core          trainer.max_epochs=30 && \
CUDA_VISIBLE_DEVICES=5 python train.py model=joint model.task=los model.imputator_kwargs.use_kgi=true data.processed_dir=data/processed_sepsis_full data.feature_subset=emergency    trainer.max_epochs=30 && \
echo 'GPU5 DONE'" C-m

# --- GPU 6: Transformer 96h [vr x4] + SAITS DKI [vr x4] ---
tmux send-keys -t $SESSION:gpu6 "$CD" C-m
tmux send-keys -t $SESSION:gpu6 "CUDA_VISIBLE_DEVICES=6 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset full          --data_dir data/processed_sepsis_full --epochs 30 --batch_size 64 && \
CUDA_VISIBLE_DEVICES=6 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset no_treatments --data_dir data/processed_sepsis_full --epochs 30 --batch_size 64 && \
CUDA_VISIBLE_DEVICES=6 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset core         --data_dir data/processed_sepsis_full --epochs 30 --batch_size 64 && \
CUDA_VISIBLE_DEVICES=6 python scripts/train_sepsis_benchmarks.py --task vr --model transformer --feature_subset emergency    --data_dir data/processed_sepsis_full --epochs 30 --batch_size 64 && \
CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=vr model.imputator_kwargs.use_kgi=true data.processed_dir=data/processed_sepsis_full data.feature_subset=full          trainer.max_epochs=30 && \
CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=vr model.imputator_kwargs.use_kgi=true data.processed_dir=data/processed_sepsis_full data.feature_subset=no_treatments trainer.max_epochs=30 && \
CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=vr model.imputator_kwargs.use_kgi=true data.processed_dir=data/processed_sepsis_full data.feature_subset=core          trainer.max_epochs=30 && \
CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=vr model.imputator_kwargs.use_kgi=true data.processed_dir=data/processed_sepsis_full data.feature_subset=emergency    trainer.max_epochs=30 && \
echo 'GPU6 DONE'" C-m

# --- GPU 7: Transformer 96h [ss x4] + SAITS DKI [ss x4] ---
tmux send-keys -t $SESSION:gpu7 "$CD" C-m
tmux send-keys -t $SESSION:gpu7 "CUDA_VISIBLE_DEVICES=7 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset full          --data_dir data/processed_sepsis_full --epochs 30 --batch_size 64 && \
CUDA_VISIBLE_DEVICES=7 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset no_treatments --data_dir data/processed_sepsis_full --epochs 30 --batch_size 64 && \
CUDA_VISIBLE_DEVICES=7 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset core         --data_dir data/processed_sepsis_full --epochs 30 --batch_size 64 && \
CUDA_VISIBLE_DEVICES=7 python scripts/train_sepsis_benchmarks.py --task ss --model transformer --feature_subset emergency    --data_dir data/processed_sepsis_full --epochs 30 --batch_size 64 && \
CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=ss model.imputator_kwargs.use_kgi=true data.processed_dir=data/processed_sepsis_full data.feature_subset=full          trainer.max_epochs=30 && \
CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=ss model.imputator_kwargs.use_kgi=true data.processed_dir=data/processed_sepsis_full data.feature_subset=no_treatments trainer.max_epochs=30 && \
CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=ss model.imputator_kwargs.use_kgi=true data.processed_dir=data/processed_sepsis_full data.feature_subset=core          trainer.max_epochs=30 && \
CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=ss model.imputator_kwargs.use_kgi=true data.processed_dir=data/processed_sepsis_full data.feature_subset=emergency    trainer.max_epochs=30 && \
echo 'GPU7 DONE'" C-m

echo ""
echo "=== 32 jobs dispatched (16 Transformer 96h + 16 SAITS Joint DKI) ==="
echo "  GPU4: IHM  (Transformer 96h ×4 + SAITS DKI ×4)"
echo "  GPU5: LOS  (Transformer 96h ×4 + SAITS DKI ×4)"
echo "  GPU6: VR   (Transformer 96h ×4 + SAITS DKI ×4)"
echo "  GPU7: SS   (Transformer 96h ×4 + SAITS DKI ×4)"
echo ""
echo "Monitor with: tmux attach -t $SESSION"
