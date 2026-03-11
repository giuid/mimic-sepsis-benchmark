#!/bin/bash

# Dispatches all 4 Sepsis predictive tasks for LSTM and Transformer variants.
# This uses all features (treatment inclusive) to reproduce the paper metrics.

# GPUs 4-7
# We have 4 tasks (ihm, los, vr, ss) * 3 models (lstm, transformer, transformer_kgi) = 12 runs.
# Distribute across 4 GPUs (3 jobs per GPU).

SESSION="sepsis_baselines"
tmux kill-session -t $SESSION 2>/dev/null
tmux new-session -d -s $SESSION
tmux set-option -t $SESSION remain-on-exit on

# GPU 4: IHM
tmux rename-window -t $SESSION:0 "ihm_runs"
tmux send-keys -t $SESSION:0 "CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --model lstm --task ihm --epochs 100" C-m
tmux split-window -v -t $SESSION:0
tmux send-keys -t $SESSION:0.1 "CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --model transformer --task ihm --epochs 100" C-m
tmux split-window -v -t $SESSION:0
tmux send-keys -t $SESSION:0.2 "CUDA_VISIBLE_DEVICES=4 python scripts/train_sepsis_benchmarks.py --model transformer --task ihm --epochs 100 --use_kgi" C-m
tmux select-layout -t $SESSION:0 even-vertical

# GPU 5: LOS
tmux new-window -t $SESSION -n "los_runs"
tmux send-keys -t $SESSION:1 "CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --model lstm --task los --epochs 100" C-m
tmux split-window -v -t $SESSION:1
tmux send-keys -t $SESSION:1.1 "CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --model transformer --task los --epochs 100" C-m
tmux split-window -v -t $SESSION:1
tmux send-keys -t $SESSION:1.2 "CUDA_VISIBLE_DEVICES=5 python scripts/train_sepsis_benchmarks.py --model transformer --task los --epochs 100 --use_kgi" C-m
tmux select-layout -t $SESSION:1 even-vertical

# GPU 6: VR
tmux new-window -t $SESSION -n "vr_runs"
tmux send-keys -t $SESSION:2 "CUDA_VISIBLE_DEVICES=6 python scripts/train_sepsis_benchmarks.py --model lstm --task vr --epochs 100" C-m
tmux split-window -v -t $SESSION:2
tmux send-keys -t $SESSION:2.1 "CUDA_VISIBLE_DEVICES=6 python scripts/train_sepsis_benchmarks.py --model transformer --task vr --epochs 100" C-m
tmux split-window -v -t $SESSION:2
tmux send-keys -t $SESSION:2.2 "CUDA_VISIBLE_DEVICES=6 python scripts/train_sepsis_benchmarks.py --model transformer --task vr --epochs 100 --use_kgi" C-m
tmux select-layout -t $SESSION:2 even-vertical

# GPU 7: SS
tmux new-window -t $SESSION -n "ss_runs"
tmux send-keys -t $SESSION:3 "CUDA_VISIBLE_DEVICES=7 python scripts/train_sepsis_benchmarks.py --model lstm --task ss --epochs 100" C-m
tmux split-window -v -t $SESSION:3
tmux send-keys -t $SESSION:3.1 "CUDA_VISIBLE_DEVICES=7 python scripts/train_sepsis_benchmarks.py --model transformer --task ss --epochs 100" C-m
tmux split-window -v -t $SESSION:3
tmux send-keys -t $SESSION:3.2 "CUDA_VISIBLE_DEVICES=7 python scripts/train_sepsis_benchmarks.py --model transformer --task ss --epochs 100 --use_kgi" C-m
tmux select-layout -t $SESSION:3 even-vertical

echo "Dispatched 12 baseline runs to tmux session: sepsis_baselines"
