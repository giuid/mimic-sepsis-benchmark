#!/bin/bash

# Launch script for TimesFM-KGI joint training experiments

echo "Starting TimesFM-KGI joint training jobs on GPUs 4-7..."

# 1. TimesFM Vanilla (for baseline in joint setting)
tmux new-session -d -s joint_timesfm_vanilla_a01 "CUDA_VISIBLE_DEVICES=4 python scripts/training/train_joint.py model=timesfm ++model.alpha_joint=0.1 ++trainer.max_epochs=40 ++data.processed_dir=data/sota ++data.batch_size=128; read"

# 2. TimesFM KGI (alpha 0.1)
tmux new-session -d -s joint_timesfm_kgi_a01 "CUDA_VISIBLE_DEVICES=5 python scripts/training/train_joint.py model=timesfm ++model.use_kgi=true ++model.alpha_joint=0.1 ++trainer.max_epochs=40 ++data.processed_dir=data/sota ++data.batch_size=128; read"

# 3. TimesFM KGI (alpha 0.01)
tmux new-session -d -s joint_timesfm_kgi_a001 "CUDA_VISIBLE_DEVICES=6 python scripts/training/train_joint.py model=timesfm ++model.use_kgi=true ++model.alpha_joint=0.01 ++trainer.max_epochs=40 ++data.processed_dir=data/sota ++data.batch_size=128; read"

# 4. TimesFM KGI (alpha 0.005)
tmux new-session -d -s joint_timesfm_kgi_a0005 "CUDA_VISIBLE_DEVICES=7 python scripts/training/train_joint.py model=timesfm ++model.use_kgi=true ++model.alpha_joint=0.005 ++trainer.max_epochs=40 ++data.processed_dir=data/sota ++data.batch_size=128; read"



echo "All 4 TimesFM joint training jobs have been dispatched!"
echo "Use the following commands to monitor them:"
echo "-------------------------------------------"
echo "tmux a -t joint_timesfm_vanilla_a01"
echo "tmux a -t joint_timesfm_kgi_a01"
echo "tmux a -t joint_timesfm_kgi_a001"
echo "tmux a -t joint_timesfm_kgi_a0005"
echo "-------------------------------------------"
