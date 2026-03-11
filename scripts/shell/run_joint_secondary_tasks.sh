#!/bin/bash
# Train SAITS Joint KGI (96h Context, Grounded DKI) on Secondary Tasks (LOS, VR, SS)

SESSION="joint_96h_secondary"
tmux new-session -d -s $SESSION

# Train LOS on GPU 4
tmux new-window -t $SESSION -n "joint_los"
tmux send-keys -t $SESSION:"joint_los" "CUDA_VISIBLE_DEVICES=4 python train.py model=joint model.task=los data.processed_dir=data/processed_sepsis_full model.imputator_kwargs.use_kgi=true" C-m

# Train VR on GPU 5
tmux new-window -t $SESSION -n "joint_vr"
tmux send-keys -t $SESSION:"joint_vr" "CUDA_VISIBLE_DEVICES=5 python train.py model=joint model.task=vr data.processed_dir=data/processed_sepsis_full model.imputator_kwargs.use_kgi=true" C-m

# Train SS on GPU 6
tmux new-window -t $SESSION -n "joint_ss"
tmux send-keys -t $SESSION:"joint_ss" "CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=ss data.processed_dir=data/processed_sepsis_full model.imputator_kwargs.use_kgi=true" C-m

# Kill default window
tmux kill-window -t $SESSION:0

echo "Parallel runs dispatched on GPUs 4 (LOS), 5 (VR), and 6 (SS)."
echo "Monitor with: tmux attach -t $SESSION"
