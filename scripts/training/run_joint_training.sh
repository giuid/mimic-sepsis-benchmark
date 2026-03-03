#!/bin/bash

# Array of GPUs to use (4 to 7 included)
GPUS=(4 5 6 7)

# Configurations for SAITS only
CONFIGS=(
    "model=saits ++model.embedding_type=vanilla ++model.use_kgi=False ++model.alpha_joint=0.005"
    "model=saits ++model.embedding_type=sapbert ++model.use_kgi=True ++model.kgi_embedding_file=medbert_relation_embeddings_generic.pkl ++model.alpha_joint=0.005"
    "model=saits ++model.embedding_type=vanilla ++model.use_kgi=False ++model.alpha_joint=0.01"
    "model=saits ++model.embedding_type=sapbert ++model.use_kgi=True ++model.kgi_embedding_file=medbert_relation_embeddings_generic.pkl ++model.alpha_joint=0.01"
)

NAMES=(
    "joint_saits_vanilla_a0005"
    "joint_saits_kgi_a0005"
    "joint_saits_vanilla_a01"
    "joint_saits_kgi_a01"
)

echo "Starting SAITS joint training jobs with reduced alpha (0.005 and 0.01) on GPUs 4-7..."

for i in "${!CONFIGS[@]}"; do
    CFG="${CONFIGS[$i]}"
    NAME="${NAMES[$i]}"
    GPU="${GPUS[$i]}"
    
    echo "Launching $NAME on GPU $GPU..."
    
    # Kill any old session with the same name
    tmux kill-session -t "$NAME" 2>/dev/null
    
    # Launch new specialized tmux session
    tmux new-session -d -s "$NAME" "cd /home/guido/Code/charite/baselines && CUDA_VISIBLE_DEVICES=$GPU python scripts/training/train_joint.py $CFG data.processed_dir=data/sota trainer.devices=1 trainer.accelerator=gpu trainer.max_epochs=40; echo 'DONE - press enter to close'; read"
done

echo ""
echo "All 4 SAITS joint training jobs have been dispatched!"
echo "Use the following commands to monitor them:"
echo "-------------------------------------------"
for name in "${NAMES[@]}"; do
    echo "tmux a -t $name"
done
echo "-------------------------------------------"
echo "(Ctrl+b, then d to detach from a session)"
