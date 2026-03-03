#!/bin/bash
# ============================================================
#  DKI Evaluation Script — 2026-02-27
#  Evaluates SAITS, MRNN, BRITS (DKI vs Vanilla)
#  GPUs: 4, 5, 6, 7
# ============================================================

SESSION="DKI_EVAL"
tmux kill-session -t "$SESSION" 2>/dev/null

# Define Models and Checkpoints
# Batching: 4 GPUs, 6 models -> 2 GPUs will run 2 models sequentially
MODELS=(
    "SAITS_DKI" "outputs/checkpoints/joint_saits/sapbert_KGI_a0.01/best-epoch=17-val/auprc=0.2963.ckpt" 4
    "MRNN_DKI" "outputs/checkpoints/joint_mrnn/vanilla_KGI_a0.01/best-epoch=25-val/auprc=0.2625.ckpt" 5
    "BRITS_DKI" "outputs/checkpoints/joint_brits/vanilla_KGI_a0.01/best-epoch=08-val/auprc=0.1308.ckpt" 6
    "SAITS_VANILLA" "outputs/checkpoints/joint_saits/vanilla_Vanilla_a0.01/best-epoch=20-val/auprc=0.2952-v1.ckpt" 7
    "MRNN_VANILLA" "outputs/checkpoints/joint_mrnn/vanilla_Vanilla/best-epoch=31-val/auprc=0.2814-v1.ckpt" 4
    "BRITS_VANILLA" "outputs/checkpoints/joint_brits/vanilla_Vanilla/best-epoch=37-val/auprc=0.1487-v1.ckpt" 5
)

echo "Launching DKI Evaluation in tmux session: $SESSION"

# Create session and first pane
tmux new-session -d -s "$SESSION" -n "Evals"

# GPU 4: SAITS_DKI then MRNN_VANILLA
tmux send-keys -t "$SESSION:0" "CUDA_VISIBLE_DEVICES=4 python scripts/eval/eval_kgi.py --model_name 'SAITS_DKI' --ckpt_path 'outputs/checkpoints/joint_saits/sapbert_KGI_a0.01/best-epoch=17-val/auprc=0.2963.ckpt'; 
CUDA_VISIBLE_DEVICES=4 python scripts/eval/eval_kgi.py --model_name 'MRNN_VANILLA' --ckpt_path 'outputs/checkpoints/joint_mrnn/vanilla_Vanilla/best-epoch=31-val/auprc=0.2814-v1.ckpt'; 
echo 'GPU 4 DONE'; read" Enter

# GPU 5: MRNN_DKI then BRITS_VANILLA
tmux split-window -h -t "$SESSION:0"
tmux send-keys -t "$SESSION:0.1" "CUDA_VISIBLE_DEVICES=5 python scripts/eval/eval_kgi.py --model_name 'MRNN_DKI' --ckpt_path 'outputs/checkpoints/joint_mrnn/vanilla_KGI_a0.01/best-epoch=25-val/auprc=0.2625.ckpt'; 
CUDA_VISIBLE_DEVICES=5 python scripts/eval/eval_kgi.py --model_name 'BRITS_VANILLA' --ckpt_path 'outputs/checkpoints/joint_brits/vanilla_Vanilla/best-epoch=37-val/auprc=0.1487-v1.ckpt'; 
echo 'GPU 5 DONE'; read" Enter

# GPU 6: BRITS_DKI
tmux split-window -v -t "$SESSION:0.0"
tmux send-keys -t "$SESSION:0.2" "CUDA_VISIBLE_DEVICES=6 python scripts/eval/eval_kgi.py --model_name 'BRITS_DKI' --ckpt_path 'outputs/checkpoints/joint_brits/vanilla_KGI_a0.01/best-epoch=08-val/auprc=0.1308.ckpt'; 
echo 'GPU 6 DONE'; read" Enter

# GPU 7: SAITS_VANILLA
tmux split-window -v -t "$SESSION:0.1"
tmux send-keys -t "$SESSION:0.3" "CUDA_VISIBLE_DEVICES=7 python scripts/eval/eval_kgi.py --model_name 'SAITS_VANILLA' --ckpt_path 'outputs/checkpoints/joint_saits/vanilla_Vanilla_a0.01/best-epoch=20-val/auprc=0.2952-v1.ckpt'; 
echo 'GPU 7 DONE'; read" Enter

tmux select-layout -t "$SESSION" tiled

echo "Success! Monitor with: tmux a -t $SESSION"
