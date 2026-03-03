#!/bin/bash
# Benchmark Script for MIMIC-IV Full Dataset (17 Features)
# Parallel Execution on GPUs 4, 5, 6

# Ensure script stops on error
set -e

echo "Starting Parallel Benchmark Sequence..."

# 1. Vanilla SAITS (Baseline) -> GPU 4
echo "Launching Vanilla SAITS on GPU 4..."
CUDA_VISIBLE_DEVICES=4 python3 train.py \
    model=saits \
    +model.use_graph_layer=False \
    masking=random masking.p=0.3 \
    trainer.devices=1 \
    output_dir=outputs/benchmark/vanilla_saits \
    +logging.name=vanilla_saits_full &
PID1=$!

# 2. Prior Nullo (Graph Baseline) -> GPU 5
echo "Launching Prior Nullo on GPU 5..."
CUDA_VISIBLE_DEVICES=5 python3 train.py \
    model=saits \
    +model.use_graph_layer=True \
    model.use_prior_init=False \
    model.graph_loss_weight=0.0 \
    masking=random masking.p=0.3 \
    trainer.devices=1 \
    output_dir=outputs/benchmark/prior_nullo \
    +logging.name=prior_nullo_full &
PID2=$!

# 3. SapBERT + CI-GNN (Roadmap 3.0 Best Code) -> GPU 6
echo "Launching SapBERT + CI-GNN on GPU 6..."
CUDA_VISIBLE_DEVICES=6 python3 train.py \
    model=saits \
    +model.use_graph_layer=True \
    model.use_prior_init=True \
    model.graph_loss_weight=0.001 \
    model.warmup_epochs=20 \
    model.dag_loss_weight=0.1 \
    masking=random masking.p=0.3 \
    trainer.devices=1 \
    output_dir=outputs/benchmark/sapbert_ci_gnn \
    +logging.name=sapbert_ci_gnn_full &
PID3=$!

echo "All training jobs launched in background."
echo "PIDs: $PID1 (Vanilla), $PID2 (Nullo), $PID3 (SapBERT)"
echo "Waiting for completion..."
wait $PID1 $PID2 $PID3
echo "Training Complete. Starting Evaluation..."

# Evaluation
# We need to find the best checkpoint for each model.
# Since filenames are dynamic (best-epoch=XX-val_loss=YY.ckpt), we use a helper or glob.

# Helper function to evaluate
evaluate_model() {
    MODEL_DIR=$1
    NAME=$2
    GPU=$3
    
    # Find best checkpoint
    CKPT=$(find ${MODEL_DIR}/checkpoints -name "best-*.ckpt" | head -n 1)
    
    if [ -z "$CKPT" ]; then
        echo "Error: No checkpoint found for $NAME in ${MODEL_DIR}"
        return
    fi
    
    echo "Evaluating $NAME on GPU $GPU (Ckpt: $CKPT)..."
    CUDA_VISIBLE_DEVICES=$GPU python3 evaluate.py \
        --model saits \
        --checkpoint "$CKPT" \
        --masking random --masking_p 0.3 \
        --device cuda \
        --output_dir results/benchmark/${NAME}
}

evaluate_model "outputs/benchmark/vanilla_saits" "vanilla_saits" 4 &
evaluate_model "outputs/benchmark/prior_nullo" "prior_nullo" 5 &
evaluate_model "outputs/benchmark/sapbert_ci_gnn" "sapbert_ci_gnn" 6 &

wait
echo "Benchmark Sequence Complete!"
