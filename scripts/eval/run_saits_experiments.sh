#!/bin/bash
# Run 5 SAITS experimental configurations

# 1. Vanilla (No Embeddings, No Graph Loss)
echo "Running Experiment 1: Vanilla"
python train.py \
    --model saits \
    --config configs/data/mimic4.yaml \
    model.embedding_type=vanilla \
    model.graph_loss_weight=0.0 \
    training.max_epochs=30 \
    training.gpus=1 \
    training.run_name=saits_vanilla

# 2. SapBERT (No Loss)
# Uses SapBERT embeddings for initialization/features but NO graph regularization loss.
echo "Running Experiment 2: SapBERT (No Loss)"
python train.py \
    --model saits \
    --config configs/data/mimic4.yaml \
    model.embedding_type=sapbert \
    model.graph_loss_weight=0.0 \
    training.max_epochs=30 \
    training.gpus=1 \
    training.run_name=saits_sapbert_noloss

# 3. SapBERT (With Loss)
# Uses SapBERT embeddings + Graph regularization (enforcing similarity to SapBERT prior).
echo "Running Experiment 3: SapBERT (With Loss)"
python train.py \
    --model saits \
    --config configs/data/mimic4.yaml \
    model.embedding_type=sapbert \
    model.graph_loss_weight=1.0 \
    training.max_epochs=30 \
    training.gpus=1 \
    training.run_name=saits_sapbert_loss

# 4. BERT (Non-Medical)
# Uses BERT embeddings. We enable loss to test the prior derived from generic language.
# (If user wanted no loss, set graph_loss_weight=0.0)
echo "Running Experiment 4: BERT (Non-Medical)"
python train.py \
    --model saits \
    --config configs/data/mimic4.yaml \
    model.embedding_type=bert \
    model.graph_loss_weight=1.0 \
    training.max_epochs=30 \
    training.gpus=1 \
    training.run_name=saits_bert

# 5. Random Embeddings
# Uses Random embeddings. We enable loss to test the prior derived from random noise (baseline).
echo "Running Experiment 5: Random"
python train.py \
    --model saits \
    --config configs/data/mimic4.yaml \
    model.embedding_type=random \
    model.graph_loss_weight=1.0 \
    training.max_epochs=30 \
    training.gpus=1 \
    training.run_name=saits_random
