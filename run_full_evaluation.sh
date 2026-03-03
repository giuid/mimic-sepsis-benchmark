#!/bin/bash

# ==============================================================================
# FULL EVALUATION PIPELINE (Imputation + Mortality Prediction)
# GPU: 7
# ==============================================================================

export CUDA_VISIBLE_DEVICES=7
RESULTS_DIR="results"
mkdir -p $RESULTS_DIR

# Configurazione Masking (Default per valutazione)
MASKING="random"
P=0.3

echo "===================================================================="
echo "🚀 STARTING COMPREHENSIVE EVALUATION ON GPU 7"
echo "===================================================================="

# Funzione per trovare il miglior checkpoint (più recente)
find_best_ckpt() {
    local dir=$1
    if [ -d "$dir" ]; then
        # Cerca file .ckpt ricorsivamente e prendi il più recente
        find "$dir" -name "*.ckpt" -type f -printf '%T@ %p
' | sort -n | tail -1 | cut -f2- -d" "
    else
        echo ""
    fi
}

# 1. VALUTAZIONE DEI SETUP (SOTA e HANDPICKED)
for SETUP in "sota" "handpicked"; do
    DATA_DIR="data/$SETUP"
    
    if [ ! -d "$DATA_DIR" ]; then
        echo "⚠️  Data directory $DATA_DIR not found. Skipping setup $SETUP."
        continue
    fi

    echo ""
    echo "--------------------------------------------------------------------"
    echo "📊 PROCESSING SETUP: $SETUP"
    echo "--------------------------------------------------------------------"

    # --- A. Baselines Semplici ---
    echo "🔹 Running simple baselines (mean, locf, linear_interp) for $SETUP..."
    python evaluate.py --baselines_only --data_dir $DATA_DIR --masking $MASKING --masking_p $P
    
    for BASE in "mean" "locf" "linear_interp"; do
        python scripts/evaluate_downstream.py --model_name $BASE --setup $SETUP --data_dir $DATA_DIR --device cuda
    done

    # --- B. Modelli Deep (Scansione Automatica) ---
    # Percorso: outputs/<setup>/<model>/<masking>/D17_<timestamp>/checkpoints/
    for MODEL_DIR in outputs/$SETUP/*; do
        MODEL=$(basename $MODEL_DIR)
        
        # Saltiamo le cartelle che non sono modelli
        if [ ! -d "$MODEL_DIR" ]; then continue; fi
        
        for MASK_DIR in $MODEL_DIR/*; do
            # Cerchiamo gli esperimenti all'interno del tipo di masking
            for EXP_DIR in $MASK_DIR/*; do
                if [ ! -d "$EXP_DIR" ]; then continue; fi
                
                CKPT=$(find_best_ckpt "$EXP_DIR")
                
                if [ -n "$CKPT" ]; then
                    echo ""
                    echo "📍 Evaluating Model: $MODEL ($SETUP)"
                    echo "   Path: $EXP_DIR"
                    echo "   CKPT: $(basename $CKPT)"
                    
                    # 1. Imputation Evaluation
                    python evaluate.py 
                        --model $MODEL 
                        --checkpoint "$CKPT" 
                        --masking $MASKING 
                        --masking_p $P 
                        --data_dir $DATA_DIR 
                        --gpus 1

                    # 2. Downstream Evaluation (Mortality)
                    python scripts/evaluate_downstream.py 
                        --model_name $MODEL 
                        --setup $SETUP 
                        --checkpoint "$CKPT" 
                        --data_dir $DATA_DIR 
                        --device cuda
                fi
            done
        done
    done
done

echo ""
echo "===================================================================="
echo "✅ ALL EVALUATIONS COMPLETED"
echo "Check 'results/' for master_benchmark.csv and mortality_master_benchmark.csv"
echo "===================================================================="
