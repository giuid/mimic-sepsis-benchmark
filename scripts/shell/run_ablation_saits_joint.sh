#!/bin/bash
# scripts/shell/run_ablation_saits_joint.sh
# Multi-task Ablation Study for SAITS Joint

mkdir -p logs/ablation_joint

run_joint_ablation() {
    local gpu=$1
    local task=$2
    local subset=$3
    local variant=$4 # vanilla, dki
    local session_name="joint_${task}_${subset}_${variant}"
    
    local kgi="false"
    if [ "$variant" == "dki" ]; then kgi="true"; fi
    
    echo "Launching SAITS Joint: Task=$task, Subset=$subset, DKI=$kgi on GPU $gpu"
    # Use train.py with model=joint
    # Note: We don't have a direct 'subset' param in Hydra config for joint, 
    # but MIMICSepsisTaskDataset supports feature_subset via its __init__.
    # We need to ensure JointSepsisModule passes this correctly.
    
    # Actually, train.py uses MIMICDataModule which doesn't support subsets yet.
    # PROPER FIX: I need to update MIMICDataModule to pass feature_subset.
}

# --- WAIT: I need to update MIMICDataModule first to support subsets in Hydra ---
