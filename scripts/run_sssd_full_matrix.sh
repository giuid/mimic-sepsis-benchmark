#!/bin/bash
# SSSD Master Pipeline: Full Experimental Matrix
# Targeting GPU 4-5

TASKS=("ihm" "ss" "vr" "los")
SUBSETS=("full" "no_treatments" "core" "emergency")

# Helper to find checkpoint for specific task/subset/kgi
find_sssd_ckpt() {
  local task=$1
  local subset=$2
  local kgi=$3
  # Search in Hydra logs
  grep -l "task: $task" outputs/mimic4/joint/random/*/experiment_metadata.txt 2>/dev/null | while read meta; do
    if grep -q "feature_subset: $subset" "$meta" && grep -q "use_kgi: $kgi" "$meta"; then
      dir=$(dirname "$meta")
      [ -f "$dir/checkpoints/default/last.ckpt" ] && echo "$dir/checkpoints/default/last.ckpt" && return
    fi
  done
}

# --- GPU 4: SSSD Vanilla Suite ---
echo "Launching SSSD Vanilla Suite on GPU 4..."
(
  for subset in "${SUBSETS[@]}"; do
    for task in "${TASKS[@]}"; do
      CKPT=$(find_sssd_ckpt "$task" "$subset" "false")
      RESUME_ARG=""
      [ -n "$CKPT" ] && RESUME_ARG="+checkpoint=$CKPT"
      
      echo "[GPU 4] SSSD Vanilla | Task: $task | Subset: $subset"
      CUDA_VISIBLE_DEVICES=4 python train.py model=joint model.task=$task \
        data.feature_subset=$subset data.processed_dir=data/processed_sepsis_full \
        model.imputator_name=sssd +model.imputator_kwargs.inference_steps=5 \
        model.imputator_kwargs.use_kgi=false trainer.max_epochs=100 $RESUME_ARG
    done
  done
) &

# --- GPU 5: SSSD DGI v2 Suite ---
echo "Launching SSSD DGI v2 Suite on GPU 5..."
(
  for subset in "${SUBSETS[@]}"; do
    for task in "${TASKS[@]}"; do
      CKPT=$(find_sssd_ckpt "$task" "$subset" "true")
      RESUME_ARG=""
      [ -n "$CKPT" ] && RESUME_ARG="+checkpoint=$CKPT"

      echo "[GPU 5] SSSD DGI v2 | Task: $task | Subset: $subset"
      CUDA_VISIBLE_DEVICES=5 python train.py model=joint model.task=$task \
        data.feature_subset=$subset data.processed_dir=data/processed_sepsis_full \
        model.imputator_name=sssd +model.imputator_kwargs.inference_steps=5 \
        model.imputator_kwargs.use_kgi=true +model.imputator_kwargs.kgi_mode=dgi_mask \
        trainer.max_epochs=100 $RESUME_ARG
    done
  done
) &

wait
echo "MASTER SSSD PIPELINE COMPLETED!"
