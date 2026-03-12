#!/bin/bash
# SSSD Ablations Pipeline (NoTr, Core, Emergency)
# Targeting GPU 6-7

TASKS=("ihm" "ss" "vr" "los")
SUBSETS=("no_treatments" "core" "emergency")

# --- GPU 6: SSSD Vanilla Ablations ---
(
  for subset in "${SUBSETS[@]}"; do
    for task in "${TASKS[@]}"; do
      echo "[GPU 6] SSSD Vanilla | Task: $task | Subset: $subset"
      CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=$task \
        data.feature_subset=$subset data.processed_dir=data/processed_sepsis_full \
        model.imputator_name=sssd +model.imputator_kwargs.inference_steps=5 \
        model.imputator_kwargs.use_kgi=false trainer.max_epochs=100
    done
  done
) &

# --- GPU 7: SSSD DGI v2 Ablations ---
(
  for subset in "${SUBSETS[@]}"; do
    for task in "${TASKS[@]}"; do
      echo "[GPU 7] SSSD DGI v2 | Task: $task | Subset: $subset"
      CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=$task \
        data.feature_subset=$subset data.processed_dir=data/processed_sepsis_full \
        model.imputator_name=sssd +model.imputator_kwargs.inference_steps=5 \
        model.imputator_kwargs.use_kgi=true +model.imputator_kwargs.kgi_mode=dgi_mask \
        trainer.max_epochs=100
    done
  done
) &

wait
echo "SSSD ABLATIONS COMPLETED!"
