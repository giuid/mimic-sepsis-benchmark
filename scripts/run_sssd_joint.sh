#!/bin/bash
# SSSD Joint Training Benchmark (Fresh Start)
# Targeting GPU 4-5 to 100 epochs

TASKS=("ihm" "ss" "vr" "los")

# --- GPU 4: SSSD Vanilla ---
(
  for task in "${TASKS[@]}"; do
    echo "Launching SSSD Vanilla: Task=$task"
    CUDA_VISIBLE_DEVICES=4 python train.py model=joint model.task=$task \
      data.processed_dir=data/processed_sepsis_full \
      model.imputator_name=sssd +model.imputator_kwargs.inference_steps=5 \
      model.imputator_kwargs.use_kgi=false trainer.max_epochs=100
  done
) &

# --- GPU 5: SSSD DGI v2 ---
(
  for task in "${TASKS[@]}"; do
    echo "Launching SSSD DGI v2: Task=$task"
    CUDA_VISIBLE_DEVICES=5 python train.py model=joint model.task=$task \
      data.processed_dir=data/processed_sepsis_full \
      model.imputator_name=sssd +model.imputator_kwargs.inference_steps=5 \
      model.imputator_kwargs.use_kgi=true +model.imputator_kwargs.kgi_mode=dgi_mask \
      trainer.max_epochs=100
  done
) &

wait
echo "SSSD Joint Benchmarks (100 Epochs) Completed!"
