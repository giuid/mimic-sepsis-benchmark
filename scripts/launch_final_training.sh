#!/bin/bash
# Final Sepsis-3 Training Matrix - REVISION GRADE
# Uses GPU 4-7 exclusively. Data: data/processed_sepsis_full (55 feat)

DATA_DIR="data/processed_sepsis_full"
MAX_EPOCHS=30

# --- GPU 4: SAITS DGI v2 (Static & Regression) ---
echo "Launching GPU 4: SAITS DGI v2 (IHM & LOS)"
(
  CUDA_VISIBLE_DEVICES=4 python train.py model=joint model.task=ihm data.feature_subset=full \
    model.imputator_kwargs.use_kgi=true +model.imputator_kwargs.kgi_mode=dgi_mask trainer.max_epochs=$MAX_EPOCHS
  
  CUDA_VISIBLE_DEVICES=4 python train.py model=joint model.task=los data.feature_subset=full \
    model.imputator_kwargs.use_kgi=true +model.imputator_kwargs.kgi_mode=dgi_mask trainer.max_epochs=$MAX_EPOCHS
) &

# --- GPU 5: SAITS DGI v2 (Dynamic Tasks) ---
echo "Launching GPU 5: SAITS DGI v2 (SS & VR)"
(
  CUDA_VISIBLE_DEVICES=5 python train.py model=joint model.task=ss data.feature_subset=full \
    model.imputator_kwargs.use_kgi=true +model.imputator_kwargs.kgi_mode=dgi_mask trainer.max_epochs=$MAX_EPOCHS
  
  CUDA_VISIBLE_DEVICES=5 python train.py model=joint model.task=vr data.feature_subset=no_treatments \
    model.imputator_kwargs.use_kgi=true +model.imputator_kwargs.kgi_mode=dgi_mask trainer.max_epochs=$MAX_EPOCHS
) &

# --- GPU 6: Transformer DGI v2 (Reference Benchmarks) ---
echo "Launching GPU 6: Transformer DGI v2"
(
  CUDA_VISIBLE_DEVICES=6 python scripts/train_sepsis_benchmarks.py --task ihm --model transformer \
    --feature_subset full --use_kgi --kgi_mode dgi_mask --epochs $MAX_EPOCHS
    
  CUDA_VISIBLE_DEVICES=6 python scripts/train_sepsis_benchmarks.py --task vr --model transformer \
    --feature_subset no_treatments --use_kgi --kgi_mode dgi_mask --epochs $MAX_EPOCHS
) &

# --- GPU 7: SSSD (Generative Baseline SOTA) ---
echo "Launching GPU 7: SSSD Downstream"
(
  CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=ihm model.imputator_name=sssd \
    +model.imputator_kwargs.residual_layers=6 +model.imputator_kwargs.T=200 \
    +model.imputator_kwargs.inference_steps=5 trainer.max_epochs=20
) &

wait
echo "All Revision-Grade trainings completed!"
