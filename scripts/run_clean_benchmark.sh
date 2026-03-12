#!/bin/bash
# Sepsis-3 Strict Benchmark (Cleanest Version)
# Targeting GPU 6-7

TASKS=("ihm" "ss" "vr" "los")
INJECTIONS=("vanilla" "dki" "dgi" "dgi_mask")

# --- GPU 6: SAITS Suite ---
echo "Starting GPU 6: SAITS Suite (Clean Dataset)"
(
  for task in "${TASKS[@]}"; do
    for inj in "${INJECTIONS[@]}"; do
      echo "Launching SAITS: Task=$task, Inj=$inj"
      CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=$task \
        data.feature_subset=full data.processed_dir=data/processed_sepsis_full \
        model.imputator_name=saits trainer.max_epochs=30 \
        model.imputator_kwargs.use_kgi=$( [ "$inj" == "vanilla" ] && echo "false" || echo "true" ) \
        $( [ "$inj" != "vanilla" ] && echo "+model.imputator_kwargs.kgi_mode=$inj" )
    done
  done
) &

# --- GPU 7: Transformer Suite + SSSD ---
echo "Starting GPU 7: Transformer & SSSD (Clean Dataset)"
(
  # Transformer variants
  for task in "${TASKS[@]}"; do
    for inj in "${INJECTIONS[@]}"; do
      echo "Launching Transformer: Task=$task, Inj=$inj"
      CUDA_VISIBLE_DEVICES=7 python scripts/train_sepsis_benchmarks.py \
        --task $task --model transformer --feature_subset full --epochs 30 --batch_size 64 \
        --data_dir data/processed_sepsis_full \
        $( [ "$inj" != "vanilla" ] && echo "--use_kgi --kgi_mode $inj" )
    done
  done

  # SSSD Baseline (for IHM)
  echo "Launching SSSD: Task=ihm"
  CUDA_VISIBLE_DEVICES=7 python train.py model=joint model.task=ihm \
    data.processed_dir=data/processed_sepsis_full \
    model.imputator_name=sssd +model.imputator_kwargs.inference_steps=5 trainer.max_epochs=20
) &

wait
echo "Clean Sepsis Benchmark Completed!"

# --- ABLATION EXTENSION (Added dynamically) ---
echo "Starting Ablation Suite (No Treatments, Core, Emergency)..."

SUBSETS=("no_treatments" "core" "emergency")

# --- GPU 6: SAITS Ablations ---
(
  for subset in "${SUBSETS[@]}"; do
    for task in "${TASKS[@]}"; do
      for inj in "${INJECTIONS[@]}"; do
        echo "Queuing SAITS Ablation: Task=$task, Subset=$subset, Inj=$inj"
        CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=$task \
          data.feature_subset=$subset data.processed_dir=data/processed_sepsis_full \
          model.imputator_name=saits trainer.max_epochs=30 \
          model.imputator_kwargs.use_kgi=$( [ "$inj" == "vanilla" ] && echo "false" || echo "true" ) \
          $( [ "$inj" != "vanilla" ] && echo "+model.imputator_kwargs.kgi_mode=$inj" )
      done
    done
  done
) &

# --- GPU 7: Transformer Ablations ---
(
  for subset in "${SUBSETS[@]}"; do
    for task in "${TASKS[@]}"; do
      for inj in "${INJECTIONS[@]}"; do
        echo "Queuing Transformer Ablation: Task=$task, Subset=$subset, Inj=$inj"
        CUDA_VISIBLE_DEVICES=7 python scripts/train_sepsis_benchmarks.py \
          --task $task --model transformer --feature_subset $subset --epochs 30 --batch_size 64 \
          --data_dir data/processed_sepsis_full \
          $( [ "$inj" != "vanilla" ] && echo "--use_kgi --kgi_mode $inj" )
      done
    done
  done
) &

wait
echo "All Sepsis-3 Strict Benchmarks (Full + Ablations) Completed!"

# --- DUMMY SEMANTICS CONTROL (The Reviewer's Challenge) ---
echo "Starting Dummy Semantics Control Suite (Random Noise instead of UMLS)..."

DUMMY_EMB="data/embeddings/dummy_random_embeddings.pkl"

# --- GPU 6: SAITS Dummy Runs ---
(
  for task in "${TASKS[@]}"; do
    echo "Queuing SAITS Dummy Control: Task=$task"
    CUDA_VISIBLE_DEVICES=6 python train.py model=joint model.task=$task \
      data.feature_subset=full data.processed_dir=data/processed_sepsis_full \
      model.imputator_name=saits trainer.max_epochs=30 \
      model.imputator_kwargs.use_kgi=true +model.imputator_kwargs.kgi_mode=dgi_mask \
      model.imputator_kwargs.kgi_embedding_file=$DUMMY_EMB
  done
) &

# --- GPU 7: Transformer Dummy Runs ---
(
  for task in "${TASKS[@]}"; do
    echo "Queuing Transformer Dummy Control: Task=$task"
    CUDA_VISIBLE_DEVICES=7 python scripts/train_sepsis_benchmarks.py \
      --task $task --model transformer --feature_subset full --epochs 30 --batch_size 64 \
      --data_dir data/processed_sepsis_full \
      --use_kgi --kgi_mode dgi_mask --kgi_embedding $DUMMY_EMB
  done
) &

wait
echo "All Control Experiments Completed!"
