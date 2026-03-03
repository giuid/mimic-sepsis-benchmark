# Scripts Directory: Catalog & Usage

This directory contains various utility and experiment scripts for the project, organized into subdirectories to reduce noise.

## Directory Structure

### [scripts/data/](data/)
Utility scripts for data processing, knowledge graph extraction, and exploratory data analysis.
- `ablate_knowledge.py`: Ablation studies on knowledge graph components.
- `extract_bert_embeddings.py`: Extracts BERT-based representations.
- `extract_true_umls.py`: Fetches UMLS relations.
- `fix_graph.sh`: Utility for fixing common graph data errors.
- `generate_medbert_relations.py`: MedBERT-based relation generation.
- `generate_sapbert_mimic.py`: SapBERT embeddings for MIMIC.
- `generate_sota_embeddings.py`: SOTA embedding generation.
- `inspect_triplets.py`: Knowledge graph triplet check.
- `lasso_quick_subset.py` / `lasso_selection.py`: Feature selection scripts.
- `show_data_example.py`: Data visualization.
- `umls_exploration.py`: UMLS data analysis.
- `verify_parquet.py`: Data integrity verification.

### [scripts/eval/](eval/)
Evaluation pipelines, benchmarks, and downstream task tests.
- `eval_imputation_kgi.py`: KGI-specific imputation evaluation.
- `eval_kgi.py` / `eval_sota.py`: KGI vs SOTA benchmarks.
- `evaluate_downstream.py` / `run_downstream_evals.py`: Clinical task evaluation.
- `evaluate_sssd_crash.sh`: SSSD-specific crash recovery evaluation.
- `evaluate_true_baselines.sh`: Baseline model evaluation.
- `launch_benchmark_tmux.sh` / `launch_true_baselines.sh`: Parallel execution utilities.
- `run_all_evals.py` / `run_benchmark.sh`: Batch evaluation triggers.
- `run_kgi_evals.py` / `run_mask_evals.py`: Masking and KGI experiments.
- `test_imputation_timesfm.py` / `test_kgi_fusion.py` / `test_loading_timesfm.py` / `test_saits_kgi.py`: Specialized test cases.

### [scripts/training/](scripts/training/)
Scripts related to model training, resumption, and experiment pipelines.
- `resume_training.sh`: Resumes training from a checkpoint.
- `run_joint_training.sh`: Parallel TMUX launcher for joint imputation+classification training.
- `run_sssd_*.sh`: Various SSSD training configurations.
- `train_baselines_*.sh`: Training pipelines for baselines.
- `train_joint.py`: End-to-end PyTorch Lightning script for combined representation learning.
- `train_saits_*.sh`: SAITS specific training pipelines.

### [scripts/analysis/](analysis/)
Post-hoc analysis of model behavior and results.
- `analyze_calibration.py`: Model calibration analysis.
- `analyze_feature_importance.py`: Feature importance computation.
- `analyze_learned_graph.py`: Graph structure analysis.
- `visualize_kgi_attention.py`: Attention pattern visualization.

### [scripts/debug/](debug/)
Internal debugging tools and sanity checks.
- `check_gates.py`: Gate value inspection.
- `debug_data.py`: Data loader debugging.
- `debug_gpvae*.py`: GP-VAE specific debugging.
- `debug_sssd.py`: SSSD diffusion debugging.
- `disambiguate_checkpoints.sh`: Checkpoint cleanup/linking.
- `test_s4_kernel.py`: S4 kernel verification.

### [scripts/legacy/](legacy/)
Older scripts that are kept for historical reference but are no longer active.

## Guidelines for New Scripts
1. **No Root Scripts**: Do not place new scripts in `scripts/`. Use the appropriate subfolder.
2. **Update README**: When adding a file, update this catalog with a brief description.
3. **Debug Session**: For temporary debugging, use `scripts/debug/debug_session.py`.
