# MIMIC-IV Imputation Baselines: Project Structure

This document provides a high-level overview of the folder structure for the MIMIC-IV Imputation Baselines project.

## Root Directories
- **`.agent/rules/`**: Custom rules for the AI assistant (Antigravity IDE rules).
- **`.daily_summaries/`**: Automatically generated summaries of daily work.
- **`.logs/`**: Detailed logs of session actions and research progress.
- **`artifacts_pruned/`**: Pruned versions of graph and relation data for efficiency.
- **`baselines_simple/`**: Simple imputation models (Mean, LOCF, Linear).
- **`checkpoints/`**: Model weight checkpoints saved during training.
- **`configs/`**: Hydra YAML configuration files for models, data, and masking.
- **`data/`**: Core data processing logic, dataset classes, and masking strategies.
- **`docs/`** & **`documentation/`**: Project-specific documentation and research notes.
- **`graph/`**: Knowledge graph-related data and processing.
- **`logs/`**: Training and execution logs (stdout/stderr).
- **`metrics/`**: Implementation of evaluation metrics (MAE, RMSE, etc.).
- **`models/`**: Deep learning model implementations (e.g., SAITS, SSSD).
- **`notebooks/`**: Jupyter notebooks for exploration and visualization.
- **`outputs/`**: Default output directory for Hydra runs (results, logs, configs).
- **`results/`**: Final evaluation results and tables.
- **`scripts/`**: Utility scripts for experiments, evaluation, and data management.
  - `data/`: Data processing, graph extraction, and dataset preparation utilities.
  - `eval/`: Scripts for evaluating models and generating downstream benchmarks.
  - `training/`: All Python and Bash scripts dedicated to model training (e.g. `train_joint.py`).
  - `analysis/`: Post-hoc analysis scripts, graph visualization, and feature reports.
  - `debug/`: Centralized debugging scripts (e.g., `debug_session.py`).
- **`tests/`**: Unit tests for the codebase.

## Key Files
- `train.py`: Main entry point for model training.
- `evaluate.py`: Main entry point for model evaluation.
- `preprocess_all.py`: Utility for preprocessing all necessary data.
- `requirements.txt`: Python package dependencies.
- `pyproject.toml`: Project configuration and dependency management.
- `README.md`: General project introduction and quick start guide.
- `STRUCTURE.md`: This file, documenting the folder structure.
