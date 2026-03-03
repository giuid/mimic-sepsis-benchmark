# MIMIC-IV Imputation Baselines (SSSD & SAITS)

Structured Python repository for multivariate time-series imputation on MIMIC-IV ICU data.

## Models
- **SAITS** – Self-Attention-based Imputation (Du et al., ESWA 2023)
- **SSSD** – Diffusion + Structured State Spaces (Alcaraz & Strodthoff, TMLR 2023)
- Simple baselines: Mean, LOCF, Linear Interpolation

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Extract & preprocess MIMIC-IV data
python -m data.extract
python -m data.preprocess

# Train SAITS with random masking (30%)
python train.py model=saits masking=random masking.p=0.3

# Train SSSD with block masking
python train.py model=sssd masking=block

# Evaluate
python evaluate.py model=saits masking=random masking.p=0.3 checkpoint=outputs/saits/...
```

## Project Structure
```
configs/     – Hydra YAML configs (data, model, masking)
data/        – MIMIC-IV extraction, preprocessing, dataset, masking
models/      – SAITS and SSSD implementations
metrics/     – MAE, RMSE (global + per-variable)
baselines_simple/ – Mean, LOCF, Linear Interpolation
results/     – Output tables and results README
```

## Data
- **Source**: MIMIC-IV v3.1 ICU (`chartevents`, `labevents`)
- **Features (D=9)**: HR, SBP, DBP, RR, SpO₂, Creatinine, Lactate, Glucose, BUN
- **Window**: First 48h of ICU stay, 1h resolution (T=48)
- **Coverage filter**: ≥30% observed values per stay
- **Split**: 70/15/15 by patient (no leakage)
