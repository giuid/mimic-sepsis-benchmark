#!/bin/bash
subsets=("full" "no_treatments" "core" "emergency")
for sub in "${subsets[@]}"; do
    python train.py model=joint data=mimic4_sepsis_full ++model.task=ihm ++data.feature_subset=$sub model.imputator_kwargs.use_kgi=false trainer.max_epochs=30 > logs/ablation_joint/ihm_${sub}_vanilla.log 2>&1
    python train.py model=joint data=mimic4_sepsis_full ++model.task=ihm ++data.feature_subset=$sub model.imputator_kwargs.use_kgi=true trainer.max_epochs=30 > logs/ablation_joint/ihm_${sub}_dki.log 2>&1
done
