#!/bin/bash
tasks=("los" "ss" "vr")
for task in "${tasks[@]}"; do
    echo "Starting SAITS Joint Vanilla - Task: $task"
    python train.py model=joint data=mimic4_sepsis_full ++model.task=$task model.imputator_kwargs.use_kgi=false data=mimic4_sepsis_full trainer.max_epochs=50 trainer.devices=[0] > logs/ablation/saits_joint_${task}_vanilla.log 2>&1
done
