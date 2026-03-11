import pandas as pd
import os
import subprocess
import time

# Configuration
TASKS = ["ihm", "los", "ss", "vr"]
SUBSETS = ["full", "no_treatments", "core", "emergency"]
TRANSFORMER_INJ = ["vanilla", "dki", "dgi"]
SAITS_INJ = ["vanilla", "dki", "dgi"]
GPUS = [4, 5, 6, 7]

# Load existing inventory
try:
    inventory = pd.read_csv("results/checkpoints_inventory.csv")
except:
    inventory = pd.DataFrame(columns=["model", "task", "subset", "injection"])

def exists(model, task, subset, inj):
    match = inventory[
        (inventory['model'] == model) & 
        (inventory['task'] == task) & 
        (inventory['subset'] == subset) & 
        (inventory['injection'] == inj)
    ]
    return len(match) > 0

commands = []

# 1. TRANSFORMER RUNS (via scripts/train_sepsis_benchmarks.py)
for task in TASKS:
    for subset in SUBSETS:
        for inj in TRANSFORMER_INJ:
            if not exists("transformer", task, subset, inj):
                cmd = f"CUDA_VISIBLE_DEVICES={GPUS[len(commands)%2]} python scripts/train_sepsis_benchmarks.py --task {task} --model transformer --feature_subset {subset} --epochs 30 --batch_size 64"
                if inj != "vanilla":
                    cmd += f" --use_kgi --kgi_mode {inj}"
                commands.append(cmd)

# 2. SAITS RUNS (via train.py - Hydra)
# Note: SAITS DKI is often mapped to 'kgi' in older logs.
for task in TASKS:
    for subset in SUBSETS:
        for inj in SAITS_INJ:
            if not exists("saits", task, subset, inj):
                # Using current train.py logic for joint models
                cmd = f"CUDA_VISIBLE_DEVICES={GPUS[2 + (len(commands)%2)]} python train.py model=joint model.task={task} data.feature_subset={subset} model.imputator_name=saits trainer.max_epochs=30"
                if inj == "vanilla":
                    cmd += " model.imputator_kwargs.use_kgi=false"
                else:
                    cmd += f" model.imputator_kwargs.use_kgi=true model.imputator_kwargs.kgi_mode={inj}"
                commands.append(cmd)

# Save commands to a bash script for execution in tmux
with open("scripts/run_weekend_matrix.sh", "w") as f:
    f.write("#!/bin/bash\n")
    f.write("# Sepsis Matrix Completion Script - Weekend Run\n\n")
    # Add some delay between launches to avoid overwhelming GPU memory/IO
    for i, cmd in enumerate(commands):
        f.write(f"echo 'Launching Run {i+1}/{len(commands)}: {cmd}'\n")
        f.write(f"{cmd}\n")
        f.write("sleep 10\n\n")

print(f"Generated {len(commands)} missing runs in scripts/run_weekend_matrix.sh")
print("Matrix Completion Targets (GPU 4,5 for Transf, 6,7 for SAITS)")
