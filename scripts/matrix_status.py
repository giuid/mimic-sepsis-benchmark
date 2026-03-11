import subprocess
import os
import psutil
import time

tasks = ["ihm", "los", "ss", "vr"]
subsets = ["full", "no_treatments", "core", "emergency"]
injections = ["vanilla", "dki", "dgi"] # For Transformer

def is_running(task, subset, model, injection):
    for proc in psutil.process_iter(['cmdline']):
        try:
            cmd = proc.info['cmdline']
            if not cmd: continue
            cmd_str = " ".join(cmd)
            if "train_sepsis_benchmarks.py" in cmd_str:
                if f"--task {task}" in cmd_str and f"--feature_subset {subset}" in cmd_str:
                    if injection == "vanilla" and "--use_kgi" not in cmd_str:
                        return True
                    if injection in ["dki", "dgi"] and f"--kgi_mode {injection}" in cmd_str:
                        return True
            if "train.py" in cmd_str and "model=joint" in cmd_str:
                # Hydra joint training usually means SAITS DKI on Full
                if f"model.task={task}" in cmd_str and subset == "full" and injection == "dki":
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False

def check_exists(task, subset, model, injection):
    # Check hierarchical directory for ANY file matching the pattern
    path = f"checkpoints/sepsis/{task}/{subset}"
    if not os.path.exists(path): return False
    for f in os.listdir(path):
        if model in f and injection in f:
            return True
    return False

# COMPLETION LOOP for Transformer
print("--- Sepsis Matrix Completion Script (Transformer) ---")
for task in tasks:
    for subset in subsets:
        for inj in injections:
            if check_exists(task, subset, "transformer", inj):
                # print(f"Skipping {task}-{subset}-{inj}: Already exists.")
                continue
            
            if is_running(task, subset, "transformer", inj):
                print(f"Skipping {task}-{subset}-{inj}: Currently running.")
                continue
            
            # Launch command
            cmd = [
                "python", "scripts/train_sepsis_benchmarks.py",
                "--task", task,
                "--model", "transformer",
                "--feature_subset", subset,
                "--epochs", "30",
                "--batch_size", "64"
            ]
            if inj in ["dki", "dgi"]:
                cmd.append("--use_kgi")
                cmd.append("--kgi_mode")
                cmd.append(inj)
            
            print(f"Launching: {' '.join(cmd)}")
            # For demonstration, we launch one at a time or in background
            # subprocess.Popen(cmd) 
            # In a real scenario, we might want to limit concurrent runs based on GPU
            # For now, let's just print the commands or launch them carefully.
            # (Executing actual launch requires user confirmation or background management)

print("\n--- Next Steps for SAITS (Vanilla, DKI, DGI) ---")
print("Missing SAITS ablations and DGI mode should be launched similarly via train_sepsis_pypots.py or joint trainer.")
