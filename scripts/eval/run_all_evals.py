import os
import glob
import subprocess

setups = ["sota", "handpicked"]
models_simple = ["mean", "locf", "linear_interp"]
# Base models for discovery
models_deep = ["brits", "gpvae", "mrnn", "sssd", "timesfm", "timesfm_sapbert"]

# SAITS variants will be handled specially
saits_variants = ["saits_vanilla", "saits_sapbert"]

available_gpus = [4, 5, 6, 7]

def get_saits_checkpoint(setup, variant):
    base_dir = f"outputs/{setup}/saits/random"
    if not os.path.exists(base_dir):
        return None
    
    dirs = glob.glob(f"{base_dir}/D17_*")
    dirs.sort(reverse=True)
    
    target_type = "sapbert" if variant == "saits_sapbert" else "vanilla"
    
    for d in dirs:
        meta_path = f"{d}/experiment_metadata.txt"
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                content = f.read()
                # Check for SapBERT or Vanilla
                if variant == "saits_sapbert":
                    if "embedding_type: sapbert" in content or "use_prior_init: true" in content:
                        pass # Match
                    else:
                        continue
                else: # saits_vanilla
                    if "embedding_type: vanilla" in content or "use_prior_init: false" in content:
                        pass # Match
                    else:
                        continue
            
            # Found matching dir, now find best ckpt
            patterns = [
                f"{d}/checkpoints/best-*.ckpt",
                f"{d}/checkpoints/best-epoch=*-val/*.ckpt"
            ]
            for p in patterns:
                checkpoints = glob.glob(p)
                if checkpoints:
                    checkpoints.sort()
                    return checkpoints[0]
                    
    return None

def get_best_checkpoint(setup, model):
    # For models other than SAITS variants
    base_dir = f"outputs/{setup}/{model}/random"
    if not os.path.exists(base_dir):
        return None
    
    dirs = glob.glob(f"{base_dir}/D17_*")
    if not dirs:
        return None
    
    dirs.sort(reverse=True)
    
    for d in dirs:
        patterns = [
            f"{d}/checkpoints/best-*.ckpt",
            f"{d}/checkpoints/best-epoch=*-val/*.ckpt"
        ]
        for p in patterns:
            checkpoints = glob.glob(p)
            if checkpoints:
                checkpoints.sort() 
                return checkpoints[0]
            
    return None

script_content = """#!/bin/bash
export OMP_NUM_THREADS=4

run_eval_on_gpu() {
    local gpu=$1
    local model=$2
    local setup=$3
    local data_dir=$4
    local ckpt=$5
    
    export CUDA_VISIBLE_DEVICES=$gpu
    echo "Starting $model on $setup using GPU $gpu"
    
    cmd="python scripts/evaluate_downstream.py --model_name $model --setup $setup --data_dir $data_dir --device cuda:0 --batch_size 256"
    if [ ! -z "$ckpt" ]; then
        cmd="$cmd --checkpoint '$ckpt'"
    fi
    
    eval $cmd || echo "Failed $model on $setup"
}

"""

gpu_idx = 0
tasks = []

# Prepare tasks
for setup in setups:
    data_dir = f"data/{setup}"
    
    # Baselines
    for model in models_simple:
        tasks.append((model, setup, data_dir, None))
        
    # SAITS variants
    for variant in saits_variants:
        ckpt = get_saits_checkpoint(setup, variant)
        if ckpt:
            tasks.append((variant, setup, data_dir, ckpt))
        else:
            print(f"Warning: Checkpoint not found for {variant} on {setup}")

    # Other deep models
    for model in models_deep:
        ckpt = get_best_checkpoint(setup, model)
        if ckpt:
            tasks.append((model, setup, data_dir, ckpt))
        else:
            # SSSD check in alternative path
            if model == "sssd":
                # Special check for sssd in outputs/sssd/random
                sssd_ckpt = "outputs/sssd/random/2026-02-18_12-20-48/checkpoints/best-epoch=49-val/loss=0.1457.ckpt"
                if os.path.exists(sssd_ckpt):
                     tasks.append((model, setup, data_dir, sssd_ckpt))
                     continue
            print(f"Warning: Checkpoint not found for {model} on {setup}")

# Generate bash commands with batches
for i, task in enumerate(tasks):
    model, setup, data_dir, ckpt = task
    gpu = available_gpus[gpu_idx % len(available_gpus)]
    gpu_idx += 1
    
    ckpt_arg = f"'{ckpt}'" if ckpt else ""
    script_content += f"run_eval_on_gpu {gpu} {model} {setup} {data_dir} {ckpt_arg} & \n"
    
    if gpu_idx % len(available_gpus) == 0:
        script_content += "wait\necho 'Batch finished, continuing...'\n\n"

script_content += "wait\necho 'All evaluations complete!'\n"

with open("run_evals_parallel.sh", "w") as f:
    f.write(script_content)

os.chmod("run_evals_parallel.sh", 0o755)

# Killing old session if exists
subprocess.run(["tmux", "kill-session", "-t", "downstream_eval_parallel"], stderr=subprocess.DEVNULL)

print("Generated run_evals_parallel.sh. Starting tmux session 'downstream_eval_parallel'...")
subprocess.run(["tmux", "new-session", "-d", "-s", "downstream_eval_parallel", "./run_evals_parallel.sh"])
print("Tmux session started. You can attach to it using: tmux attach-session -t downstream_eval_parallel")
