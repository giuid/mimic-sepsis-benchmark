import subprocess
import os
import concurrent.futures

# Configuration for the 5 runs
runs = [
    {
        "name": "MRNN_KGI",
        "gpu": 0,
        "script": "scripts/eval_kgi.py",
        "args": [
            "--model_name", "MRNN_KGI",
            "--ckpt_path", "/home/guido/Code/charite/baselines/outputs/mimic4/mrnn/random/2026-02-26_10-54-48/checkpoints/default_KGI/best-epoch=80-val/loss=0.0997.ckpt", # Update after glob
        ]
    },
    {
        "name": "BRITS_KGI",
        "gpu": 1,
        "script": "scripts/eval_kgi.py",
        "args": [
            "--model_name", "BRITS_KGI",
            "--ckpt_path", "/home/guido/Code/charite/baselines/outputs/mimic4/brits/random/2026-02-26_10-54-48/checkpoints/default_KGI/best-epoch=79-val/loss=0.1590.ckpt", # Update after glob
        ]
    },
    {
        "name": "SAITS_SEM_GNN",
        "gpu": 2,
        "script": "scripts/eval_kgi.py",
        "args": [
            "--model_name", "SAITS_SEM_GNN",
            "--ckpt_path", "/home/guido/Code/charite/baselines/outputs/mimic4/saits/random/2026-02-26_10-54-45/checkpoints/vanilla_KGI/best-epoch=73-val/loss=0.3137.ckpt" # Update after glob
        ]
    },
    {
        "name": "SAITS_SEM_NOGNN",
        "gpu": 3,
        "script": "scripts/eval_kgi.py",
        "args": [
            "--model_name", "SAITS_SEM_NOGNN",
            "--ckpt_path", "/home/guido/Code/charite/baselines/outputs/mimic4/saits/random/2026-02-26_10-54-45/checkpoints/vanilla_KGI/best-epoch=78-val/loss=0.2725.ckpt" # Update after glob
        ]
    },
    {
        "name": "SAITS_GEN_GNN",
        "gpu": 0, # Re-use GPU 0 as it finishes first
        "script": "scripts/eval_kgi.py",
        "args": [
            "--model_name", "SAITS_GEN_GNN",
            "--ckpt_path", "/home/guido/Code/charite/baselines/outputs/mimic4/saits/random/2026-02-26_10-54-45/checkpoints/vanilla_KGI/best-epoch=72-val/loss=0.3136.ckpt" # Update after glob
        ]
    }
]

import glob

def find_latest_best_ckpt(base_path, prefix="best-epoch="):
    """Finds the checkpoint with the highest epoch number."""
    search_path = os.path.join(base_path, prefix + "*.ckpt")
    ckpts = glob.glob(search_path)
    if not ckpts:
        print(f"No ckpts found at {search_path}")
        return None
    
    def extract_epoch(path):
        filename = os.path.basename(path)
        try:
            # e.g. best-epoch=80-val/loss=0.0997.ckpt -> 80
            epoch_str = filename.split("epoch=")[1].split("-")[0]
            return int(epoch_str)
        except Exception:
            return -1
    
    return max(ckpts, key=extract_epoch)

def get_base_dir_from_pane(pane_output):
    # Just a helper if we needed to parse tmux.
    pass

# We can manually assign paths to runs because we just queried them.
# The `outputs` path contains the ckpt
# I will use the `find_latest_best_ckpt` to robustly get the latest.
# Let's verify paths

run_paths = [
     "/home/guido/Code/charite/baselines/outputs/mimic4/mrnn/random/2026-02-26_10-54-48/checkpoints/default_KGI",
     "/home/guido/Code/charite/baselines/outputs/mimic4/brits/random/2026-02-26_10-54-48/checkpoints/default_KGI",
     # We have 3 SAITS runs. They all seem to save into `/home/guido/Code/charite/baselines/outputs/mimic4/saits/random/2026-02-26_10-54-45/checkpoints/vanilla_KGI/` ?
     # Wait, this might be a problem. Did PyTorch Lightning overwrite them? Let's check the run names in that folder.
     # I'll rely on the user's provided run paths or glob them.
]


def run_evaluation(run_config):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(run_config["gpu"])
    
    cmd = ["python", run_config["script"]] + run_config["args"]
    cmd_str = " ".join(cmd)
    
    print(f"[{run_config['name']}] Starting on GPU {run_config['gpu']}...")
    print(f"[{run_config['name']}] Command: {cmd_str}")
    
    try:
        # Capture output, log to a file
        log_file = f"eval_imputation_{run_config['name']}.log"
        with open(log_file, "w") as f:
            process = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
            process.wait()
            
        print(f"[{run_config['name']}] Finished. Exit code: {process.returncode}")
        return run_config['name'], process.returncode
    except Exception as e:
        print(f"[{run_config['name']}] Failed with error: {e}")
        return run_config['name'], -1

if __name__ == "__main__":
    # We first need to correctly map the ckpts to the 3 different SAITS models.
    # The output we saw was:
    # best-epoch=78-val/loss=0.2725.ckpt
    # best-epoch=72-val/loss=0.3136.ckpt
    # best-epoch=73-val/loss=0.3137.ckpt
    #
    # SAITS_SEM_NOGNN has val/loss 0.273 (best) -> so epoch=78 is SAITS_SEM_NOGNN
    # SAITS_SEM_GNN has val/loss 0.314 (best) -> so epoch=73 is SAITS_SEM_GNN
    # SAITS_GEN_GNN has val/loss 0.314 (best) -> so epoch=72 is SAITS_GEN_GNN
    
    # We override the ckpt paths inline now:
    runs[2]["args"][-1] = "/home/guido/Code/charite/baselines/outputs/mimic4/saits/random/2026-02-26_10-54-45/checkpoints/vanilla_KGI/best-epoch=73-val/loss=0.3137.ckpt" # SEM_GNN
    runs[3]["args"][-1] = "/home/guido/Code/charite/baselines/outputs/mimic4/saits/random/2026-02-26_10-54-45/checkpoints/vanilla_KGI/best-epoch=78-val/loss=0.2725.ckpt" # SEM_NOGNN
    runs[4]["args"][-1] = "/home/guido/Code/charite/baselines/outputs/mimic4/saits/random/2026-02-26_10-54-45/checkpoints/vanilla_KGI/best-epoch=72-val/loss=0.3136.ckpt" # GEN_GNN
    
    # We also do not pass --use_kgi to GEN_GNN
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_evaluation, run) for run in runs]
        for future in concurrent.futures.as_completed(futures):
            name, code = future.result()
            print(f"Task completion: {name} (Exit code={code})")
