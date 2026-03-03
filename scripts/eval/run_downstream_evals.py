import subprocess
import os
import concurrent.futures

runs = [
    {
        "name": "MRNN_KGI",
        "gpu": 0,
        "script": "scripts/evaluate_downstream.py",
        "args": [
            "--model_name", "mrnn_kgi",
            "--setup", "sota",
            "--data_dir", "data/sota",
            "--checkpoint", "/home/guido/Code/charite/baselines/outputs/mimic4/mrnn/random/2026-02-26_10-54-48/checkpoints/default_KGI/last.ckpt",
        ]
    },
    {
        "name": "BRITS_KGI",
        "gpu": 1,
        "script": "scripts/evaluate_downstream.py",
        "args": [
            "--model_name", "brits_kgi",
            "--setup", "sota",
            "--data_dir", "data/sota",
            "--checkpoint", "/home/guido/Code/charite/baselines/outputs/mimic4/brits/random/2026-02-26_10-54-48/checkpoints/default_KGI/last.ckpt",
        ]
    },
    {
        "name": "SAITS_SEM_GNN",
        "gpu": 2,
        "script": "scripts/evaluate_downstream.py",
        "args": [
            "--model_name", "saits_sem_gnn_kgi",
            "--setup", "sota",
            "--data_dir", "data/sota",
            "--checkpoint", "/home/guido/Code/charite/baselines/outputs/mimic4/saits/random/2026-02-26_10-54-45/checkpoints/vanilla_KGI/last-v1.ckpt"
        ]
    },
    {
        "name": "SAITS_SEM_NOGNN",
        "gpu": 3,
        "script": "scripts/evaluate_downstream.py",
        "args": [
            "--model_name", "saits_sem_nognn_kgi",
            "--setup", "sota",
            "--data_dir", "data/sota",
            "--checkpoint", "/home/guido/Code/charite/baselines/outputs/mimic4/saits/random/2026-02-26_10-54-45/checkpoints/vanilla_KGI/last.ckpt"
        ]
    },
    {
        "name": "SAITS_GEN_GNN",
        "gpu": 0, # Re-use GPU 0 as it finishes first
        "script": "scripts/evaluate_downstream.py",
        "args": [
            "--model_name", "saits_sem_gnn_kgi", # Keep model name to allow config matching if needed, though KGI is forced here. Wait! Evaluate downstream parses 'kgi'. Let's keep it safe.
            "--setup", "sota",
            "--data_dir", "data/sota",
            "--checkpoint", "/home/guido/Code/charite/baselines/outputs/mimic4/saits/random/2026-02-26_10-54-45/checkpoints/vanilla_KGI/last-v2.ckpt"
        ]
    }
]

def run_evaluation(run_config):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(run_config["gpu"])
    
    cmd = ["python", run_config["script"]] + run_config["args"]
    cmd_str = " ".join(cmd)
    
    print(f"[{run_config['name']}] Starting on GPU {run_config['gpu']}...")
    print(f"[{run_config['name']}] Command: {cmd_str}")
    
    try:
        log_file = f"eval_downstream_{run_config['name']}.log"
        with open(log_file, "w") as f:
            process = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
            process.wait()
            
        print(f"[{run_config['name']}] Finished. Exit code: {process.returncode}")
        return run_config['name'], process.returncode
    except Exception as e:
        print(f"[{run_config['name']}] Failed with error: {e}")
        return run_config['name'], -1

if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_evaluation, run) for run in runs]
        for future in concurrent.futures.as_completed(futures):
            name, code = future.result()
            print(f"Task completion: {name} (Exit code={code})")
