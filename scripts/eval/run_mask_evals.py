import subprocess
import os
import concurrent.futures

runs = [
    {
        "name": "gpvae",
        "gpu": 0,
        "script": "scripts/evaluate_downstream.py",
        "args": [
            "--model_name", "gpvae",
            "--setup", "sota",
            "--data_dir", "data/sota",
            "--checkpoint", "/home/guido/Code/charite/baselines/outputs/sota/gpvae/random/D17_2026-02-19_15-17-02/checkpoints/last.ckpt"
        ]
    },
    {
        "name": "mrnn_vanilla",
        "gpu": 1,
        "script": "scripts/evaluate_downstream.py",
        "args": [
            "--model_name", "mrnn",
            "--setup", "sota",
            "--data_dir", "data/sota",
            "--checkpoint", "/home/guido/Code/charite/baselines/outputs/sota/mrnn/random/D17_2026-02-19_15-22-28/checkpoints/last.ckpt"
        ]
    },
    {
        "name": "brits_vanilla",
        "gpu": 2,
        "script": "scripts/evaluate_downstream.py",
        "args": [
            "--model_name", "brits",
            "--setup", "sota",
            "--data_dir", "data/sota",
            "--checkpoint", "/home/guido/Code/charite/baselines/outputs/sota/brits/random/D17_2026-02-19_15-22-24/checkpoints/last.ckpt"
        ]
    },
    {
        "name": "saits_vanilla",
        "gpu": 3,
        "script": "scripts/evaluate_downstream.py",
        "args": [
            "--model_name", "saits_vanilla",
            "--setup", "sota",
            "--data_dir", "data/sota",
            "--checkpoint", "/home/guido/Code/charite/baselines/outputs/sota/saits/random/D17_2026-02-19_15-11-55/checkpoints/last.ckpt"
        ]
    },
    {
        "name": "timesfm",
        "gpu": 0,
        "script": "scripts/evaluate_downstream.py",
        "args": [
            "--model_name", "timesfm",
            "--setup", "sota",
            "--data_dir", "data/sota",
            "--checkpoint", "/home/guido/Code/charite/baselines/outputs/sota/timesfm/random/D17_2026-02-23_12-57-43/checkpoints/last.ckpt"
        ]
    },
    {
        "name": "timesfm_sapbert",
        "gpu": 1,
        "script": "scripts/evaluate_downstream.py",
        "args": [
            "--model_name", "timesfm_sapbert",
            "--setup", "sota",
            "--data_dir", "data/sota",
            "--checkpoint", "/home/guido/Code/charite/baselines/outputs/sota/timesfm/sapbert/D17_2026-02-23_12-57-41/checkpoints/last.ckpt"
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
        log_file = f"eval_downstream_mask_{run_config['name']}.log"
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
