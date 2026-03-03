import subprocess
import os

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "3"

cmd = [
    "python", "scripts/evaluate_downstream.py",
    "--model_name", "timesfm",
    "--setup", "sota",
    "--data_dir", "data/sota",
    "--checkpoint", "/home/guido/Code/charite/baselines/outputs/sota/timesfm/random/D17_2026-02-23_12-57-43/checkpoints/last.ckpt"
]

print("Starting TimesFM Vanilla on GPU 3...")
with open("eval_downstream_mask_timesfm_retry.log", "w") as f:
    process = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    process.wait()
    
print(f"Finished. Exit code: {process.returncode}")
