import torch
import os
import pandas as pd

base_dir = "checkpoints/sepsis"
results = []

def inspect_pt(file_path):
    try:
        data = torch.load(file_path, map_location="cpu")
        config = data.get("config", {})
        sd = data.get("model_state_dict", data) # Some might be just the state_dict
        
        # Identification Logic
        task = config.get("task", "unknown")
        subset = config.get("feature_subset", "unknown")
        model = config.get("model", "transformer")
        use_kgi = config.get("use_kgi", False)
        
        # Check if DGI (Layer-wise) vs DKI (Input-level)
        # For Transformer: if 'layers.0.gate' exists, it's DGI.
        is_dgi = any("layers.0.gate" in k for k in sd.keys()) or any("gate" in k for k in sd.keys() if "shared_kgi" in k)
        is_dki = use_kgi and not is_dgi
        
        inj = "vanilla"
        if is_dgi: inj = "dgi"
        elif is_dki or use_kgi: inj = "dki"
        
        return {
            "path": file_path,
            "model": model,
            "task": task,
            "subset": subset,
            "injection": inj,
            "status": "ready"
        }
    except Exception as e:
        return {"path": file_path, "status": f"error: {str(e)}"}

# Scan PT files in subdirs
for root, dirs, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".pt"):
            res = inspect_pt(os.path.join(root, f))
            results.append(res)

# Scan PyPOTS (SAITS) - usually directory based with .ckpt
pypots_dir = os.path.join(base_dir, "pypots")
if os.path.exists(pypots_dir):
    for d in os.listdir(pypots_dir):
        path = os.path.join(pypots_dir, d)
        if os.path.isdir(path):
            # Parse directory name: sepsis_{task}_{model}_{inj}
            parts = d.split("_")
            task = parts[1] if len(parts) > 1 else "unknown"
            model = parts[2] if len(parts) > 2 else "saits"
            inj = parts[3] if len(parts) > 3 else "vanilla"
            results.append({
                "path": path,
                "model": model,
                "task": task,
                "subset": "full", # PyPOTS runs were mostly full
                "injection": inj,
                "status": "pypots_dir"
            })

df = pd.DataFrame(results)
print(df.to_string())
df.to_csv("results/checkpoints_inventory.csv", index=False)
