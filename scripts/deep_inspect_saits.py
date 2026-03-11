import torch
import os
import pandas as pd

base_pypots = "checkpoints/sepsis/pypots"
results = []

def inspect_ckpt(path):
    # Find the best .ckpt in the folder
    ckpt_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".ckpt"):
                ckpt_files.append(os.path.join(root, f))
    
    if not ckpt_files:
        return {"path": path, "status": "no .ckpt found"}
    
    # Check the latest/best one
    ckpt_path = ckpt_files[0] 
    try:
        # Load state_dict (map_location='cpu' for safety)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        sd = checkpoint.get("state_dict", checkpoint)
        
        # Identification
        is_graph = any("dmsa_block_1.feature_gates" in k for k in sd.keys())
        is_vanilla = not any("kgi" in k or "fusion" in k or "feature_gates" in k for k in sd.keys())
        
        inj = "dki"
        if is_graph: inj = "dgi"
        elif is_vanilla: inj = "vanilla"
        
        return {
            "path": path,
            "injection": inj,
            "status": "ready"
        }
    except Exception as e:
        return {"path": path, "status": f"error: {str(e)}"}

if os.path.exists(base_pypots):
    for d in os.listdir(base_pypots):
        res = inspect_ckpt(os.path.join(base_pypots, d))
        res["folder"] = d
        results.append(res)

df = pd.DataFrame(results)
print(df.to_string())
