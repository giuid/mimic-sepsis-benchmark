import os
import shutil
from datetime import datetime

base_dir = "checkpoints/sepsis"
tasks = ["ihm", "los", "ss", "vr"]
subsets = ["full", "no_treatments", "core", "emergency"]
today = datetime.now().strftime("%Y%m%d")

# Create hierarchical structure
for task in tasks:
    for subset in subsets:
        os.makedirs(os.path.join(base_dir, task, subset), exist_ok=True)

# Move .pt files
files = [f for f in os.listdir(base_dir) if f.endswith(".pt")]
for f in files:
    parts = f.replace(".pt", "").split("_")
    # Expected format: transformer_{task}_{subset}_{injection}.pt or transformer_{task}_{injection}.pt
    # Some files like transformer_ihm_vanilla.pt don't have subset, we'll assume 'full'
    
    target_task = None
    target_subset = "full"
    target_inj = "vanilla"
    
    for t in tasks:
        if t in parts:
            target_task = t
            break
            
    for s in subsets:
        if s in parts:
            target_subset = s
            break
            
    if "dki" in parts:
        target_inj = "dki"
    elif "vanilla" in parts:
        target_inj = "vanilla"
        
    if target_task:
        new_name = f"transformer_{target_task}_{target_subset}_{target_inj}_{today}_migrated.pt"
        dest_path = os.path.join(base_dir, target_task, target_subset, new_name)
        shutil.move(os.path.join(base_dir, f), dest_path)
        print(f"Moved {f} -> {dest_path}")

# Move PyPOTS directories to a separate subfolder to keep root clean
pypots_dir = os.path.join(base_dir, "pypots")
os.makedirs(pypots_dir, exist_ok=True)
dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("sepsis_")]
for d in dirs:
    shutil.move(os.path.join(base_dir, d), os.path.join(pypots_dir, d))
    print(f"Moved directory {d} -> {pypots_dir}")
