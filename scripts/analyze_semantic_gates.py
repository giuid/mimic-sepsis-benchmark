import torch
import pickle
import numpy as np
import pandas as pd
import os

# Load the best SAITS DGI model (Emergency IHM)
task = "ihm"
subset = "emergency"
ckpt_dir = f"checkpoints/sepsis/{task}/{subset}"
files = [f for f in os.listdir(ckpt_dir) if "dgi" in f] if os.path.exists(ckpt_dir) else []
if not files:
    print("No DGI checkpoint found for analysis.")
    exit()

ckpt_path = os.path.join(ckpt_dir, files[0])
data = torch.load(ckpt_path, map_location="cpu")
sd = data.get("model_state_dict", data)

with open("data/embeddings/medbert_relation_embeddings_sepsis_full.pkl", "rb") as f:
    rel_dict = pickle.load(f)

results = []
for (id1, id2), embed in rel_dict.items():
    mag = np.linalg.norm(embed)
    results.append({"id1": id1, "id2": id2, "magnitude": mag})

df_rel = pd.DataFrame(results).sort_values(by="magnitude", ascending=False)

mapping = {
    220045: "Heart Rate", 50813: "Lactate", 50912: "Creatinine", 
    220210: "Resp Rate", 223762: "Temperature", 50820: "pH",
    51265: "Platelets", 220277: "SpO2", 220052: "MAP"
}

df_rel['name1'] = df_rel['id1'].map(mapping)
df_rel['name2'] = df_rel['id2'].map(mapping)

print("--- Top 10 Semantic Relations (SapBERT/MedBERT Influence) ---")
print(df_rel.dropna().head(10)[['name1', 'name2', 'magnitude']])
