import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle

def analyze_relations():
    # 1. Load Model and Data
    ckpt_path = "outputs/analysis_feature_importance/checkpoints/default/best-epoch=02-val/loss=0.0954.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from models.joint.sepsis_model import JointSepsisModule
    model = JointSepsisModule.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()

    test_data = np.load("data/processed_sepsis_full/test.npz")
    X = torch.from_numpy(test_data["data"]).float().to(device)[:200]
    M = torch.from_numpy(test_data["orig_mask"]).float().to(device)[:200]
    
    # 2. Extract Relation Names
    emb_file = model.imputer.hparams.imputator_kwargs.get("kgi_embedding_file", "data/embeddings/medbert_relation_embeddings_sepsis_full.pkl")
    with open(emb_file, "rb") as f:
        medbert_dict = pickle.load(f)
    
    mapping = {
        220045: "Heart Rate", 220179: "SBP (NI)", 220180: "DBP (NI)", 220181: "MBP (NI)",
        220050: "SBP (I)", 220051: "DBP (I)", 220052: "MBP (I)", 220210: "Resp Rate",
        220277: "SpO2", 223762: "Temp", 226755: "GCS", 50931: "Glucose", 50912: "Creatinine",
        50885: "Bilirubin", 51006: "BUN", 51301: "WBC", 50983: "Sodium", 50971: "Potassium",
        51221: "Hematocrit", 50882: "Bicarb", 50813: "Lactate", 51265: "Platelets", 
        221906: "Norepi", 221289: "Epi", 222315: "Vaso", 221749: "Pheny", 221662: "Dopa"
    }
    
    relation_keys = list(medbert_dict.keys())
    relation_names = []
    for k in relation_keys:
        if isinstance(k, tuple):
            name1 = mapping.get(k[0], f"ID:{k[0]}")
            name2 = mapping.get(k[1], f"ID:{k[1]}")
            relation_names.append(f"{name1} -> {name2}")
        else:
            relation_names.append(mapping.get(k, f"ID:{k}"))

    # 3. Inference
    with torch.no_grad():
        surviving_mask = M.bool()
        x_proj = model.imputer.pos_encoding(model.imputer.input_proj(torch.cat([X * M, M], dim=-1)))
        
        # SSSD might call it differently, but for SAITS it is kgi_injector
        injector = getattr(model.imputer, "kgi_injector", None)
        if injector is None:
            # Fallback for dgi_mask mode where we might have missed instantiating it
            from models.saits.kgi_layer import DynamicKnowledgeInjector
            injector = DynamicKnowledgeInjector(text_embed_dim=768, hidden_dim=model.imputer.hparams.d_model).to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            injector_state = {k.replace("imputer.kgi_injector.", ""): v for k, v in ckpt["state_dict"].items() if "imputer.kgi_injector." in k}
            if injector_state: injector.load_state_dict(injector_state)
        
        # Get Keys
        text_keys = torch.stack(list(medbert_dict.values())).to(device)
        text_adapted = injector.text_adapter(text_keys)
        
        Q = injector.W_q(x_proj)
        K = injector.W_k(text_adapted)
        
        scores = torch.einsum('bth, kh -> btk', Q, K) / np.sqrt(injector.hidden_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        avg_attn = attn_weights.mean(dim=[0, 1]).cpu().numpy()

    # 4. Results
    df = pd.DataFrame({
        "relation": relation_names,
        "attention_weight": avg_attn
    }).sort_values("attention_weight", ascending=False)

    print("\n--- TOP 30 MEDICAL RELATIONS BY MODEL ATTENTION ---")
    print(df.head(30))
    
    plt.figure(figsize=(12, 14))
    plt.barh(df["relation"][:25][::-1], df["attention_weight"][:25][::-1], color='darkorange', alpha=0.8)
    plt.xlabel("Average Attention Score")
    plt.title("Knowledge Graph Interpretability: Top Attended Relations (SAITS DGI v2)")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/relation_importance_interpretability.png")

if __name__ == "__main__":
    analyze_relations()
