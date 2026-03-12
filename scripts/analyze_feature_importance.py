import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze():
    # 1. Load the model and data
    ckpt_path = "outputs/analysis_feature_importance/checkpoints/default/best-epoch=02-val/loss=0.0954.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from models.joint.sepsis_model import JointSepsisModule
    model = JointSepsisModule.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()

    test_data = np.load("data/processed_sepsis_full/test.npz")
    X = torch.from_numpy(test_data["data"]).float().to(device)[:200]
    M = torch.from_numpy(test_data["orig_mask"]).float().to(device)[:200]
    
    # 2. Extract Weights
    with torch.no_grad():
        surviving_mask = M.bool()
        x_proj = model.imputer.pos_encoding(model.imputer.input_proj(torch.cat([X * M, M], dim=-1)))
        
        from models.saits.kgi_layer import DynamicKnowledgeInjector
        injector = DynamicKnowledgeInjector(text_embed_dim=768, hidden_dim=model.imputer.hparams.d_model).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        injector_state = {k.replace("imputer.kgi_injector.", ""): v for k, v in ckpt["state_dict"].items() if "imputer.kgi_injector." in k}
        if injector_state: injector.load_state_dict(injector_state)
        
        f_emb = injector(x_proj, surviving_mask, model.imputer.medbert_dict, model.imputer.kgi_itemids)
        if f_emb.dim() == 3:
            f_emb = f_emb.unsqueeze(2).expand(-1, -1, 55, -1)
        
        block = model.imputer.dmsa_block_1
        x_feat = torch.stack([X * M, M], dim=-1)
        x = block.feat_proj(x_feat) 
        
        all_weights = []
        m_exp = M.unsqueeze(-1)
        
        for i, layer in enumerate(block.layers):
            B, T, D, H = x.shape
            x_in = x.transpose(1, 2).reshape(B*D, T, H)
            x_temp, _ = layer['temporal'](x_in, x_in, x_in)
            x = x + x_temp.reshape(B, D, T, H).transpose(1, 2)
            
            gate = block.feature_gates[i]
            concat = torch.cat([x * m_exp, m_exp, f_emb], dim=-1)
            w = gate.gate_net(concat) 
            all_weights.append(w)
            x = (1.0 - w) * x + w * f_emb
            x = x + layer['ffn'](x)

        # Final average weights per feature [D]
        avg_weights = torch.stack(all_weights).mean(dim=[0, 1, 2, 4]).cpu().numpy()

    # 3. Features (Standard 55 order from mimic4_sepsis_full.yaml)
    # Recovering actual names from extraction logic
    feature_names = [
        "heart_rate", "sbp_ni", "dbp_ni", "mbp_ni", "sbp_i", "dbp_i", "mbp_i", "resp_rate", "spo2", "temp", "gcs",
        "glucose", "creatinine", "bilirubin", "bun", "wbc", "sodium", "potassium", "hct", "bicarb", "lactate",
        "platelets", "inr", "pt", "ptt", "alt", "ast", "alp", "albumin", "calcium", "glucose_lab", "chloride",
        "magnesium", "phosphate", "be", "pco2", "ph", "po2", "fio2", "pao2_fio2", "urine_output",
        "norepi", "epi", "vaso", "pheny", "dopa", "fluid_volume", "vent_status", "age", "gender", "weight"
    ]
    # Padding if needed
    if len(feature_names) < len(avg_weights):
        feature_names += [f"feat_{i}" for i in range(len(avg_weights) - len(feature_names))]
    
    df = pd.DataFrame({
        "feature": feature_names[:len(avg_weights)],
        "semantic_importance": avg_weights
    }).sort_values("semantic_importance", ascending=False)

    print("\n--- CLINICAL FEATURE RANKING BY SEMANTIC RELIANCE (SAITS DGI v2) ---")
    print(df.head(30))
    
    plt.figure(figsize=(10, 12))
    top_n = 25
    plt.barh(df["feature"][:top_n][::-1], df["semantic_importance"][:top_n][::-1], color='teal', alpha=0.8)
    plt.xlabel("Average Semantic Gate Weight (alpha)")
    plt.title("Clinical Feature Relevance: Semantic Fallback Intensity")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/feature_importance_semantics.png")
    print("\nSuccess. Results printed and plot saved to results/feature_importance_semantics.png")

if __name__ == "__main__":
    analyze()
