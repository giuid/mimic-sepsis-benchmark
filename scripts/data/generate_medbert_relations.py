import os
import torch
import itertools
from transformers import AutoTokenizer, AutoModel
import pickle
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

def create_verbalization(label_a: str, label_b: str, relationship: str = "associated with") -> str:
    """
    Creates a templated clinical sentence associating two features.
    """
    template = f"Clinical knowledge indicates that {label_a} is {relationship} {label_b}."
    return template

# Natural Language Mapping for technical UMLS relations (RELA)
RELA_MAP = {
    "isa": "is a type of",
    "inverse isa": "is a category that includes",
    "has component": "is composed of",
    "component of": "is a component of",
    "has scale": "is measured on the base scale of",
    "has scale type": "uses the measurement scale",
    "divisor of": "is the divisor for",
    "has measurement unit": "is measured in",
    "ssc": "is a sign or symptom of",
    "has manifestation": "is manifested by",
    "is manifestation of": "is a manifestation of",
    "causes": "causes",
    "precipitates": "precipitates",
    "result of": "is a result of",
    "has property": "has the property of",
    "is interpreted by": "can be interpreted using",
    "may treat": "may be used to treat",
    "may be treated by": "is treated with",
    "clinically associated with": "is clinically associated with",
    "co-occurs with": "co-occurs with",
    "member of": "is a member of",
    "has active ingredient": "contains the active ingredient",
    "mapped from": "is derived from",
}

def generate_medbert_relations(vocab_df: pd.DataFrame, specific_rels: dict, graph_data: dict, batch_size: int = 64):
    """
    Generates pairwise relation embeddings using specific UMLS relationships and 2-hop paths.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Graph extraction
    names = graph_data['names']
    edges = graph_data['edges']
    adj = {}
    for u, v, w in edges:
        if u not in adj: adj[u] = []
        if v not in adj: adj[v] = []
        adj[u].append(v)
        adj[v].append(u)

    labels = vocab_df['label'].tolist()
    itemids = vocab_df['itemid'].tolist()
    
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # In this phase, we process relations for all 17 target features
    top_n = len(labels)
    pairs = list(itertools.combinations(range(top_n), 2))
    print(f"Building high-fidelity templates for {len(pairs)} pairwise relationships...")
    
    sentences = []
    pair_keys = []
    audit_data = []
    
    MIMIC_TARGETS_INVERTED = {
        "Heart Rate": "C0018810",
        "Systolic Blood Pressure": "C0871470",
        "Diastolic Blood Pressure": "C0428883",
        "Respiratory Rate": "C0231832",
        "Oxygen Saturation": "C0483415",
        "Temperature": "C0039476",
        "Glucose": "C0017725",
        "Potassium": "C0032821",
        "Sodium": "C0037473",
        "Chloride": "C0008203",
        "Creatinine": "C0010294",
        "BUN": "C0005845",
        "White Blood Cells": "C0023516",
        "Platelets": "C0005821",
        "Hemoglobin": "C0019046",
        "Hematocrit": "C0018935",
        "Lactate": "C0376261",
        "Glasgow Coma Scale": "C0017084",
        "Urine Output": "C0042034",
        "Norepinephrine": "C0028351",
        "IV Fluids": "C0021099",
        "Mechanical Ventilation": "C0035222"
    }

    sapbert_model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    print(f"Loading {sapbert_model_name} for Semantic Path Selection...")
    sapbert_tokenizer = AutoTokenizer.from_pretrained(sapbert_model_name)
    sapbert_model = AutoModel.from_pretrained(sapbert_model_name).to(device)
    sapbert_model.eval()

    def get_sapbert_embedding(text):
        with torch.no_grad():
            inputs = sapbert_tokenizer([text], padding=True, truncation=True, return_tensors='pt').to(device)
            outputs = sapbert_model(**inputs)
            # Use [CLS] token embedding
            return outputs.last_hidden_state[:, 0, :]

    def get_natural_relation(rel_str, is_inverse=False):
        # 1. Mapping per RELA (Specifico)
        rela_map = {
            "treats": ("treats", "is treated by"),
            "causes": ("causes", "is caused by"),
            "location of": ("is the location of", "is located in"),
            "isa": ("is a type of", "is a category for"),
            "inverse isa": ("is a category for", "is a type of"),
            "ssc": ("is a subtype of", "subsumes"),
            "has component": ("is composed of", "is a component of"),
            "clinically associated with": ("is clinically associated with", "is clinically associated with"),
        }
        
        if rel_str in rela_map:
            return rela_map[rel_str][1] if is_inverse else rela_map[rel_str][0]
        
        return f"is associated with ({rel_str})" if not is_inverse else f"is associated with (via {rel_str})"

    def find_clinical_path(cui_a, cui_b, label_a, label_b):
        # 1. Check direct (Directed CUI_A -> CUI_B)
        k_ab = f"{cui_a}_{cui_b}"
        k_ba = f"{cui_b}_{cui_a}"
        
        if k_ab in specific_rels:
            return get_natural_relation(specific_rels[k_ab], is_inverse=False)
        if k_ba in specific_rels:
            return get_natural_relation(specific_rels[k_ba], is_inverse=True)
        
        # 2. Check 2-hop
        n1 = set(adj.get(cui_a, []))
        n2 = set(adj.get(cui_b, []))
        common = list(n1.intersection(n2))
        
        if common:
            # Semantic Selection via SapBERT
            best_mid = common[0]
            if len(common) > 1:
                emb_a = get_sapbert_embedding(label_a)
                emb_b = get_sapbert_embedding(label_b)
                
                best_score = -1.0
                for candidate in common:
                    candidate_name = names.get(candidate, "")
                    if not candidate_name: continue
                    
                    emb_mid = get_sapbert_embedding(candidate_name)
                    # Compute average cosine similarity to both endpoints
                    sim_a = torch.nn.functional.cosine_similarity(emb_a, emb_mid).item()
                    sim_b = torch.nn.functional.cosine_similarity(emb_b, emb_mid).item()
                    avg_sim = (sim_a + sim_b) / 2.0
                    
                    if avg_sim > best_score:
                        best_score = avg_sim
                        best_mid = candidate
            else:
                best_mid = common[0]
                
            mid = best_mid
            mid_name = names.get(mid, "clinical concept")
            
            # Relation A -> Mid
            if f"{cui_a}_{mid}" in specific_rels:
                p1 = get_natural_relation(specific_rels[f"{cui_a}_{mid}"], is_inverse=False)
            elif f"{mid}_{cui_a}" in specific_rels:
                p1 = get_natural_relation(specific_rels[f"{mid}_{cui_a}"], is_inverse=True)
            else:
                p1 = "is associated with"
                
            # Relation Mid -> B
            if f"{mid}_{cui_b}" in specific_rels:
                p2 = get_natural_relation(specific_rels[f"{mid}_{cui_b}"], is_inverse=False)
            elif f"{cui_b}_{mid}" in specific_rels:
                p2 = get_natural_relation(specific_rels[f"{cui_b}_{mid}"], is_inverse=True)
            else:
                p2 = "is associated with"
            
            return f"{p1} {mid_name}, which {p2 or 'is associated with'}"
            
        return "is associated with"

    for idx_a, idx_b in pairs:
        label_a, label_b = labels[idx_a], labels[idx_b]
        id_a, id_b = itemids[idx_a], itemids[idx_b]
        
        cui_a = MIMIC_TARGETS_INVERTED.get(label_a)
        cui_b = MIMIC_TARGETS_INVERTED.get(label_b)
        
        rel_desc = "is associated with"
        if cui_a and cui_b:
            rel_desc = find_clinical_path(cui_a, cui_b, label_a, label_b)
            
        text = f"Clinical knowledge indicates that {label_a} {rel_desc} {label_b}."
        sentences.append(text)
        
        pair_key = tuple(sorted([id_a, id_b]))
        pair_keys.append(pair_key)
        
        audit_data.append({
            "itemid_a": id_a, "itemid_b": id_b,
            "label_a": label_a, "label_b": label_b,
            "relationship_path": rel_desc,
            "final_sentence": text
        })

    # Encode in batches
    precomputed_dict = {}
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch_text = sentences[i:i+batch_size]
        batch_keys = pair_keys[i:i+batch_size]
        
        inputs = tokenizer(batch_text, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_emb = outputs.last_hidden_state[:, 0, :].cpu()
            
        for k, emb in zip(batch_keys, batch_emb):
            precomputed_dict[k] = emb

    # Save Audit Log
    audit_df = pd.DataFrame(audit_data)
    audit_df.to_csv('data/embeddings/medbert_input_audit.csv', index=False)
    print(f"Saved audit log to data/embeddings/medbert_input_audit.csv")

    return precomputed_dict

if __name__ == '__main__':
    SPECIFIC_REL_PATH = 'graph/artifacts_pruned/specific_relations.json'
    GRAPH_PKL_PATH = 'graph/artifacts_pruned/pruned_graph.pkl'
    OUT_DIR = 'data/embeddings/'
    VOCAB_CSV = os.path.join(OUT_DIR, 'mimic_vocab_mapped.csv')
    
    with open(SPECIFIC_REL_PATH, 'r') as f:
        specific_rels = json.load(f)
    print(f"Loaded {len(specific_rels)} specific UMLS relationships.")
    
    with open(GRAPH_PKL_PATH, 'rb') as f:
        graph_data = pickle.load(f)
        
    df_vocab = pd.read_csv(VOCAB_CSV)
    
    # Target 22 Sepsis Benchmark features (Filtering by ItemID to be precise)
    target_itemids = [
        220045, 220210, 220277, 223762, 220179, 220180, 220052, 50931, 50813, 50912, 
        50885, 51265, 51301, 223835, 50820, 50821, 50818, 220739, 226559, 221906, 
        225158, 223848
    ]
    
    df_targets = df_vocab[df_vocab['itemid'].isin(target_itemids)].copy()
    print(f"Filtered to {len(df_targets)} target features by ItemID.")
    
    relation_embeddings = generate_medbert_relations(df_targets, specific_rels, graph_data)
    
    out_file = os.path.join(OUT_DIR, 'medbert_relation_embeddings_sepsis.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(relation_embeddings, f)
    print(f"Saved {len(relation_embeddings)} relation embeddings to {out_file}")
    with open(out_file, 'wb') as f:
        pickle.dump(relation_embeddings, f)
        
    print(f"Successfully saved high-fidelity relation embeddings to {out_file}")
