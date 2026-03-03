
import pickle
import pandas as pd
import numpy as np
import os

# Config
# Config
DATA_PATH = "graph/artifacts_pruned/pruned_graph.pkl"
OUTPUT_DIR = "graph/artifacts_pruned"

MIMIC_TARGETS = {
    "C0018810": "Heart Rate",
    "C0871470": "Systolic Blood Pressure",
    "C0428883": "Diastolic Blood Pressure",
    "C0231832": "Respiratory Rate",
    "C0483415": "Oxygen Saturation",
    "C0039476": "Temperature",
    "C0017725": "Glucose",
    "C0032821": "Potassium",
    "C0037473": "Sodium",
    "C0008203": "Chloride",
    "C0010294": "Creatinine",
    "C0005845": "BUN",
    "C0023516": "White Blood Cells",
    "C0005821": "Platelets",
    "C0019046": "Hemoglobin",
    "C0018935": "Hematocrit",
    "C0376261": "Lactate"
}

# Target Hub Types (TUIs) - verified from MRSTY search
# T038, T046, T059, T184
# We don't have TUIs in pruned_graph.pkl directly, but we can re-filter if needed
# OR we assume any non-target node in the pruned graph IS a semantic hub (since they were filtered by TUI).

def extract_metapaths():
    print(f"Loading pruned graph for meta-path extraction...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
        
    nodes = set(data['nodes'])
    edges = data['edges']
    target_cuis = list(MIMIC_TARGETS.keys())
    target_set = set(target_cuis)
    
    # Adjacency
    adj = {}
    adj = {}
    for edge in edges:
        if len(edge) == 3:
            u, v, w = edge
        else:
            u, v = edge
            
        if u not in adj: adj[u] = set()
        if v not in adj: adj[v] = set()
        adj[u].add(v)
        adj[v].add(u)
        
    n = len(target_cuis)
    # Meta-path matrix M: M[i, j] = count of paths Item_i -> Hub -> Item_j
    M = np.zeros((n, n))
    
    print("Counting Item -> Hub -> Item meta-paths...")
    for i in range(n):
        for j in range(i + 1, n):
            cui_i = target_cuis[i]
            cui_j = target_cuis[j]
            
            # Common neighbors that are NOT in the target set
            neighbors_i = adj.get(cui_i, set())
            neighbors_j = adj.get(cui_j, set())
            
            hubs = neighbors_i.intersection(neighbors_j) - target_set
            
            if hubs:
                M[i, j] = len(hubs)
                M[j, i] = len(hubs)
                
    # Normalize M by max count or log-scale
    print(f"Max meta-path count: {np.max(M)}")
    M_norm = np.log1p(M) # Use log1p for stability
    if np.max(M_norm) > 0:
        M_norm = M_norm / np.max(M_norm)
        
    # Save M
    np.save(f"{OUTPUT_DIR}/metapath_prior.npy", M_norm)
    
    # Save as CSV with labels
    labels = [MIMIC_TARGETS[cui] for cui in target_cuis]
    df_m = pd.DataFrame(M_norm, index=labels, columns=labels)
    df_m.to_csv(f"{OUTPUT_DIR}/metapath_prior.csv")
    
    print(f"Meta-path prior saved to {OUTPUT_DIR}/metapath_prior.npy")
    print("\nMeta-path Strength Sample (Top 5 items):")
    print(df_m.iloc[:5, :5])
    
    # Count how many pairs have at least one meta-path
    conn_pairs = np.sum(M > 0) // 2
    print(f"\nPairs connected via Meta-Paths: {conn_pairs} / {n*(n-1)//2}")

if __name__ == "__main__":
    extract_metapaths()
