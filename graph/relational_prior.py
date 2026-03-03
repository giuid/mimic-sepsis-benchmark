
import pickle
import networkx as nx
import numpy as np
import pandas as pd
import os

# Config
DATA_PATH = "artifacts_pruned/pruned_graph.pkl"
OUTPUT_DIR = "artifacts_pruned"

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

def calculate_prior():
    print(f"Loading pruned graph from {DATA_PATH}...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
        
    G = nx.Graph()
    G.add_nodes_from(data['nodes'])
    
    # Edges are now (u, v, w)
    # Convert weight w (relevance) to cost (distance) for Dijkstra
    # Cost = 1 / w
    # w=1.0 -> cost=1.0
    # w=0.8 -> cost=1.25
    # w=0.2 -> cost=5.0
    
    edges_with_cost = []
    if len(data['edges']) > 0 and len(data['edges'][0]) == 3:
        for u, v, w in data['edges']:
            cost = 1.0 / w
            edges_with_cost.append((u, v, cost))
        G.add_weighted_edges_from(edges_with_cost, weight='weight')
        print("Loaded weighted edges.")
    else:
        # Fallback for old format
        print("Warning: Old edge format detected. Using default weights.")
        G.add_edges_from(data['edges'])
    
    print(f"Graph stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    # 17 Targets
    target_cuis = list(MIMIC_TARGETS.keys())
    n = len(target_cuis)
    
    # Distance matrix
    dist_matrix = np.full((n, n), np.inf)
    
    print("Computing weighted shortest paths between targets...")
    for i in range(n):
        for j in range(i, n):
            cui1, cui2 = target_cuis[i], target_cuis[j]
            if cui1 == cui2:
                dist_matrix[i, j] = 0
            else:
                try:
                    # Use 'weight' as the cost attribute
                    d = nx.shortest_path_length(G, cui1, cui2, weight='weight')
                    dist_matrix[i, j] = d
                    dist_matrix[j, i] = d
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
                    
    print("\nDistance Matrix (Hops):")
    # Replace inf with a large value for prior calculation (e.g. 10 hops)
    max_d = np.max(dist_matrix[np.isfinite(dist_matrix)]) if np.any(np.isfinite(dist_matrix)) else 10
    dist_matrix_filled = np.where(np.isinf(dist_matrix), max_d + 1, dist_matrix)
    
    # Calculate Relational Prior P
    # P_ij = 1 / (1 + dist)
    P = 1.0 / (1.0 + dist_matrix_filled)
    
    # Save P
    np.save(f"{OUTPUT_DIR}/relational_prior.npy", P)
    
    # Save as CSV with labels for inspection
    labels = [MIMIC_TARGETS[cui] for cui in target_cuis]
    df_p = pd.DataFrame(P, index=labels, columns=labels)
    df_p.to_csv(f"{OUTPUT_DIR}/relational_prior.csv")
    
    print(f"Relational prior matrix saved to {OUTPUT_DIR}/relational_prior.npy")
    print("\nRelational Prior Sample (Top 5 items):")
    print(df_p.iloc[:5, :5])
    
    # Check connectivity
    connected_pairs = np.sum(np.isfinite(dist_matrix)) - n
    print(f"\nConnected pairs (excluding self): {connected_pairs // 2} / {n*(n-1)//2}")

if __name__ == "__main__":
    calculate_prior()
