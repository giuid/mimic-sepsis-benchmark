import pickle
import os

with open('artifacts_pruned/pruned_graph.pkl', 'rb') as f:
    data = pickle.load(f)

bp_cuis = ["C0871470", "C0428883"]
for cui in bp_cuis:
    exists = cui in data['nodes']
    name = data['names'].get(cui, "N/A")
    print(f"CUI {cui} ({name}) in graph nodes: {exists}")
    
    # Check edges for this CUI
    cui_edges = [e for e in data['edges'] if e[0] == cui or e[1] == cui]
    print(f"Edges for {cui}: {len(cui_edges)}")
    if cui_edges:
        print(f"Sample edge: {cui_edges[0]}")

# Count total nodes and edges
print(f"\nTotal nodes: {len(data['nodes'])}")
print(f"Total edges: {len(data['edges'])}")
