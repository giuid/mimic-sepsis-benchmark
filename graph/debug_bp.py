import os
import pickle

# Check what's in the pruned graph first
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

# If edges are 0, we need to know why
if all(len([e for e in data['edges'] if e[0] == cui or e[1] == cui]) == 0 for cui in bp_cuis):
    print("\n--- Deep Scan of MRREL for BP neighbors ---")
    UMLS_DIR = "/home/guido/Data/indus_data/ontologies/2025AB/META"
    MRREL_PATH = f"{UMLS_DIR}/MRREL.RRF"
    MRSTY_PATH = f"{UMLS_DIR}/MRSTY.RRF"
    
    bp_target = set(bp_cuis)
    neighbors = set()
    with open(MRREL_PATH, 'r') as f:
        for line in f:
            parts = line.split('|')
            if parts[0] in bp_target: neighbors.add(parts[4])
            elif parts[4] in bp_target: neighbors.add(parts[0])
    
    print(f"Found {len(neighbors)} unique neighbors in MRREL.")
    
    neighbor_types = {}
    with open(MRSTY_PATH, 'r') as f:
        for line in f:
            parts = line.split('|')
            if parts[0] in neighbors:
                if parts[0] not in neighbor_types: neighbor_types[parts[0]] = []
                neighbor_types[parts[0]].append(parts[1])
                
    from collections import Counter
    tui_counts = Counter([t for tuis in neighbor_types.values() for t in tuis])
    print("\nTop TUIs of neighbors:")
    for tui, count in tui_counts.most_common(10):
        print(f"{tui}: {count}")
