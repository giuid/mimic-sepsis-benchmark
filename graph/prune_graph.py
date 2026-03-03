
import pandas as pd
from tqdm import tqdm
import os
import pickle

# Config
UMLS_DIR = "/home/guido/Data/indus_data/ontologies/2025AB/META"
MRREL_PATH = f"{UMLS_DIR}/MRREL.RRF"
MRSTY_PATH = f"{UMLS_DIR}/MRSTY.RRF"
MRCONSO_PATH = f"{UMLS_DIR}/MRCONSO.RRF"
OUTPUT_DIR = "artifacts_pruned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target Semantic Types (TUIs)
TARGET_TUIS = {
    "T038": "Biologic Function",
    "T046": "Pathologic Function",
    "T059": "Laboratory Procedure",
    "T184": "Sign or Symptom",
    "T201": "Clinical Attribute",
    "T196": "Element, Ion, or Isotope",
    "T109": "Organic Chemical",
    "T116": "Amino Acid, Peptide, or Protein",
    "T121": "Pharmacologic Substance",
    "T123": "Biologically Active Substance",
    "T081": "Quantitative Concept",
    "T033": "Finding",
    "T032": "Organism Attribute",
    "T040": "Organism Function"
}

# Validated CUIs for 17 MIMIC items
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

def load_semantic_mapping():
    print("Loading MRSTY.RRF...")
    cui_to_tui = {}
    with open(MRSTY_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            parts = line.split('|')
            cui, tui = parts[0], parts[1]
            if tui in TARGET_TUIS:
                if cui not in cui_to_tui:
                    cui_to_tui[cui] = []
                cui_to_tui[cui].append(tui)
    return cui_to_tui

def prune_edges(cui_to_tui):
    print("Pruning MRREL.RRF based on semantic types and weights...")
    # Weights based on RELA (Relationship Attribute)
    # 1.0 = Causal (causes, precipitates, manifestation_of)
    # 0.8 = Physiological (associated_with, co-occurs_with)
    # 0.2 = Hierarchical (isa, part_of, inverse_isa)
    # Default = 0.5
    
    CAUSAL_RELS = {"causes", "precipitates", "manifestation_of", "method_of", "result_of"}
    PHYSIO_RELS = {"associated_with", "co-occurs_with", "process_of", "isa", "inverse_isa"} # Adjusted isa to 0.8? No, user said 0.2. Let's stick to plan.
    
    # Let's refine based on user request:
    # Causal = 1.0
    # Physio = 0.8
    # Hierarchical = 0.2
    
    def get_weight(rela, rel):
        rela = rela.lower() if rela else ""
        rel = rel.upper()
        
        if any(x in rela for x in ["cause", "precipit", "manifest", "result"]):
            return 1.0
        elif any(x in rela for x in ["associated", "occur", "process"]):
            return 0.8
        elif rel in ["CHD", "PAR", "ISA"]: # Hierarchical RELs often have empty RELA
            return 0.2
        elif "isa" in rela or "part_of" in rela:
            return 0.2
        else:
            return 0.5 # Unknown/Other

    pruned_adj = {}
    specific_relations = {} # (cui1, cui2) -> rela
    
    target_cuis_set = set(MIMIC_TARGETS.keys())
    semantic_cuis_set = set(cui_to_tui.keys())
    
    with open(MRREL_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing Edges"):
            parts = line.split('|')
            cui1, rel, cui2, rela, sab = parts[0], parts[3], parts[4], parts[7], parts[10]
            
            is_c1_relevant = (cui1 in target_cuis_set) or (cui1 in semantic_cuis_set)
            is_c2_relevant = (cui2 in target_cuis_set) or (cui2 in semantic_cuis_set)
            
            if is_c1_relevant and is_c2_relevant:
                w = get_weight(rela, rel)
                
                if cui1 not in pruned_adj: pruned_adj[cui1] = {}
                if cui2 not in pruned_adj: pruned_adj[cui2] = {}
                
                # Keep max weight
                if w >= pruned_adj[cui1].get(cui2, 0):
                    pruned_adj[cui1][cui2] = w
                    pruned_adj[cui2][cui1] = w
                    if rela or rel in ['PAR', 'CHD']:
                        # Standardize Hierarchy: Always Child -> Parent
                        if rel == 'PAR': # CUI2 is parent of CUI1
                            edge_key = f"{cui1}_{cui2}"
                            relation_text = "isa"
                        elif rel == 'CHD': # CUI2 is child of CUI1
                            edge_key = f"{cui2}_{cui1}"
                            relation_text = "isa"
                        else:
                            # Non-hierarchical: Use original direction CUI2 -> CUI1
                            edge_key = f"{cui2}_{cui1}"
                            relation_text = rela.replace("_", " ") if rela else "associated with"
                        
                        # Save the normalized specific relation, prioritizing high-fidelity or hierarchical ones
                        if edge_key not in specific_relations or relation_text == "isa":
                            specific_relations[edge_key] = relation_text

    return pruned_adj, specific_relations

def get_subgraph(adj, start_cuis, hops=2):
    print(f"Extracting {hops}-hop subgraph...")
    visited = set(start_cuis)
    current_layer = set(start_cuis)
    
    for _ in range(hops):
        next_layer = set()
        for node in current_layer:
            if node in adj:
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_layer.add(neighbor)
        current_layer = next_layer
        if not current_layer: break
        
    print(f"Subgraph size: {len(visited)} nodes.")
    return visited

def save_metadata(nodes, adj, specific_relations):
    print("Saving subgraph metadata...")
    import json
    cui_to_name = {}
    with open(MRCONSO_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Scanning Names"):
            parts = line.split('|')
            cui, lat, ts, name = parts[0], parts[1], parts[2], parts[14]
            if cui in nodes and lat == 'ENG':
                if cui not in cui_to_name or ts == 'P':
                    cui_to_name[cui] = name
                    
    final_edges = []
    final_specific_rels = {}
    for u in nodes:
        if u in adj:
            for v, w in adj[u].items():
                if v in nodes:
                    if u < v:
                        final_edges.append((u, v, w))
                        edge_key = (u, v)
                        if edge_key in specific_relations:
                            final_specific_rels[f"{u}_{v}"] = specific_relations[edge_key]
                        
    # Save pkl
    with open(f"{OUTPUT_DIR}/pruned_graph.pkl", 'wb') as f:
        pickle.dump({'nodes': nodes, 'names': cui_to_name, 'edges': final_edges}, f)
        
    # Save specific relations as JSON for MedBERT script
    with open(f"{OUTPUT_DIR}/specific_relations.json", 'w') as f:
        json.dump(final_specific_rels, f, indent=4)

    # Also save a CSV for inspection
    nodes_data = [{'cui': c, 'name': cui_to_name.get(c, 'N/A')} for c in nodes]
    pd.DataFrame(nodes_data).to_csv(f"{OUTPUT_DIR}/subgraph_nodes.csv", index=False)
    print(f"Saved {len(nodes)} nodes, {len(final_edges)} edges and specific relations to {OUTPUT_DIR}")

if __name__ == "__main__":
    cui_to_tui = load_semantic_mapping()
    adj, specific_relations = prune_edges(cui_to_tui)
    subgraph_nodes = get_subgraph(adj, list(MIMIC_TARGETS.keys()), hops=2)
    save_metadata(subgraph_nodes, adj, specific_relations)
