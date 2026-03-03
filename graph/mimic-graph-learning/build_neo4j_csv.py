
import pandas as pd
import os

INPUT_HIERARCHY = "artifacts_step7/mimic_hierarchy.csv"
OUTPUT_DIR = "artifacts_step7/neo4j_import"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_neo4j_csv():
    if not os.path.exists(INPUT_HIERARCHY):
        print(f"Input {INPUT_HIERARCHY} not found. Run step 7.3 first.")
        return

    df = pd.read_csv(INPUT_HIERARCHY)
    
    # Lists for Nodes and Edges
    nodes = []
    edges = []
    
    # Sets to track uniqueness
    seen_nodes = set()
    
    # Load CUI Names
    cui_to_name = {}
    names_path = "artifacts_step7/cui_names.csv"
    if os.path.exists(names_path):
        names_df = pd.read_csv(names_path)
        cui_to_name = pd.Series(names_df.name.values, index=names_df.cui).to_dict()
    else:
        print(f"Warning: {names_path} not found. Using placeholders.")
    
    # 1. MIMIC Item Nodes
    for label in df['mimic_label'].unique():
        node_id = f"MIMIC:{label}"
        if node_id not in seen_nodes:
            nodes.append({
                'nodeId': node_id,
                ':LABEL': 'MimicItem',
                'name': label,
                'type': 'source_data'
            })
            seen_nodes.add(node_id)
            
    # 2. UMLS Concept Nodes & Edges
    for _, row in df.iterrows():
        mimic_label = row['mimic_label']
        cui = row['cui']
        match_str = row['match_str']
        
        cui_node_id = f"CUI:{cui}"
        mimic_node_id = f"MIMIC:{mimic_label}"
        
        # CUI Node
        if cui_node_id not in seen_nodes:
            nodes.append({
                'nodeId': cui_node_id,
                ':LABEL': 'UMLSConcept',
                'name': match_str,
                'cui': cui,
                'type': 'ontology'
            })
            seen_nodes.add(cui_node_id)
        
        # Edge: MIMIC -> CUI
        edges.append({
            ':START_ID': mimic_node_id,
            ':END_ID': cui_node_id,
            ':TYPE': 'MAPPED_TO',
            'weight': 1.0 
        })
        
        # 3. Hierarchy (Parents)
        if pd.notna(row['parents_cui']):
            parents = str(row['parents_cui']).split(';')
            for parent_cui in parents:
                if not parent_cui: continue
                parent_node_id = f"CUI:{parent_cui}"
                
                # We might not have the parent's string name if it wasn't in our SapBERT search results
                # We'll create a placeholder node or use fetched name
                if parent_node_id not in seen_nodes:
                    node_name = cui_to_name.get(parent_cui, f"Parent_{parent_cui}")
                    nodes.append({
                        'nodeId': parent_node_id,
                        ':LABEL': 'UMLSConcept',
                        'name': node_name,
                        'cui': parent_cui,
                        'type': 'ontology_parent'
                    })
                    seen_nodes.add(parent_node_id)
                
                # Edge: CUI -> Parent
                edges.append({
                    ':START_ID': cui_node_id,
                    ':END_ID': parent_node_id,
                    ':TYPE': 'IS_A',
                    'weight': 1.0
                })
                
        # 4. Semantic Typse
        if pd.notna(row['semantic_types']):
            stys = str(row['semantic_types']).split(';')
            for sty in stys:
                if not sty: continue
                sty_node_id = f"STY:{sty}"
                
                if sty_node_id not in seen_nodes:
                    nodes.append({
                        'nodeId': sty_node_id,
                        ':LABEL': 'SemanticType',
                        'name': sty,
                        'type': 'ontology_type'
                    })
                    seen_nodes.add(sty_node_id)
                
                # Edge: CUI -> STY
                edges.append({
                    ':START_ID': cui_node_id,
                    ':END_ID': sty_node_id,
                    ':TYPE': 'HAS_TYPE',
                    'weight': 1.0
                })

        # 5. Other Relations (from RO)
        if 'other_relations' in row and pd.notna(row['other_relations']):
            others = str(row['other_relations']).split(';')
            for item in others:
                if not item: continue
                try:
                    rel_type, target_cui = item.split(':')
                    target_node_id = f"CUI:{target_cui}"
                    
                    if target_node_id not in seen_nodes:
                        node_name = cui_to_name.get(target_cui, f"Related_{target_cui}")
                        nodes.append({
                            'nodeId': target_node_id,
                            ':LABEL': 'UMLSConcept',
                            'name': node_name,
                            'cui': target_cui,
                            'type': 'ontology_related'
                        })
                        seen_nodes.add(target_node_id)
                    
                    edges.append({
                        ':START_ID': cui_node_id,
                        ':END_ID': target_node_id,
                        ':TYPE': rel_type,
                        'weight': 1.0
                    })
                except ValueError:
                    continue  # Skip malformed entries

    # Save CSVs
    nodes_df = pd.DataFrame(nodes)
    edges_df = pd.DataFrame(edges)
    
    print(f"Generated {len(nodes_df)} Nodes and {len(edges_df)} Edges.")
    
    nodes_df.to_csv(f"{OUTPUT_DIR}/nodes.csv", index=False)
    edges_df.to_csv(f"{OUTPUT_DIR}/edges.csv", index=False)
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    build_neo4j_csv()
