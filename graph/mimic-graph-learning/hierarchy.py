
import pandas as pd
import os
from tqdm import tqdm

# Config
UMLS_DIR = "/home/guido/Data/indus_data/ontologies/2025AB/META"
INPUT_CSV = "artifacts_step7/mimic_to_cui_sapbert.csv"
OUTPUT_DIR = "artifacts_step7"

def extract_hierarchy():
    if not os.path.exists(INPUT_CSV):
        print(f"Input file {INPUT_CSV} not found.")
        return

    print("Loading mapped CUIs...")
    df = pd.read_csv(INPUT_CSV)
    # Filter top-1 matches
    top1 = df[df['rank'] == 1].copy()
    target_cuis = set(top1['cui'].unique())
    print(f"Targeting {len(target_cuis)} unique CUIs for hierarchy extraction.")
    
    # 1. semantic Types (MRSTY)
    # CUI|TUI|STN|STY|ATUI|CVF
    print("Scanning MRSTY.RRF for Semantic Types...")
    cui_to_sty = {}
    
    with open(f"{UMLS_DIR}/MRSTY.RRF", 'r') as f:
        for line in f:
            parts = line.split('|')
            cui = parts[0]
            if cui in target_cuis:
                sty = parts[3] # Semantic Type
                if cui not in cui_to_sty:
                    cui_to_sty[cui] = []
                cui_to_sty[cui].append(sty)
                
    # 2. Hierarchy (MRREL)
    # CUI1|AUI1|STYPE1|REL|CUI2|AUI2|STYPE2|RELA|RUI|SRUI|SAB|SL|RG|DIR|SUPPRESS|CVF
    # We look for CUI1 in target_cuis AND REL in ['PAR'] (CUI2 is parent of CUI1)
    # Or CUI2 is target AND REL is CHD?
    # Usually PAR means CUI2 is parent of CUI1.
    print("Scanning MRREL.RRF for Parents (PAR) and Relations (measures, analyzes)...")
    cui_relations = {} # Key: CUI, Value: list of (rel_type, target_cui)
    
    # Relations of interest
    target_relas = {
        'measures', 'analyzes', 'has_finding_site', 'has_component', 
        'has_ingredient', 'gene_product_of', 'associated_with',
        'treats', 'causes', 'location_of', 'diagnoses', 'method_of'
    }
    
    with open(f"{UMLS_DIR}/MRREL.RRF", 'r') as f:
        for line in tqdm(f, desc="MRREL"):
            # Optimization: fast check if line contains any target CUI
            parts = line.split('|')
            cui1 = parts[0]
            rel = parts[3]
            cui2 = parts[4]
            rela = parts[7]
            
            if cui1 in target_cuis:
                # 1. Parents
                if rel == 'PAR':
                    if cui1 not in cui_relations:
                        cui_relations[cui1] = []
                    cui_relations[cui1].append(('IS_A', cui2))
                
                # 2. Other Relations (RO)
                elif rel == 'RO' and rela in target_relas:
                    if cui1 not in cui_relations:
                        cui_relations[cui1] = []
                    cui_relations[cui1].append((rela.upper(), cui2))
                
    # 3. Fetch English Names for ALL referenced CUIs (MRCONSO)
    # Collect all CUIs we need names for (Parents + Related)
    all_referenced_cuis = set()
    for _, row in top1.iterrows():
        cui = row['cui']
        rels = cui_relations.get(cui, [])
        
        # Add parents
        all_referenced_cuis.update({target for r_type, target in rels if r_type == 'IS_A'})
        # Add others
        all_referenced_cuis.update({target for r_type, target in rels if r_type != 'IS_A'})
        
    print(f"Need names for {len(all_referenced_cuis)} related CUIs (Parents/Others).")
    
    cui_to_name = {}
    print("Scanning MRCONSO.RRF for English Names...")
    
    with open(f"{UMLS_DIR}/MRCONSO.RRF", 'r') as f:
        for line in tqdm(f, desc="MRCONSO"):
            parts = line.split('|')
            cui = parts[0]
            lat = parts[1] # Language
            ts = parts[2]  # Term Status (P=Preferred)
            stt = parts[4] # String Type (PF=Preferred form)
            name_str = parts[14]
            
            if cui in all_referenced_cuis and lat == 'ENG':
                # Heuristic: Prefer 'P' (Preferred) and 'PF' (Preferred Form)
                # If we don't have a name yet, take it.
                # If we have a name, overwrite ONLY if this is a Preferred Term (TS='P') and previous wasn't
                # Or if both are P, maybe check length? simpler is better.
                
                if cui not in cui_to_name:
                    cui_to_name[cui] = (name_str, ts)
                else:
                    current_name, current_ts = cui_to_name[cui]
                    if current_ts != 'P' and ts == 'P':
                        cui_to_name[cui] = (name_str, ts)
                    elif current_ts == 'P' and ts == 'P' and stt == 'PF':
                         cui_to_name[cui] = (name_str, ts)

    # Save Names to CSV
    name_data = [{'cui': c, 'name': n[0]} for c, n in cui_to_name.items()]
    names_df = pd.DataFrame(name_data)
    names_df.to_csv(f"{OUTPUT_DIR}/cui_names.csv", index=False)
    print(f"Saved {len(names_df)} names to {OUTPUT_DIR}/cui_names.csv")

    # Compile Results (same as before)
    results = []
    for _, row in top1.iterrows():
        cui = row['cui']
        stys = cui_to_sty.get(cui, [])
        # Parse relations
        rels = cui_relations.get(cui, [])
        parents = {target for r_type, target in rels if r_type == 'IS_A'}
        others = [f"{r_type}:{target}" for r_type, target in rels if r_type != 'IS_A']
        
        results.append({
            'mimic_label': row['mimic_label'],
            'cui': cui,
            'match_str': row['match_str'],
            'semantic_types': ";".join(stys),
            'parents_cui': ";".join(parents),
            'other_relations': ";".join(others)
        })
        
    final_df = pd.DataFrame(results)
    final_df.to_csv(f"{OUTPUT_DIR}/mimic_hierarchy.csv", index=False)
    print(f"Saved hierarchy to {OUTPUT_DIR}/mimic_hierarchy.csv")
    print(final_df[['mimic_label', 'semantic_types', 'parents_cui']].head(20))

if __name__ == "__main__":
    extract_hierarchy()
