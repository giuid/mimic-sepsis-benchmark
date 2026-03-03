import os
import zipfile
import pandas as pd
from tqdm import tqdm

def extract_true_umls_relations(zip_path: str, target_file: str, max_lines: int = 5000000):
    """
    Extracts relations from UMLS MRREL.RRF directly from the ZIP file.
    MRREL format: CUI1 | AUI1 | STYPE1 | REL | CUI2 | AUI2 | STYPE2 | RELA | ...
    REL: Relationship (e.g., RQ, PAR, CHD, SY, RO)
    RELA: Relationship Attribute (e.g., 'causative_agent_of', 'diagnoses', 'associated_with')
    """
    print(f"Opening UMLS archive: {zip_path}")
    
    # We will store a subset of interesting relations
    # To keep it manageable in memory, we just look for specific REL/RELA
    # focusing on clinical causality
    interesting_rels = ['RN', 'RO'] # Narrower, Other related
    interesting_relas = [
        'causative_agent_of', 'causes', 'diagnoses', 
        'disease_has_finding', 'has_finding', 'indicates',
        'associated_with', 'measured_by', 'measures'
    ]
    
    extracted_triplets = []
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        if target_file not in z.namelist():
            raise FileNotFoundError(f"{target_file} not found in zip.")
            
        print(f"Streaming {target_file}...")
        with z.open(target_file) as f:
            for i, line in tqdm(enumerate(f), total=max_lines, desc="Parsing MRREL"):
                if i >= max_lines:
                    break
                    
                # RRF is pipe-delimited
                parts = line.decode('utf-8').strip().split('|')
                if len(parts) < 8:
                    continue
                    
                cui1 = parts[0]
                rel = parts[3]
                cui2 = parts[4]
                rela = parts[7]
                
                # Filter for clinical causal/diagnostic relations (ignore hierarchies like PAR/CHD)
                if rel in interesting_rels or rela in interesting_relas:
                    if rela: # Prefer the specific attribute if available
                        extracted_triplets.append((cui1, rela, cui2))
                    else:
                        extracted_triplets.append((cui1, rel, cui2))
                        
    print(f"\nExtracted {len(extracted_triplets)} clinical relations.")
    
    # Let's showcase a few unique ones
    df_triplets = pd.DataFrame(extracted_triplets, columns=['CUI1', 'Relation', 'CUI2'])
    df_unique = df_triplets.drop_duplicates()
    
    print(f"Unique canonical relations: {len(df_unique)}")
    
    print("\n--- SAMPLE REAL UMLS TRIPLETS (CUI form) ---")
    
    # Show relations specifically using clinical verbs
    causal = df_unique[df_unique['Relation'].isin(['causes', 'causative_agent_of', 'diagnoses', 'disease_has_finding'])].head(10)
    for _, row in causal.iterrows():
        print(f"[{row['CUI1']}] -- {row['Relation']} --> [{row['CUI2']}]")
        
    return df_unique

if __name__ == '__main__':
    UMLS_ZIP = "/home/guido/Data/indus_data/umls-2025AB-metathesaurus-full.zip"
    MRREL_PATH = "2025AB/META/MRREL.RRF"
    
    extract_true_umls_relations(UMLS_ZIP, MRREL_PATH, max_lines=5000000)
    
    # NOTE: To turn CUI1 and CUI2 into readable English Text (e.g. Heart Rate), 
    # we would also need to stream MRCONSO.RRF to map CUI -> STR.
    # For this demonstration snippet, we show the Graph Edges with the raw CUIs and Verbs.
