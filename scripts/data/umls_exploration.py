import pandas as pd
import numpy as np
import os
from tqdm import tqdm

UMLS_DIR = "/home/guido/Data/indus_data/ontologies/2025AB/META"
MRREL_PATH = f"{UMLS_DIR}/MRREL.RRF"
MRSTY_PATH = f"{UMLS_DIR}/MRSTY.RRF"
MRCONSO_PATH = f"{UMLS_DIR}/MRCONSO.RRF"

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

def analyze_semantic_types():
    print("--- Analysis 1: Semantic Types (TUIs) of Target Concepts ---")
    results = []
    target_cuis = set(MIMIC_TARGETS.keys())
    with open(MRSTY_PATH, 'r') as f:
        for line in f:
            parts = line.split('|')
            if parts[0] in target_cuis:
                results.append({'CUI': parts[0], 'Label': MIMIC_TARGETS[parts[0]], 'TUI': parts[1]})
    df = pd.DataFrame(results)
    
    # Load TUI names (optional, but good)
    # TUI names are in SRDEF usually, but we can hardcode common ones
    tui_map = {
        "T201": "Clinical Attribute",
        "T059": "Laboratory Procedure",
        "T116": "Amino Acid, Peptide, or Protein",
        "T121": "Pharmacologic Substance",
        "T123": "Biologically Active Substance",
        "T196": "Element, Ion, or Isotope",
        "T081": "Quantitative Concept",
        "T109": "Organic Chemical",
        "T025": "Cell"
    }
    df['Semantic Type'] = df['TUI'].map(lambda x: tui_map.get(x, x))
    print(df.sort_values('Label'))
    return df

def analyze_connectivity():
    print("\n--- Analysis 2: Connectivity in MRREL ---")
    target_cuis = set(MIMIC_TARGETS.keys())
    counts = {cui: 0 for cui in target_cuis}
    rel_types = {cui: set() for cui in target_cuis}
    
    with open(MRREL_PATH, 'r') as f:
        for line in tqdm(f, desc="Scanning MRREL"):
            parts = line.split('|')
            c1, c2, rel = parts[0], parts[4], parts[3]
            if c1 in target_cuis:
                counts[c1] += 1
                rel_types[c1].add(rel)
            if c2 in target_cuis:
                counts[c2] += 1
                rel_types[c2].add(rel)
                
    summary = []
    for cui, label in MIMIC_TARGETS.items():
        summary.append({
            'Label': label,
            'Total Edges': counts[cui],
            'Relationship Types': ', '.join(list(rel_types[cui])[:5])
        })
    print(pd.DataFrame(summary).sort_values('Label'))

if __name__ == "__main__":
    analyze_semantic_types()
    analyze_connectivity()
