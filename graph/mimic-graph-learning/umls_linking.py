
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import os
import gc

# Config
UMLS_PATH = "/home/guido/Data/indus_data/ontologies/2025AB/META/MRCONSO.RRF"
MIMIC_BASE = "/home/guido/Data/indus_data/physionet.org/files/mimiciv/2.2/icu" # Assuming typical structure
# If path is different, we can adjust. Based on previous exploration, physionet.org exists.

# We will look for d_items.csv everywhere we can think of or just define 17 key items
# But let's try to be general if we find the file.
OUTPUT_DIR = "artifacts_step7"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load SapBERT
print("Loading SapBERT model and tokenizer...")
# Use the base PubMedBERT tokenizer to avoid vocabulary issues
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").cuda()
model.eval()

def get_embeddings(texts, batch_size=2048):
    all_embeds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=64).to('cuda')
        with torch.no_grad():
            outputs = model(**inputs)
            # CLS token embedding
            embeds = outputs.last_hidden_state[:, 0, :]
            all_embeds.append(embeds.cpu())
    return torch.cat(all_embeds, dim=0)

# 2. Load MIMIC Items
# Targeted list for our specific experiment
TARGET_ITEMS = [
    "Heart Rate", "Systolic Blood Pressure", "Diastolic Blood Pressure", "Respiratory Rate", "Oxygen Saturation",
    "Temperature Celsius", "Glucose", "Potassium", "Sodium", "Chloride", "Creatinine", "Blood Urea Nitrogen",
    "White Blood Cells", "Platelets", "Hemoglobin", "Hematocrit", "Lactate"
]
# Create a DataFrame for targets
mimic_df = pd.DataFrame({'label': TARGET_ITEMS})
mimic_df['source'] = 'experiment_targets'

# 2b. Add Top Medications (if available)
meds_path = f"{OUTPUT_DIR}/top_medications.csv"
if os.path.exists(meds_path):
    print(f"Loading medications from {meds_path}...")
    meds_df = pd.read_csv(meds_path)
    # drug column has names
    med_items = meds_df['drug'].tolist()
    
    med_rows = pd.DataFrame({'label': med_items})
    med_rows['source'] = 'medications'
    
    mimic_df = pd.concat([mimic_df, med_rows], ignore_index=True)
    print(f"Added {len(med_items)} medications. Total items: {len(mimic_df)}")

print(f"Targeting {len(mimic_df)} MIMIC items.")

# 3. Load UMLS (English Only)
print("Loading UMLS MRCONSO (this might take a while)...")
# CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|SRL|SUPPRESS|CVF
# We only need CUI (0) and STR (14), filtered by LAT (1) == 'ENG'
# RRF doesn't have headers, so we used index.
umls_data = []
chunk_size = 1000000
processed_rows = 0

# Count lines first? Nah, just read.
try:
    with open(UMLS_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading RRF"):
            parts = line.split('|')
            if len(parts) > 14 and parts[1] == 'ENG':
                # Filter by key clinical vocabularies to reduce noise and compute time
                # Column 11 is SAB
                sab = parts[11]
                if sab in ['SNOMEDCT_US', 'LOINC', 'RXNORM', 'MSH', 'ICD10CM', 'ICD9CM']:
                    umls_data.append((parts[0], parts[14]))
            
            # Memory safety - if we go over 10M, maybe stop/prune?
            # A100 40GB is huge but system RAM matters too.
            # MRCONSO is ~2GB file, so in memory it assumes ~10-20GB. Should be fine.
except FileNotFoundError:
    print(f"File not found: {UMLS_PATH}. Please verify mount.")
    exit(1)

print(f"Loaded {len(umls_data)} English UMLS terms.")
umls_df = pd.DataFrame(umls_data, columns=['CUI', 'STR'])
# Drop duplicates to save compute (same string, same embedding)
# But multiple CUIs can have same string. We want mapping string -> set(CUI).
# Actually SapBERT maps string -> vector.
# Let's map unique strings first.
unique_umls_strings = umls_df['STR'].unique().tolist()
print(f"Unique UMLS strings to embed: {len(unique_umls_strings)}")

# 4. Embed UMLS (Batch)
print("Embedding UMLS strings...")
umls_embeds = get_embeddings(unique_umls_strings, batch_size=4096) # A100 can handle large batches
# Shape: [N_unique, 768]

# 5. Embed MIMIC Targets
print("Embedding MIMIC targets...")
target_embeds = get_embeddings(mimic_df['label'].tolist(), batch_size=len(mimic_df))

# 6. Nearest Neighbor Search
print("Linking...")
# Use cosine similarity
# Normalize
umls_embeds = torch.nn.functional.normalize(umls_embeds, p=2, dim=1)
target_embeds = torch.nn.functional.normalize(target_embeds, p=2, dim=1)

results = []
for i, target_vec in enumerate(target_embeds):
    # Compute similarity: [1, 768] @ [768, N] -> [1, N]
    sim = torch.matmul(target_vec.unsqueeze(0), umls_embeds.T).squeeze(0)
    
    # Top 5
    topk_vals, topk_idxs = torch.topk(sim, k=5)
    
    target_label = mimic_df.iloc[i]['label']
    
    for rank, (score, idx) in enumerate(zip(topk_vals, topk_idxs)):
        match_str = unique_umls_strings[idx.item()]
        # Find all CUIs for this string
        cuis = umls_df[umls_df['STR'] == match_str]['CUI'].unique().tolist()
        
        for cui in cuis:
            results.append({
                'mimic_label': target_label,
                'rank': rank + 1,
                'score': score.item(),
                'match_str': match_str,
                'cui': cui
            })

results_df = pd.DataFrame(results)
output_path = f"{OUTPUT_DIR}/mimic_to_cui_sapbert.csv"
results_df.to_csv(output_path, index=False)
print(f"Saved mapping to {output_path}")

# Show top 1 for each
print("\nTop-1 Matches:")
top1 = results_df[results_df['rank'] == 1].groupby('mimic_label').first()
print(top1[['match_str', 'cui', 'score']])
