import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import argparse
import pickle
import numpy as np
from tqdm import tqdm

def load_mimic_vocabulary(data_path: str):
    """
    Loads MIMIC-IV 3.1 dicts and returns a concatenated DataFrame
    of itemid, label, and origin.
    """
    dic_items = os.path.join(data_path, 'icu/d_items.csv.gz')
    dic_labitems = os.path.join(data_path, 'hosp/d_labitems.csv.gz')
    
    print(f"Loading dictionaries from {data_path}...")
    
    # Check if files exist
    if not os.path.exists(dic_items):
        raise FileNotFoundError(f"Missing {dic_items}")
    if not os.path.exists(dic_labitems):
        raise FileNotFoundError(f"Missing {dic_labitems}")
        
    df_items = pd.read_csv(dic_items, usecols=['itemid', 'label', 'category', 'param_type'])
    df_items['origin'] = 'chartevents'
    
    df_labitems = pd.read_csv(dic_labitems, usecols=['itemid', 'label', 'fluid', 'category'])
    df_labitems['origin'] = 'labevents'
    # Rename fluid/category to match items somewhat for easier debugging later
    df_labitems.rename(columns={'fluid': 'param_type'}, inplace=True)
    
    df_all = pd.concat([df_items, df_labitems], ignore_index=True)
    
    # We drop any null labels
    df_all.dropna(subset=['label'], inplace=True)
    
    print(f"Loaded {len(df_all)} total unique concepts from MIMIC-IV.")
    return df_all


def generate_sapbert_embeddings(labels: list, batch_size: int = 256):
    """
    Generates [CLS] embeddings for a list of clinical strings using SapBERT.
    Returns a numpy array of shape [len(labels), 768].
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    model = AutoModel.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext').to(device)
    model.eval()

    all_embeddings = []
    
    print(f"Encoding {len(labels)} labels with SapBERT...")
    for i in tqdm(range(0, len(labels), batch_size)):
        batch_labels = labels[i:i+batch_size]
        
        inputs = tokenizer(batch_labels, padding=True, truncation=True, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # The [CLS] token representation (index 0)
            batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
        all_embeddings.append(batch_emb)
        
    return np.vstack(all_embeddings)


if __name__ == '__main__':
    # Configuration
    DATA_PATH = '/home/guido/Data/indus_data/physionet.org/files/mimiciv/3.1/'
    OUT_DIR = '/home/guido/Code/charite/baselines/data/embeddings/'
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. Load data
    df_vocab = load_mimic_vocabulary(DATA_PATH)
    
    # Filter for the most common specific variables (Optional prototype step)
    # To save time during the prototype, let's just encode EVERYTHING. 
    # SapBERT on a 3090 is fast enough for ~18k strings.
    labels_to_encode = df_vocab['label'].tolist()
    
    # 2. Get embeddings
    embeddings = generate_sapbert_embeddings(labels_to_encode, batch_size=256)
    
    # 3. Create mapping dictionary
    itemid_to_idx = {itemid: idx for idx, itemid in enumerate(df_vocab['itemid'])}
    
    # 4. Save
    print("Saving artifacts...")
    
    # Save the dataframe mapping
    df_vocab.to_csv(os.path.join(OUT_DIR, 'mimic_vocab_mapped.csv'), index=False)
    
    # Save the huge numpy tensor and index mapping
    output_data = {
        'embeddings': embeddings, # [18562, 768] shape
        'itemid_to_idx': itemid_to_idx,
        'idx_to_label': {idx: label for idx, label in enumerate(df_vocab['label'])}
    }
    
    with open(os.path.join(OUT_DIR, 'sapbert_mimic_embeddings.pkl'), 'wb') as f:
        pickle.dump(output_data, f)
        
    print(f"Done! Embeddings saved to {OUT_DIR}sapbert_mimic_embeddings.pkl")
    print(f"Matrix shape: {embeddings.shape}")
