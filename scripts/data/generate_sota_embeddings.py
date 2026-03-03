
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_embeddings():
    # 1. Configuration
    sapbert_model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    bert_model_name = "bert-base-uncased" # For generic BERT embeddings
    
    # Target directory for storing embeddings (Local SOTA directory)
    output_dir = "data/processed_sota"
    os.makedirs(output_dir, exist_ok=True)
    
    sapbert_output_path = os.path.join(output_dir, "sapbert_embeddings_17.npy")
    bert_output_path = os.path.join(output_dir, "bert_embeddings_17.npy")
    
    # 2. Define Features (17) - SOTA Configuration
    # Mapping raw feature names from mimic4.yaml to medical concept names
    feature_map = {
        "heart_rate": "Heart Rate",
        "sbp": "Systolic Blood Pressure",
        "dbp": "Diastolic Blood Pressure",
        "mbp": "Mean Blood Pressure",         # NEW
        "respiratory_rate": "Respiratory Rate",
        "spo2": "Oxygen Saturation",
        "temperature": "Body Temperature",
        "gcs": "Glasgow Coma Scale",          # NEW
        "glucose": "Serum Glucose",
        "creatinine": "Serum Creatinine",
        "bilirubin": "Total Bilirubin",       # NEW
        "bun": "Blood Urea Nitrogen",
        "wbc": "White Blood Cell Count",
        "sodium": "Serum Sodium",
        "potassium": "Serum Potassium",
        "hematocrit": "Hematocrit",
        "bicarbonate": "Serum Bicarbonate"    # NEW
    }
    
    # Ordered list of keys as used in the model/yaml
    feature_keys = [
        "heart_rate", "sbp", "dbp", "mbp", "respiratory_rate", "spo2", "temperature",
        "gcs", "glucose", "creatinine", "bilirubin", "bun", "wbc", "sodium",
        "potassium", "hematocrit", "bicarbonate"
    ]
    
    medical_terms = [feature_map[k] for k in feature_keys]
    logging.info(f"Generating embeddings for {len(medical_terms)} terms:")
    for k, t in zip(feature_keys, medical_terms):
        logging.info(f"  {k} -> {t}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Generate SapBERT Embeddings ---
    logging.info(f"Loading SapBERT model: {sapbert_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(sapbert_model_name)
    model = AutoModel.from_pretrained(sapbert_model_name).to(device)
    
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(medical_terms, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # Use [CLS] token embedding (index 0)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
    logging.info(f"SapBERT Embeddings shape: {embeddings.shape}")
    np.save(sapbert_output_path, embeddings)
    logging.info(f"Saved SapBERT embeddings to {sapbert_output_path}")
    
    # --- Generate Generic BERT Embeddings ---
    logging.info(f"Loading Generic BERT model: {bert_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    model = AutoModel.from_pretrained(bert_model_name).to(device)
    
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(medical_terms, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # Use [CLS] token embedding (index 0)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
    logging.info(f"BERT Embeddings shape: {embeddings.shape}")
    np.save(bert_output_path, embeddings)
    logging.info(f"Saved BERT embeddings to {bert_output_path}")

if __name__ == "__main__":
    generate_embeddings()
