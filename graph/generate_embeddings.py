
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import pandas as pd

def generate_embeddings():
    # 1. Configuration
    model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    # Target directory for storing embeddings (Indus)
    home_dir = os.environ.get("HOME")
    output_dir = os.path.join(home_dir, "Data/indus_data/physionet.org/files/mimiciv/3.1/embeddings")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sapbert_embeddings_17.npy")
    
    print(f"Loading SapBERT model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name) # Run on CPU to avoid OOM
    
    # 2. Define Features (17)
    # Mapping raw feature names to medical concept names for better embedding
    feature_map = {
        "heart_rate": "Heart Rate",
        "sbp": "Systolic Blood Pressure",
        "dbp": "Diastolic Blood Pressure",
        "respiratory_rate": "Respiratory Rate",
        "spo2": "Oxygen Saturation",
        "temperature": "Body Temperature",
        "glucose": "Serum Glucose",
        "potassium": "Serum Potassium",
        "sodium": "Serum Sodium",
        "chloride": "Serum Chloride",
        "creatinine": "Serum Creatinine",
        "bun": "Blood Urea Nitrogen",
        "wbc": "White Blood Cell Count",
        "platelets": "Platelet Count",
        "hemoglobin": "Hemoglobin",
        "hematocrit": "Hematocrit",
        "lactate": "Serum Lactate"
    }
    
    # Ordered list of keys as used in the model
    feature_keys = [
        "heart_rate", "sbp", "dbp", "respiratory_rate", "spo2", "temperature",
        "glucose", "potassium", "sodium", "chloride", "creatinine", "bun",
        "wbc", "platelets", "hemoglobin", "hematocrit", "lactate"
    ]
    
    medical_terms = [feature_map[k] for k in feature_keys]
    print(f"Generating embeddings for: {medical_terms}")

    # 3. Generate Embeddings
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(medical_terms, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        # Use [CLS] token embedding (index 0)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
    print(f"Embeddings shape: {embeddings.shape}")
    
    # 4. Save
    np.save(output_path, embeddings)
    print(f"Saved embeddings to {output_path}")

if __name__ == "__main__":
    generate_embeddings()
