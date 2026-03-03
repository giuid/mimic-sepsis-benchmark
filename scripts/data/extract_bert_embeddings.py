import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import os

def extract_bert_embeddings():
    # 17 SOTA Feature Names (as used in mimic4.yaml)
    feature_names = [
        "heart_rate",
        "sbp",
        "dbp",
        "mbp",
        "respiratory_rate",
        "spo2",
        "temperature",
        "gcs",
        "glucose",
        "creatinine",
        "bilirubin",
        "bun",
        "wbc",
        "sodium",
        "potassium",
        "hematocrit",
        "bicarbonate"
    ]
    
    # Load BERT
    model_name = "bert-base-uncased"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    # Extract
    embeddings = []
    print("Extracting embeddings...")
    with torch.no_grad():
        for name in feature_names:
            # Clean name for BERT (replace _ with space)
            text = name.replace("_", " ")
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model(**inputs)
            # Use [CLS] token embedding (index 0)
            cls_emb = outputs.last_hidden_state[0, 0, :].numpy()
            embeddings.append(cls_emb)
            print(f"  {name} -> {text} -> shape {cls_emb.shape}")
            
    embeddings = np.array(embeddings)
    print(f"Final shape: {embeddings.shape}")
    
    # Save
    out_dir = os.path.expanduser("~/Data/indus_data/physionet.org/files/mimiciv/3.1/embeddings")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bert_embeddings_17.npy")
    np.save(out_path, embeddings)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    extract_bert_embeddings()
