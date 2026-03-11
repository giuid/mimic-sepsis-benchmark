import pickle
import torch
import numpy as np
import os

# Create data/embeddings directory if not exists
os.makedirs("data/embeddings", exist_ok=True)

# Original relations file path
original_path = "data/embeddings/medbert_relation_embeddings_sepsis_full.pkl"
dummy_path = "data/embeddings/dummy_gaussian_embeddings.pkl"

if not os.path.exists(original_path):
    print(f"Original embeddings not found at {original_path}. Generating dummy skeleton.")
    # Fallback to some common itemids if original is missing
    itemids = [220045, 50813, 50912, 220210, 223762, 50820, 51265, 220277, 220052]
    # Create all pairs
    relations = [(id1, id2) for id1 in itemids for id2 in itemids]
else:
    with open(original_path, "rb") as f:
        rel_dict = pickle.load(f)
    relations = rel_dict.keys()

dummy_dict = {}
embedding_dim = 768 # MedBERT standard

print(f"Generating dummy Gaussian embeddings for {len(relations)} relations...")
for rel in relations:
    # Sample from N(0, I)
    dummy_dict[rel] = torch.randn(embedding_dim)

with open(dummy_path, "wb") as f:
    pickle.dump(dummy_dict, f)

print(f"Dummy semantics saved to {dummy_path}")
