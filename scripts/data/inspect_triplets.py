import os
import pandas as pd
import pickle
import random

def inspect_triplets(num_samples=10):
    OUT_DIR = '/home/guido/Code/charite/baselines/data/embeddings/'
    VOCAB_CSV = os.path.join(OUT_DIR, 'mimic_vocab_mapped.csv')
    
    # Load Vocabulary Map
    if not os.path.exists(VOCAB_CSV):
        print("Vocabulary CSV not found.")
        return
        
    df_vocab = pd.read_csv(VOCAB_CSV)
    
    # Create lookup dict for itemid -> label
    id_to_label = dict(zip(df_vocab['itemid'], df_vocab['label']))
    
    # Load MedBERT Relational Embeddings
    medbert_file = os.path.join(OUT_DIR, 'medbert_relation_embeddings.pkl')
    if not os.path.exists(medbert_file):
        print("MedBERT embeddings not found.")
        return
        
    with open(medbert_file, 'rb') as f:
        relation_dict = pickle.load(f)
        
    print(f"Total relation pairs generated: {len(relation_dict)}")
    print("\n--- SAMPLE EXTRACTED MEDBERT TRIPLETS ---\n")
    
    # Pick random samples to show the user
    keys = list(relation_dict.keys())
    sample_keys = random.sample(keys, min(num_samples, len(keys)))
    
    for i, (item_a, item_b) in enumerate(sample_keys):
        label_a = id_to_label.get(item_a, f"Unknown ID {item_a}")
        label_b = id_to_label.get(item_b, f"Unknown ID {item_b}")
        
        # This matches the 'create_verbalization' template used in the previous step
        # "Clinical knowledge indicates that {label_a} is associated with {label_b}."
        sentence = f"Clinical knowledge indicates that {label_a} is associated with {label_b}."
        
        embedding_shape = relation_dict[(item_a, item_b)].shape
        print(f"Sample {i+1}:")
        print(f"Pair IDs    : ({item_a}, {item_b})")
        print(f"Sentence    : '{sentence}'")
        print(f"Ternary     : ({label_a} -> associated_with -> {label_b})")
        print(f"Vector Shape: {embedding_shape}")
        print("-" * 50)

if __name__ == '__main__':
    inspect_triplets(15)
