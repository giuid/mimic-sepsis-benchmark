import torch
import pickle
import numpy as np
from model_mock import KnowledgeGuidedImputation

def test_kgi_forward_pass():
    print("Loading extracted embeddings...")
    embeddings_dir = '/home/guido/Code/charite/baselines/data/embeddings/'
    
    with open(embeddings_dir + 'sapbert_mimic_embeddings.pkl', 'rb') as f:
        sapbert_data = pickle.load(f)
    sapbert_matrix = torch.tensor(sapbert_data['embeddings'], dtype=torch.float32)
    
    with open(embeddings_dir + 'medbert_relation_embeddings.pkl', 'rb') as f:
        medbert_dict = pickle.load(f)
        
    print(f"Loaded SapBERT matrix: {sapbert_matrix.shape}")
    print(f"Loaded {len(medbert_dict)} MedBERT relation pairs.")

    # ---------------------------------------------------------
    # Simulate a tiny batch of data based on the features we encoded relations for
    # We extracted relations for the first 50 features.
    NUM_EVAL_FEATURES = 50 
    BATCH_SIZE = 4
    SEQ_LEN = 12 # E.g., 12 hours of data
    
    # We only pass the SapBERT embeddings for the top 50 features
    sapbert_subset = sapbert_matrix[:NUM_EVAL_FEATURES, :]
    
    print("\nInitializing KGI Fusion Model...")
    model = KnowledgeGuidedImputation(
        num_features=NUM_EVAL_FEATURES,
        sapbert_dim=768,
        medbert_dim=768,
        hidden_dim=128,
        mask_ratio=0.3
    )
    
    # Enable training mode to trigger Stochastic Masking
    model.train()

    # Simulate clinical time-series [Batch, Time, Features]
    # We initialize it with random values, and randomly inject NaNs to simulate true missingness
    simulated_data = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_EVAL_FEATURES)
    # 50% naturally missing data
    true_missing_mask = torch.rand_like(simulated_data) < 0.5 
    simulated_data[true_missing_mask] = float('nan')
    
    print(f"\nPassing simulated data {simulated_data.shape} through model...")
    print(f"Applying ~30% Stochastic Masking to the remaining valid observations...")
    
    # Forward Pass
    # variable_indices just maps 0..49 to the true itemid for relation lookup
    variable_indices = [sapbert_data['itemid_to_idx'][itemid] for itemid in list(sapbert_data['itemid_to_idx'].keys())[:NUM_EVAL_FEATURES]]
    # Since our indices in medbert dict are the true itemids, we map them correctly
    top_itemids = list(sapbert_data['itemid_to_idx'].keys())[:NUM_EVAL_FEATURES]

    # Quick patch to text injector for mocked test: We need to pass the true itemid to the dict
    def patch_forward(self, surviving_mask, precomputed_embeddings, itemids):
        batch_size, seq_len, num_features = surviving_mask.shape
        device = surviving_mask.device
        context_batch = torch.zeros(batch_size, seq_len, self.text_embed_dim, device=device)
        
        for b in range(batch_size):
            for t in range(seq_len):
                present_idxs = surviving_mask[b, t, :].nonzero(as_tuple=True)[0].tolist()
                step_context = []
                for i in range(len(present_idxs)):
                    for j in range(i + 1, len(present_idxs)):
                        # Map internal index (0..49) to global itemid (e.g., 220045)
                        id_a = itemids[present_idxs[i]]
                        id_b = itemids[present_idxs[j]]
                        key = tuple(sorted([id_a, id_b]))
                        if key in precomputed_embeddings:
                            step_context.append(precomputed_embeddings[key].to(device))
                if step_context:
                    aggregated = torch.stack(step_context).mean(dim=0)
                    context_batch[b, t, :] = aggregated
        return self.adapter(context_batch)
    
    # Apply monkey patch for correct ID translation
    import types
    model.medbert_injector.forward = types.MethodType(patch_forward, model.medbert_injector)

    
    imputed_values, stochastic_mask, attn_weights = model(
        batch_data=simulated_data,
        sapbert_embeddings=sapbert_subset,
        precomputed_medbert_dict=medbert_dict,
        variable_indices=top_itemids
    )
    
    print("\n--- FORWARD PASS SUCCESSFUL ---")
    print(f"Imputed output shape  : {imputed_values.shape}")
    print(f"Stochastic mask shape : {stochastic_mask.shape}")
    print(f"Attention map shape   : {attn_weights.shape}")
    print(f"Number of artificially masked points (for Loss calculation): {stochastic_mask.sum().item()}")

if __name__ == '__main__':
    test_kgi_forward_pass()
