import torch
import numpy as np
from models.timesfm.model import TimesFMModule

def test_timesfm_kgi():
    print("Testing TimesFM-KGI Integration...")
    
    d_feature = 17
    seq_len = 48
    
    # Initialize model with KGI
    model = TimesFMModule(
        d_feature=d_feature,
        seq_len=seq_len,
        use_kgi=True,
        kgi_embedding_file="medbert_relation_embeddings_generic.pkl"
    )
    
    # Create dummy batch
    batch_size = 2
    data = torch.randn(batch_size, seq_len, d_feature)
    input_mask = torch.ones(batch_size, seq_len, d_feature)
    # Mask out some values to test the patch mask logic
    input_mask[:, 10:20, 5] = 0
    
    batch = {
        "data": data,
        "input_mask": input_mask,
        "target": data,
        "artificial_mask": torch.zeros_like(input_mask)
    }
    
    # Run forward pass
    model.eval()
    with torch.no_grad():
        try:
            output = model(batch)
            print("Forward pass successful!")
            print(f"Output shape: {output['imputed_3'].shape}")
            
            # Verify KGI attention weights were stored (if implemented in KGIFusionLayer)
            if hasattr(model.kgi_fusion, 'last_attn_weights'):
                print("KGI Attention weights found!")
                print(f"Attn weights shape: {model.kgi_fusion.last_attn_weights.shape}")
                
        except Exception as e:
            print(f"Forward pass failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_timesfm_kgi()
