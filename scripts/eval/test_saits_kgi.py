import torch
import sys
sys.path.append('/home/guido/Code/charite/baselines')

from models.saits.model import SAITSModule

def test_kgi_saits():
    print("Instantiating SAITSModule with KGI enabled...")
    model = SAITSModule(
        d_feature=17,
        d_model=64,
        seq_len=48,
        use_kgi=True,
        embedding_type="vanilla"
    )
    
    # 2. Simulate Batch
    B, T, D = 4, 48, 17
    
    batch = {
        "data": torch.randn(B, T, D),
        "input_mask": torch.ones(B, T, D),  # Fully observed for this simple test
        "artificial_mask": (torch.rand(B, T, D) < 0.2).float(), # 20% artificially missing
        "target": torch.randn(B, T, D)
    }
    
    # Randomly poke real holes into input_mask
    real_missing = torch.rand(B, T, D) < 0.3
    batch["input_mask"][real_missing] = 0.0
    batch["data"][real_missing] = 0.0
    
    print(f"Running Forward Pass with Input Data: {batch['data'].shape}")
    
    # 3. Test Forward
    try:
        model.eval()
        with torch.no_grad():
            outputs = model(batch)
            loss_dict = model._compute_loss(batch, outputs)
            
        print("\nSUCCESS! Forward Pass and Loss computation completed.")
        print(f"Imputed_3 Output shape: {outputs['imputed_3'].shape}")
        print(f"Total Loss   : {loss_dict['loss'].item():.4f}")
        print(f"MIT Loss     : {loss_dict['loss_mit'].item():.4f}")
        print(f"ORT Loss     : {loss_dict['loss_ort'].item():.4f}")
        
    except Exception as e:
        print("\nERROR during testing:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_kgi_saits()
