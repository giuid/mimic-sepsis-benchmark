import torch
import inspect
from models.gpvae.model import GPVAEModule

def debug():
    d_feature = 17
    seq_len = 48
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = GPVAEModule(d_feature=d_feature, seq_len=seq_len).to(device)
    
    # Inspect forward signature
    sig = inspect.signature(model.model.forward)
    print(f"Forward Signature: {sig}")
    
    # Run with explicit calc_criterion
    batch = {
        "data": torch.randn(4, seq_len, d_feature).to(device),
        "input_mask": torch.ones(4, seq_len, d_feature).to(device),
        "target": torch.randn(4, seq_len, d_feature).to(device)
    }
    inputs = {"X": batch["data"], "missing_mask": (1 - batch["input_mask"]).bool()}
    
    # Ensure model in train()
    model.model.train()
    
    print("Running model(inputs, calc_criterion=True)...")
    output = model.model(inputs, calc_criterion=True)
    print(f"Output keys: {list(output.keys())}")
    
    if "loss" in output:
        print("Loss found!")
        print(f"Loss: {output['loss']}")
        print(f"Loss requires_grad: {output['loss'].requires_grad}")
    else:
        print("Loss STILL NOT FOUND.")

if __name__ == "__main__":
    debug()
