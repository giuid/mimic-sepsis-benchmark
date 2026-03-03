import torch
from models.gpvae.model import GPVAEModule

def debug():
    d_feature = 17
    seq_len = 48
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = GPVAEModule(d_feature=d_feature, seq_len=seq_len).to(device)
    model.train()
    
    # Mock batch
    batch = {
        "data": torch.randn(4, seq_len, d_feature).to(device),
        "input_mask": torch.ones(4, seq_len, d_feature).to(device),
        "target": torch.randn(4, seq_len, d_feature).to(device)
    }
    
    print("Running training_step...")
    # training_step calls self.model(inputs)
    # self.model is self.pypots_model.model
    loss = model.training_step(batch, 0)
    print(f"Loss: {loss}")
    print(f"Loss type: {type(loss)}")
    
    if isinstance(loss, torch.Tensor):
        print(f"Loss requires_grad: {loss.requires_grad}")
        print(f"Loss grad_fn: {loss.grad_fn}")
        
        if loss.requires_grad:
            print("Running backward...")
            loss.backward()
            print("Backward successful.")
            has_grad = False
            for name, param in model.model.named_parameters():
                if param.grad is not None:
                    print(f"Param {name} has grad sum: {param.grad.sum()}")
                    has_grad = True
                    break
            if not has_grad:
                print("FAILED: No parameters have gradients.")
        else:
            print("FAILED: Loss does not require grad.")
    else:
        print(f"FAILED: Loss is not a tensor, it is {type(loss)}")

if __name__ == "__main__":
    debug()
