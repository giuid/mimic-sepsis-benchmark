import torch

ckpt_path_sbc = "outputs/saits/random/2026-02-18_14-16-23/checkpoints/best-epoch=96-val/loss=0.3244.ckpt"
ckpt_path_pnl = "outputs/saits/random/2026-02-18_14-16-22/checkpoints/best-epoch=93-val/loss=0.3246.ckpt"

def check_gate(path, name):
    print(f"\n--- {name} ---")
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    
    # List first 10 keys to see structure
    print("Example keys:")
    for k in list(state_dict.keys())[:20]:
        print(f"  {k}")
    
    # Search for gate_param
    gate_keys = [k for k in state_dict.keys() if "gate_param" in k]
    print(f"Found gate_keys: {gate_keys}")
    
    for key in gate_keys:
        raw_gate = state_dict[key]
        alpha = torch.sigmoid(raw_gate)
        print(f"{key}: Raw {raw_gate.numpy()}, Alpha {alpha.numpy()}")

check_gate(ckpt_path_sbc, "SapBERT + CI-GNN Parallel")
check_gate(ckpt_path_pnl, "Prior Nullo Parallel")
