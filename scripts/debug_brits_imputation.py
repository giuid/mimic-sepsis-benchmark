import torch
import numpy as np
import os
from models.joint.model import JointTrainingModule

if hasattr(torch.serialization, 'add_safe_globals'):
    import numpy
    import numpy._core.multiarray
    import numpy.dtypes
    torch.serialization.add_safe_globals([
        numpy._core.multiarray.scalar, 
        numpy.dtype, 
        numpy.dtypes.Float32DType,
        numpy.float32
    ])

def check_brits():
    ckpt_path = "outputs/checkpoints/joint_brits/vanilla_Vanilla/last-v1.ckpt"
    data_dir = "data/sota"
    
    # Load model
    model = JointTrainingModule.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    
    # Load test data
    test_data = np.load(os.path.join(data_dir, "test.npz"))
    X = torch.from_numpy(test_data["data"]).float()
    M = torch.from_numpy(test_data["orig_mask"]).float()
    D = torch.from_numpy(test_data["delta"]).float()
    
    # Run imputer
    batch = {"data": X[:500], "input_mask": M[:500], "delta": D[:500]} # Test on subset
    with torch.no_grad():
        inputs = model.imputer._assemble_inputs(batch["data"], batch["input_mask"], batch["delta"])
        out = model.imputer.model(inputs, calc_criterion=True)
        if isinstance(out, dict):
            if "imputed_data" in out:
                imputed = out["imputed_data"]
            elif "reconstruction" in out:
                imputed = out["reconstruction"]
            else:
                print(f"Warning: BRITS keys are {out.keys()}")
                imputed = out[list(out.keys())[0]]
        else:
            imputed = out[0]
            
    # Calculate MAE on missing values
    mask = 1 - M[:500] # 1 for missing in PyPOTS
    mae = torch.abs(imputed - X[:500]) * mask
    avg_mae = mae.sum() / mask.sum().clamp(min=1)
    
    print(f"BRITS Test MAE: {avg_mae.item():.4f}")
    
    # Compare with SAITS if possible
    saits_ckpt = "outputs/checkpoints/joint_saits/vanilla_Vanilla_a0.01/best-epoch=20-val/auprc=0.2952.ckpt"
    if os.path.exists(saits_ckpt):
        model_saits = JointTrainingModule.load_from_checkpoint(saits_ckpt, map_location="cpu")
        model_saits.eval()
        with torch.no_grad():
            outputs = model_saits.imputer(batch)
            imputed_saits = outputs["imputed_3"]
        mae_saits = torch.abs(imputed_saits - X[:500]) * mask
        avg_mae_saits = mae_saits.sum() / mask.sum().clamp(min=1)
        print(f"SAITS Test MAE: {avg_mae_saits.item():.4f}")

if __name__ == "__main__":
    check_brits()
