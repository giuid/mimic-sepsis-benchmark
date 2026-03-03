import torch
import pytorch_lightning as pl
from models.saits.model import SAITSModule
from models.brits.model import BRITSModule
from models.mrnn.model import MRNNModule
from data.dataset import MIMICDataModule
import os
import argparse
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig

# PyTorch 2.6 security: allow OmegaConf types for weight loading
torch.serialization.add_safe_globals([ListConfig, DictConfig, dict])

def evaluate(model_name, ckpt_path):
    model_classes = {
        "saits": SAITSModule,
        "brits": BRITSModule,
        "mrnn": MRNNModule,
    }

    processed_dir = "data/sota"
    
    # Initialize DataModule
    datamodule = MIMICDataModule(
        processed_dir=processed_dir,
        batch_size=128,
        num_workers=4
    )
    datamodule.setup("test")
    
    # We use GPU as assigned by CUDA_VISIBLE_DEVICES from run_kgi_evals.py
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        logger=False
    )

    print(f"\nEvaluating {model_name}...")
    print(f"  Checkpoint: {os.path.basename(ckpt_path)}")
    
    # Extract model type, handling potential prefixes like dkgi_
    m_type = next((m for m in model_classes.keys() if m in model_name.lower()), None)
    if not m_type:
        print(f"Error: Could not determine model type from name {model_name}")
        return
    model_class = model_classes[m_type]
    
    # Load model
    try:
        model = model_class.load_from_checkpoint(ckpt_path, weights_only=False)
    except Exception as e:
        print(f"Direct load failed, trying to load via JointTrainingModule wrapper...")
        try:
            from models.joint.model import JointTrainingModule
            # Load the joint module first
            joint_module = JointTrainingModule.load_from_checkpoint(ckpt_path, weights_only=False)
            model = joint_module.imputer
        except Exception as e2:
            print(f"Error loading checkpoint {ckpt_path}: {e2}")
            return

    # Run test
    test_results = trainer.test(model, datamodule=datamodule, verbose=False)
    
    if test_results:
        m = test_results[0]
        mae = m.get('test/mae', 0.0)
        rmse = m.get('test/rmse', 0.0)
        mre = m.get('test/mre', 0.0)
        r2 = m.get('test/r2', 0.0)
        corr_err = m.get('test/corr_err', 0.0)
        print(f"  Result: MAE={mae:.4f}, RMSE={rmse:.4f}, MRE={mre:.4f}, R2={r2:.4f}, CorrErr={corr_err:.4f}")
    else:
        print("  Evaluation returned no results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model (e.g., SAITS_SEM_GNN)")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint file")
    
    # We add these arguments just so the run_kgi_evals.py doesn't crash when passing them, 
    # but we will ignore them and load everything from the checkpoint.
    parser.add_argument("--dataset", type=str, default="mimic4")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--embedding_type", type=str, default="")
    parser.add_argument("--use_kgi", action="store_true")
    parser.add_argument("--use_graph_layer", action="store_true")
    
    args = parser.parse_args()
    evaluate(args.model_name, args.ckpt_path)
