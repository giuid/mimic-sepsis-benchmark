
import hydra
from omegaconf import OmegaConf
from data.dataset import MIMICDataModule
import os

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def debug(cfg):
    print("Debug: Initializing DataModule...")
    masking_cfg = OmegaConf.to_container(cfg.masking, resolve=True)
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    
    print(f"Data Root: {data_cfg.get('mimic_root', 'Not Set')}")
    print(f"Processed Dir: {data_cfg['processed_dir']}")
    
    dm = MIMICDataModule(
        processed_dir=data_cfg["processed_dir"],
        masking_cfg=masking_cfg,
        batch_size=64,
        num_workers=4
    )
    dm.setup("fit")
    
    train_len = len(dm.train_dataset)
    val_len = len(dm.val_dataset)
    
    print(f"Train Samples: {train_len}")
    print(f"Val Samples: {val_len}")
    print(f"Batch Size: 64")
    print(f"Expected Batches per Epoch: {train_len // 64}")

if __name__ == "__main__":
    debug()
