import os
import torch
import numpy as np
import logging
import timesfm
from tqdm import tqdm
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

class TimesFMImputer:
    """
    Wrapper for Google's TimesFM 2.5 (200M) for multivariate time-series imputation.
    """
    def __init__(self, model_id="google/timesfm-2.5-200m-pytorch", device="cuda"):
        self.device = device
        self.model_id = model_id
        
        weights_dir = os.path.expanduser("~/timesfm_25_weights")
        os.makedirs(weights_dir, exist_ok=True)
        
        logger.info(f"Loading TimesFM 2.5 from {weights_dir}...")
        snapshot_download(repo_id=model_id, local_dir=weights_dir)
        
        # Consistent filename for the mapped checkpoint
        ckpt_path = os.path.join(weights_dir, "torch_model_correctly_mapped.ckpt")
        
        if not os.path.exists(ckpt_path):
            safetensors_path = os.path.join(weights_dir, "model.safetensors")
            logger.info("Applying CORRECT state_dict mapping...")
            raw_sd = load_file(safetensors_path)
            
            # Map from HF/Safetensors names TO official TimesFM_2p5_200M_torch names
            mapping = {
                "decoder.input_ff_layer": "tokenizer",
                "decoder.layers": "stacked_xf",
                "decoder.horizon_ff_layer": "output_projection_point",
                "decoder.horizon_ff_layer_quantiles": "output_projection_quantiles",
                "self_attn.qkv_proj": "attn.qkv_proj",
                "self_attn.o_proj": "attn.out",
                "mlp.gate_proj": "ff0",
                "mlp.down_proj": "ff1",
                "input_layernorm": "pre_attn_ln",
                "mlp.layer_norm": "pre_ff_ln",
                "weight": "weight",
                "bias": "bias",
                "scale": "scale",
            }
            
            new_sd = {}
            for k, v in raw_sd.items():
                new_k = k
                # Handle prefixes and layer names
                for old_p, new_p in mapping.items():
                    if new_k.startswith(old_p):
                        new_k = new_k.replace(old_p, new_p, 1)
                        break
                
                # Handle internal layer mapping
                for old, new in mapping.items():
                    if "." + old in new_k:
                        new_k = new_k.replace("." + old, "." + new)
                
                # Specific fix for hidden_layer -> hidden_layer.0 (if model expects it, but our debug suggested it doesn't need .0)
                # Actually, the error said Missing: tokenizer.hidden_layer.weight
                # So we keep it as hidden_layer
                if "hidden_layer.0" in new_k:
                    new_k = new_k.replace("hidden_layer.0", "hidden_layer")
                
                new_sd[new_k] = v
            torch.save(new_sd, ckpt_path)

        self.model = timesfm.TimesFM_2p5_200M_torch(
            context_len=1024,
            horizon_len=128,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend="torch",
        )
        
        logger.info(f"Loading state_dict into model on {device}...")
        sd = torch.load(ckpt_path, map_location=device)
        missing, unexpected = self.model.model.load_state_dict(sd, strict=False)
        if missing: logger.warning(f"Missing keys: {missing[:5]}...")
        if unexpected: logger.warning(f"Unexpected keys: {unexpected[:5]}...")
        
        self.config = timesfm.ForecastConfig(
            max_context=1024,
            max_horizon=128,
            normalize_inputs=True,
        )
        self.model.compile(self.config)
        logger.info("TimesFM 2.5 initialized.")

    def impute(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        N, T, D = data.shape
        flat_data = data.transpose(0, 2, 1).reshape(N * D, T)
        flat_mask = mask.transpose(0, 2, 1).reshape(N * D, T)
        flat_imputed = flat_data.copy()

        needs_impute = np.where(np.any(flat_mask == 0, axis=1))[0]
        if len(needs_impute) == 0: return data

        for idx in tqdm(needs_impute, desc="TimesFM"):
            m = flat_mask[idx]
            indices = np.where(m == 0)[0]
            gaps = []
            if len(indices) > 0:
                s = indices[0]
                for i in range(1, len(indices)):
                    if indices[i] != indices[i-1] + 1:
                        gaps.append((s, indices[i-1]))
                        s = indices[i]
                gaps.append((s, indices[-1]))

            for rs, re in gaps:
                horizon = re - rs + 1
                if rs == 0:
                    obs = np.where(m == 1)[0]
                    flat_imputed[idx, :horizon] = flat_imputed[idx, obs[0]] if len(obs) > 0 else 0.0
                    continue
                
                prefix = flat_imputed[idx, :rs]
                try:
                    point, _ = self.model.forecast(horizon=horizon, inputs=[prefix])
                    flat_imputed[idx, rs:re+1] = point[0][:horizon]
                except:
                    flat_imputed[idx, rs:re+1] = prefix[-1] # LOCF fallback
        
        return flat_imputed.reshape(N, D, T).transpose(0, 2, 1)
