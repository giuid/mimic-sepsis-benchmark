import torch
import torch.nn as nn
from models.saits.model import SAITSModule
from models.saits.layers import DiagonallyMaskedMultiHeadAttention, PointWiseFeedForward, FeatureContextualGate

class GatedSemanticBlockAnalyser(nn.Module):
    """
    Copia locale del blocco DMSA che restituisce esplicitamente i gate weights.
    """
    def __init__(self, n_layers, n_heads, d_model, d_k, d_v, d_inner, dropout, d_feature, mask_aware: bool = False):
        super().__init__()
        self.feat_proj = nn.Linear(2, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'temporal': DiagonallyMaskedMultiHeadAttention(n_heads, d_model, d_k, d_v, dropout),
                'ffn': PointWiseFeedForward(d_model, d_inner, dropout)
            }) for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, 1)
        self.feature_gates = nn.ModuleList([FeatureContextualGate(d_model, mask_aware=mask_aware) for _ in range(n_layers)])

    def forward(self, x_feat, feature_embeddings=None):
        mask = x_feat[:, :, :, 1:2] if x_feat.dim() == 4 else None
        x = self.feat_proj(x_feat) if x_feat.dim() == 4 else x_feat.unsqueeze(2)
        
        B, T, D, H = x.shape
        if feature_embeddings is not None and feature_embeddings.dim() == 2:
            feature_embeddings = feature_embeddings.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        
        all_gate_weights = []
        for i, layer in enumerate(self.layers):
            x_in = x.transpose(1, 2).reshape(B*D, T, H)
            x_temp, _ = layer['temporal'](x_in, x_in, x_in)
            x_temp = x_temp.reshape(B, D, T, H).transpose(1, 2)
            x = x + x_temp
            
            if feature_embeddings is not None:
                # Modificata per catturare il peso
                # Fix dimensions: mask might be [B, T, D], need [B, T, D, 1]
                if mask is not None and mask.dim() == 3:
                    m_exp = mask.unsqueeze(-1)
                else:
                    m_exp = mask
                
                # Expand feature_embeddings to [B, T, D, H] if needed
                if feature_embeddings.dim() == 2:
                    f_emb_exp = feature_embeddings.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
                else:
                    f_emb_exp = feature_embeddings

                if self.feature_gates[i].mask_aware:
                    concat = torch.cat([x * m_exp, m_exp, f_emb_exp], dim=-1)
                else:
                    concat = torch.cat([x, f_emb_exp], dim=-1)
                
                gate_weight = self.feature_gates[i].gate_net(concat)
                x = (1.0 - gate_weight) * x + gate_weight * f_emb_exp
                all_gate_weights.append(gate_weight)
            
            x = x + layer['ffn'](x)
        
        imputed = self.output_proj(x).squeeze(-1)
        return x, imputed, all_gate_weights

class SAITSFeatureImportance(SAITSModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Sostituisco i blocchi originali con quelli analizzatori
        # (Mantenendo gli stessi parametri)
        block_kwargs = {
            "n_layers": self.hparams.n_layers, "n_heads": self.hparams.n_heads, 
            "d_model": self.hparams.d_model, "d_k": self.hparams.d_k, 
            "d_v": self.hparams.d_v, "d_inner": self.hparams.d_inner, 
            "dropout": self.hparams.dropout, "d_feature": self.hparams.d_feature, 
            "mask_aware": self.mask_aware
        }
        self.dmsa_block_1 = GatedSemanticBlockAnalyser(**block_kwargs)
        self.dmsa_block_2 = GatedSemanticBlockAnalyser(**block_kwargs)

    def forward(self, batch: dict) -> dict:
        data, input_mask = batch["data"], batch["input_mask"]
        surviving_mask = input_mask.bool() & (~batch.get("artificial_mask", torch.zeros_like(input_mask)).bool())
        x_feat = torch.stack([data * input_mask, input_mask], dim=-1)
        x_proj = self.pos_encoding(self.input_proj(torch.cat([data * input_mask, input_mask], dim=-1)))
        
        feature_embeddings = None
        if self.use_kgi and "dgi" in self.kgi_mode:
            feature_embeddings = self.kgi_injector(x_proj, surviving_mask, self.medbert_dict, self.kgi_itemids)

        h1, imp1, gw1 = self.dmsa_block_1(x_feat, feature_embeddings=feature_embeddings)
        x_replaced = data * input_mask + imp1 * (1 - input_mask)
        h2, imp2, gw2 = self.dmsa_block_2(torch.stack([x_replaced, input_mask], dim=-1), feature_embeddings=feature_embeddings)

        imp3 = self.combining_weight((imp1 + imp2) / 2.0)
        imp3 = data * input_mask + imp3 * (1 - input_mask)
        logits = self.classifier(h2[:, :self.hparams.obs_steps, :])
        
        return {"logits": logits, "gate_weights": [gw1, gw2]}
