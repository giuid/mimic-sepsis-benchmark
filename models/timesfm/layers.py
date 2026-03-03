import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class GraphFeatureInteraction(nn.Module):
    """
    Learns relationships between features and facilitates information exchange 
    at the patch/token level in a multivariate setting.
    """
    def __init__(
        self, 
        d_feature: int, 
        d_model: int, 
        embedding_type: str = "vanilla",
        use_prior_init: bool = True
    ):
        super().__init__()
        self.d_feature = d_feature
        self.d_model = d_model
        
        # 1. Feature Embeddings (SapBERT/BERT/Random)
        self.embed_dim = 768
        if embedding_type == "sapbert":
            path = os.path.expanduser("~/Data/indus_data/physionet.org/files/mimiciv/3.1/embeddings/sapbert_embeddings_17.npy")
            if os.path.exists(path):
                emb_val = np.load(path)
            else:
                emb_val = np.random.normal(0, 0.1, size=(d_feature, self.embed_dim)).astype(np.float32)
        else:
            emb_val = np.random.normal(0, 0.1, size=(d_feature, self.embed_dim)).astype(np.float32)
            
        self.feature_embeddings = nn.Parameter(torch.from_numpy(emb_val).float())
        
        # 2. Learnable Adjacency Matrix A
        self.A = nn.Parameter(torch.zeros(d_feature, d_feature))
        
        # 3. Prior Initialization
        if use_prior_init:
            prior_path = "graph/artifacts_pruned/relational_prior.npy"
            if os.path.exists(prior_path):
                P = torch.from_numpy(np.load(prior_path)).float()
                # Initialize A close to P (inverse of sigmoid/relu would be better but let's just use P for now)
                with torch.no_grad():
                    self.A.copy_(P)
        
        # 4. Feature Projection (Internal)
        self.proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D, N_patches, d_model)
        Returns: (B, D, N_patches, d_model)
        """
        B, D, N, C = x.shape
        
        # Transition to (B, N, D, C)
        x = x.transpose(1, 2) # (B, N, D, C)
        
        # Softmax normalize adjacency
        adj = torch.softmax(self.A, dim=-1) # (D, D)
        
        # Message passing across features (D dimension)
        # x_interact = Adj @ x
        # Efficiently: (B, N, D, C) -> (B*N, D, C)
        x_flat = x.reshape(B * N, D, C)
        x_interact = torch.bmm(adj.unsqueeze(0).expand(B * N, -1, -1), x_flat)
        x_interact = x_interact.reshape(B, N, D, C)
        
        # Residual connection with Gating
        x_interact = self.proj(x_interact)
        gate_val = torch.sigmoid(self.gate(torch.cat([x, x_interact], dim=-1)))
        
        out = x + gate_val * x_interact
        
        # Back to (B, D, N, C)
        return out.transpose(1, 2)

    def get_adj(self):
        return torch.softmax(self.A, dim=-1)
