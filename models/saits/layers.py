"""
SAITS Layers: Diagonally-Masked Self-Attention, Positional Encoding, FFN

Re-implemented from: Du et al., "SAITS: Self-Attention-based Imputation
for Time Series" (Expert Systems with Applications, 2023).

Key architectural details:
- Diagonal masking in self-attention prevents a position from attending
  to itself, forcing the model to learn from context.
- Two DMSA blocks produce intermediate imputations that are combined
  via learned weighted combination.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al., 2017).

    Adds position-dependent signals to the input embeddings so the
    model can learn temporal order.

    Args:
        d_model: embedding dimension
        max_len: maximum sequence length
        dropout: dropout rate
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute sinusoidal encoding table
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class DiagonallyMaskedMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with Diagonal Masking (DMSA).

    The diagonal mask prevents each time step from attending to itself,
    forcing the model to infer values from surrounding context.
    This is critical for imputation: we don't want the model to simply
    copy the input value.

    Args:
        n_heads: number of attention heads
        d_model: input/output dimension
        d_k: key dimension per head
        d_v: value dimension per head
        dropout: attention dropout rate
    """

    def __init__(
        self,
        n_heads: int = 4,
        d_model: int = 64,
        d_k: int = 16,
        d_v: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_K = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_V = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.fc_out = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Q, K, V: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
            attn_weights: (batch, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = Q.size()
        residual = Q

        # Linear projections → (batch, n_heads, seq_len, d_k or d_v)
        q = self.W_Q(Q).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_K(K).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_V(V).view(batch_size, seq_len, self.n_heads, self.d_v).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # (batch, n_heads, seq_len, seq_len)

        # Diagonal masking: prevent self-attention
        diag_mask = torch.eye(seq_len, device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(diag_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, v)  # (batch, n_heads, seq_len, d_v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.d_v)

        output = self.fc_out(context)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output, attn_weights


class AdaptiveGraphSpatialAttention(nn.Module):
    """
    Spatial Attention with Adaptive Graph Structure Learning (GSL).
    
    Instead of a fixed bias, the adjacency matrix A is learnable, initialized
    from the UMLS-derived prior P.
    
    Args:
        d_feature: number of features D (17)
        d_model: embedding dimension
        n_heads: number of heads
        dropout: dropout rate
    """
    def __init__(self, d_feature, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_feature = d_feature
        self.d_k = d_model // n_heads
        
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.fc_out = nn.Linear(d_model, d_model)
        
        # Learnable Adjacency Matrix
        # Will be initialized with P in forward or via a setter if P is not available at init
        # We start with a random specialized token or just zeros if P is provided later
        self.A_learn = nn.Parameter(torch.Tensor(d_feature, d_feature))
        nn.init.xavier_uniform_(self.A_learn) # Default init, will be overridden by P
        self.P_initialized = False

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, P):
        """
        Args:
            x: (batch, seq_len, d_feature, d_model)
            P: (d_feature, d_feature) relational prior
        """
        # Initialize A_learn with P if first pass
        if not self.P_initialized and P is not None:
             with torch.no_grad():
                 self.A_learn.copy_(P)
             self.P_initialized = True

        batch_size, seq_len, d_feat, d_model = x.size()
        residual = x
        
        # Merge batch and seq_len for feature attention: (B*T, D, d_model)
        x_flat = x.reshape(-1, d_feat, d_model)
        
        q = self.W_Q(x_flat).view(-1, d_feat, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_K(x_flat).view(-1, d_feat, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_V(x_flat).view(-1, d_feat, self.n_heads, self.d_k).transpose(1, 2)
        
        # scores: (B*T, n_heads, D, D)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply Adaptive Graph Bias
        # We use a softplus to enforce non-negativity if desired, or just raw logits
        # User suggested: Attn = Softmax(QK^T + A_learn)
        # We broadcast A_learn to (B*T, n_heads, D, D)
        
        # Enforce sparsity/non-negativity? 
        # Ideally A_learn should represent "connection strength". 
        # Using sigmoid or softplus is common. Let's use raw for now as typical bias.
        # But to interpret it as an adjacency, it's better if it's in a similar range.
        
        scores = scores + self.A_learn.unsqueeze(0).unsqueeze(0)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(-1, d_feat, d_model)
        
        output = self.fc_out(context)
        output = output.reshape(batch_size, seq_len, d_feat, d_model)
        
        return self.layer_norm(output + residual)


class PointWiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network with GELU activation.

    Two linear transformations with GELU non-linearity in between,
    followed by residual connection and layer normalization.

    Args:
        d_model: input/output dimension
        d_inner: hidden dimension
        dropout: dropout rate
    """

    def __init__(self, d_model: int = 64, d_inner: int = 128, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_inner)
        self.fc2 = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        residual = x
        out = self.fc1(x)
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return self.layer_norm(out + residual)


class DMSAEncoderLayer(nn.Module):
    """
    Single DMSA encoder layer = DiagonallyMaskedMHA + PointWiseFFN.

    Args:
        n_heads, d_model, d_k, d_v, d_inner, dropout: layer hyperparams
    """

    def __init__(
        self,
        n_heads: int = 4,
        d_model: int = 64,
        d_k: int = 16,
        d_v: int = 16,
        d_inner: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = DiagonallyMaskedMultiHeadAttention(
            n_heads=n_heads, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout
        )
        self.ffn = PointWiseFeedForward(d_model=d_model, d_inner=d_inner, dropout=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
            attn_weights: (batch, n_heads, seq_len, seq_len)
        """
        x, attn = self.attention(x, x, x)
        x = self.ffn(x)
        return x, attn


class DMSABlock(nn.Module):
    """
    Stack of DMSA encoder layers forming one DMSA block.

    Each block produces an intermediate imputation via a linear output layer.

    Args:
        n_layers: number of encoder layers in this block
        n_heads, d_model, d_k, d_v, d_inner, dropout: layer hyperparams
        d_feature: number of original features D (for output projection)
    """

    def __init__(
        self,
        n_layers: int = 3,
        n_heads: int = 4,
        d_model: int = 64,
        d_k: int = 16,
        d_v: int = 16,
        d_inner: int = 128,
        dropout: float = 0.1,
        d_feature: int = 9,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DMSAEncoderLayer(
                n_heads=n_heads, d_model=d_model, d_k=d_k, d_v=d_v,
                d_inner=d_inner, dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        # Output projection: d_model → D features
        self.output_proj = nn.Linear(d_model, d_feature)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x: (batch, T, d_model)

        Returns:
            hidden:   (batch, T, d_model) — final hidden representation
            imputed:  (batch, T, D) — intermediate imputation from this block
            attns:    list of attention weight tensors
        """
        attns = []
        for layer in self.layers:
            x, attn = layer(x)
            attns.append(attn)

        imputed = self.output_proj(x)  # (batch, T, D)
        return x, imputed, attns


class GraphDMSABlock(nn.Module):
    """
    SAITS DMSA Block with integrated Spatial-Temporal Attention and Graph Prior.
    """
    def __init__(self, n_layers, n_heads, d_model, d_k, d_v, d_inner, dropout, d_feature, parallel=True):
        super().__init__()
        self.d_feature = d_feature
        self.parallel = parallel
        # Spatial Embedding for each feature: concat(value, mask) -> d_model
        # We process each feature independently first
        self.feat_proj = nn.Linear(2, d_model)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'temporal': DiagonallyMaskedMultiHeadAttention(n_heads, d_model, d_k, d_v, dropout),
                'spatial': AdaptiveGraphSpatialAttention(d_feature, d_model, n_heads, dropout),
                'ffn': PointWiseFeedForward(d_model, d_inner, dropout)
            })
            for _ in range(n_layers)
        ])
        
        self.output_proj = nn.Linear(d_model, 1)
        self.gate_param = nn.Parameter(torch.tensor([0.0] * n_layers))

    def get_graph_structure(self):
        """Return list of learned adjacency matrices from all spatial layers."""
        return [layer['spatial'].A_learn for layer in self.layers]

    def forward(self, x_feat, P, feature_embeddings=None):
        """
        Args:
            x_feat: (batch, T, D, 2) - concatenated values and masks
            P: (D, D) relational prior
            feature_embeddings: (D, d_model) optional semantic embeddings
        """
        # 1. Project to embeddings: (B, T, D, d_model)
        x = self.feat_proj(x_feat)
        
        # Inject Semantic Embeddings
        if feature_embeddings is not None:
            # Broadcast over B and T: x è (B, T, D, d_model)
            x = x + feature_embeddings.unsqueeze(0).unsqueeze(0)
        
        attns = []
        for i, layer in enumerate(self.layers):
            if self.parallel:
                #### Parallel Fusion ####
                # Temporal
                B, T, D, C = x.shape
                x_temp_in = x.transpose(1, 2).reshape(B*D, T, C)
                # Unpacking (output, attention_weights)
                x_temp_out, attn_t = layer['temporal'](x_temp_in, x_temp_in, x_temp_in)
                x_temp_out = x_temp_out.reshape(B, D, T, C).transpose(1, 2)
                attns.append(attn_t)

                # Spatial
                x_space_out = layer['spatial'](x, P)
                
                # Gate Fusion
                alpha = torch.sigmoid(self.gate_param[i])
                x_combined = (1 - alpha) * x_temp_out + alpha * x_space_out
                
                # FFN
                x = layer['ffn'](x_combined)
            else:
                #### Sequential (Backward Compatible) ####
                # 1. Temporal
                B, T, D, C = x.shape
                x_temp_in = x.transpose(1, 2).reshape(B*D, T, C)
                x_temp_out, attn_t = layer['temporal'](x_temp_in, x_temp_in, x_temp_in)
                x = x_temp_out.reshape(B, D, T, C).transpose(1, 2)
                attns.append(attn_t)
                
                # 2. Spatial
                x = layer['spatial'](x, P)
                
                # 3. FFN
                x = layer['ffn'](x)
            
        # Final imputation projection: (B, T, D, d_model) -> (B, T, D)
        imputed = self.output_proj(x).squeeze(-1)
        
        # Pooling per compatibilità (media sulle feature)
        hidden_pooled = x.mean(dim=2)
        
        return hidden_pooled, imputed, attns
