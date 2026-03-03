import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class StochasticMasker(nn.Module):
    """
    Applies stochastic masking to time-series data during training.
    This simulates missingness to teach the model how to impute
    using the available variables and textual concepts.
    """
    def __init__(self, mask_ratio: float = 0.3):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, batch_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            batch_data: Tensor of shape [Batch, Time, Features] containing values.
                        Expected to have NaNs for naturally missing data.
                        
        Returns:
            masked_input: The input tensor with stochastic masking applied (NaNs kept).
            stochastic_mask: Boolean tensor, True where data was artificially masked.
            valid_mask: Boolean tensor, True where data was originally valid.
        """
        # 1. Identify which data points are ACTUAL observations (not NaN)
        valid_mask = ~torch.isnan(batch_data)
        
        # 2. Generate probability matrix only for valid data
        # During inference (eval mode), we don't mask anything stochastically.
        if not self.training:
            return batch_data, torch.zeros_like(valid_mask), valid_mask
            
        prob_matrix = torch.rand_like(batch_data)
        
        # 3. Create stochastic mask: mask only valid observations
        stochastic_mask = (prob_matrix < self.mask_ratio) & valid_mask
        
        # 4. Create masked input
        masked_input = batch_data.clone()
        # We fill artificially masked values with 0.0 or a specific token value
        # The model will need to learn to reconstruct these
        masked_input[stochastic_mask] = 0.0 
        
        return masked_input, stochastic_mask, valid_mask


class TextualKnowledgeInjector(nn.Module):
    """
    Retrieves and aggregates pre-computed text embeddings (MedBERT) 
    based on the variables that survived the stochastic masking.
    """
    def __init__(self, text_embed_dim: int, hidden_dim: int):
        super().__init__()
        self.text_embed_dim = text_embed_dim
        # Adapter to align MedBERT dimensionality to our model's hidden dim
        self.adapter = nn.Linear(text_embed_dim, hidden_dim)

    def forward(self, 
                surviving_mask: torch.Tensor, 
                precomputed_embeddings: Dict[Tuple[int, int], torch.Tensor],
                # variable_indices would map from a feature index (0-16) to its itemid
                variable_indices: list) -> torch.Tensor:
        """
        Args:
            surviving_mask: Boolean tensor [Batch, Time, Features] indicating 
                            what features the model actually "sees".
            precomputed_embeddings: Dict mapping pairs of (feature_idx1, feature_idx2) 
                                    to their MedBERT contextual embedding.
            variable_indices: List mapping feature column index to global concept ID.
            
        Returns:
            context_embeddings: [Batch, Time, Hidden_Dim]
        """
        # Note: In a real batch implementation, we'll want to vectorize this.
        # For this prototype, we outline the logic per batch item and time step.
        batch_size, seq_len, num_features = surviving_mask.shape
        device = surviving_mask.device
        
        # Initialize a default "empty context" tensor if no relations are found
        # In a fully vectorized version, this is pre-allocated
        context_batch = torch.zeros(batch_size, seq_len, self.text_embed_dim, device=device)
        
        # TODO: Vectorize this loop for performance. 
        # For now, it clearly shows the logic of "Retrieval based on surviving variables".
        for b in range(batch_size):
            for t in range(seq_len):
                # Which features are present at this time step for this patient?
                present_features = surviving_mask[b, t, :].nonzero(as_tuple=True)[0].tolist()
                
                step_context = []
                # Find all pairwise relations explicitly defined in our precomputed knowledge
                for i in range(len(present_features)):
                    for j in range(i + 1, len(present_features)):
                        feat_a = present_features[i]
                        feat_b = present_features[j]
                        
                        # Check bidirectional mapping (or sorting)
                        key = tuple(sorted([feat_a, feat_b]))
                        if key in precomputed_embeddings:
                            step_context.append(precomputed_embeddings[key].to(device))
                
                # If we found relations, aggregate them (e.g., Mean Pooling)
                if step_context:
                    # [Num_Relations, Text_Dim] -> [Text_Dim]
                    aggregated = torch.stack(step_context).mean(dim=0)
                    context_batch[b, t, :] = aggregated
        
        # Pass through adapter to align hidden dimensions
        return self.adapter(context_batch)


class KnowledgeGuidedImputation(nn.Module):
    """
    Main model bridging Numerical Time Series, SapBERT (structural identity),
    and MedBERT (contextual relations) using Cross-Attention.
    """
    def __init__(self, 
                 num_features: int, 
                 sapbert_dim: int, 
                 medbert_dim: int, 
                 hidden_dim: int, 
                 mask_ratio: float = 0.3):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        
        # Component 1: Stochastic Masking
        self.masker = StochasticMasker(mask_ratio=mask_ratio)
        
        # Component 2: Numerical Encoder (Simple Linear for now, can be Transformer/RNN)
        self.numerical_encoder = nn.Linear(num_features, hidden_dim)
        
        # Component 3: SapBERT Structural Identity
        # This maps the static SapBERT embedding of each variable to the hidden space
        self.sapbert_adapter = nn.Linear(sapbert_dim, hidden_dim)
        
        # Component 4: MedBERT Contextual Knowledge Injector
        self.medbert_injector = TextualKnowledgeInjector(text_embed_dim=medbert_dim, hidden_dim=hidden_dim)
        
        # Component 5: Fusion Cross-Attention
        # We will query with the Numerical+SapBERT representation
        # We will attend to (Key/Value) the MedBERT Context
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        # Component 6: Final Readout for Imputation
        self.readout = nn.Linear(hidden_dim, num_features)

    def forward(self, 
                batch_data: torch.Tensor, 
                sapbert_embeddings: torch.Tensor,
                precomputed_medbert_dict: Dict[Tuple[int, int], torch.Tensor],
                variable_indices: list):
        """
        Args:
            batch_data: [Batch, Time, Features] (Contains NaNs for true missing)
            sapbert_embeddings: [Features, SapBERT_Dim]. Static identity of each feature column.
            precomputed_medbert_dict: Dictionary of text embeddings for pairs of features.
            variable_indices: Meta-info for retrieval.
        """
        # 1. Apply Stochastic Masking
        # masked_input: the tensor with artificial holes.
        # stochastic_mask: where the artificial holes are (for Loss calculation)
        masked_input, stochastic_mask, valid_mask = self.masker(batch_data)
        
        # For numerical consistency, change NaNs to 0 in the masked input 
        # (models can't process NaNs in linear layers)
        safe_input = torch.nan_to_num(masked_input, nan=0.0)
        
        # 2. Encode Numerical Data
        h_numerical = self.numerical_encoder(safe_input) # [B, T, Hidden]
        
        # 3. Add SapBERT Structural Identity (Broadcasted across batch and time)
        # sapbert_embeddings is [F, SapBERT_Dim]
        # We process it through adapter -> [F, Hidden]
        h_sapbert = self.sapbert_adapter(sapbert_embeddings)
        
        # We need to pool or project [F, Hidden] to match [B, T, Hidden]
        # A simple approach: take the mean of SapBERT embeddings for the features
        # we are currently looking at. Or, element-wise addition if numerical encoder 
        # processes per-feature (like SAITS). 
        # For this standard MLP encoder, we'll pool the SapBERT vectors into a single 
        # vector representing the "set of variables in use" and add it to numerical.
        # (A more advanced fusion would concatenate Numerical [B, T, F, 1] + SapBERT [1, 1, F, Hidden])
        h_sapbert_pooled = h_sapbert.mean(dim=0) # [Hidden]
        
        # Query: Numerical State + Static Identity
        # Size remains: [B, T, Hidden]
        query = h_numerical + h_sapbert_pooled 
        
        # 4. Inject MedBERT Contextual Knowledge
        # The true "surviving" mask is the combination of originally valid data, 
        # MINUS the data we just stochastically masked.
        surviving_mask = valid_mask & (~stochastic_mask)
        
        # Retrieves and fuses UMLS text based ONLY on surviving features
        # Key/Value: [B, T, Hidden]
        context_kv = self.medbert_injector(surviving_mask, precomputed_medbert_dict, variable_indices)
        
        # 5. Cross-Attention Fusion
        # Attending from Numerical(Query) to Textual Context(Key/Value)
        attn_out, attn_weights = self.cross_attention(query=query, key=context_kv, value=context_kv)
        
        # Residual connection
        fused_representation = query + attn_out # [B, T, Hidden]
        
        # 6. Imputation Output
        imputed_values = self.readout(fused_representation) # [B, T, Features]
        
        return imputed_values, stochastic_mask, attn_weights
