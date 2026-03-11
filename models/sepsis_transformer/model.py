import torch
import torch.nn as nn
import math
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional, List, Union

import torch.nn.functional as F

class ContextualGate(nn.Module):
    """
    Calculates a dynamic gate based on current patient state and medical context.
    Allows the model to 'open or close' the influence of the Knowledge Graph step-by-step.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x, context):
        # x: [B, T, H] | context: [B, T, H]
        concat = torch.cat([x, context], dim=-1)
        gate_weight = self.gate_net(concat) # [B, T, H]
        return (1.0 - gate_weight) * x + gate_weight * context

class KGITransformerLayer(nn.Module):
    """
    Custom Transformer Layer with internal Gated Knowledge Injection (DGI).
    Supports v1 (H, E) and v2 (Mask-Aware).
    """
    def __init__(self, hidden_dim, nhead, dim_feedforward, dropout, kgi_injector, mask_aware: bool = False):
        super().__init__()
        from models.saits.layers import FeatureContextualGate
        # 1. Self Attention
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        # 2. Knowledge Injection
        self.kgi_injector = kgi_injector
        self.gate = FeatureContextualGate(hidden_dim, mask_aware=mask_aware) if kgi_injector is not None else None
        self.norm2 = nn.LayerNorm(hidden_dim)

        # 3. Feed Forward Network
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, surviving_mask, medbert_dict, kgi_itemids):
        # --- Step 1: Temporal Self-Attention ---
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_out))

        # --- Step 2: Gated Knowledge Injection ---
        if self.kgi_injector is not None:
            # query_hidden: [B, T, H]
            kgi_context = self.kgi_injector(
                query_hidden=x, 
                surviving_mask=surviving_mask, 
                precomputed_embeddings=medbert_dict, 
                variable_indices=kgi_itemids
            )
            
            # CRITICAL FIX: The Transformer latent state x is [B, T, H].
            # surviving_mask is [B, T, D]. We need a mask [B, T, 1] for the gate.
            # We use the mean observation density as the gate's confidence signal.
            if surviving_mask is not None:
                gate_mask = surviving_mask.float().mean(dim=-1, keepdim=True) # [B, T, 1]
            else:
                gate_mask = None
                
            # Adaptive fusion via Mask-Aware Gate
            x = self.gate(x, kgi_context, mask=gate_mask)
            x = self.norm2(x)
        else:
            x = self.norm2(x)

        # --- Step 3: FFN ---
        ffn_out = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        x = self.norm3(x + self.dropout2(ffn_out))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class TimeSeriesTransformer(nn.Module):
    def __init__(self,
                 input_dim: Optional[int] = None,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 nhead: int = 8,
                 dropout: float = 0.1,
                 task_type: str = 'classification',
                 use_kgi: bool = False,
                 kgi_mode: str = 'dgi', # 'dki' (input), 'dgi' (layer v1), 'dgi_mask' (layer v2)
                 kgi_alpha_value: float = 1.0,
                 kgi_alpha_trainable: bool = True,
                 kgi_embedding_file: str = "data/embeddings/medbert_relation_embeddings_sepsis.pkl"):
        super().__init__()
        self.task_type = task_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.use_kgi = use_kgi
        self.kgi_mode = kgi_mode
        self.mask_aware = (kgi_mode == 'dgi_mask')
        self.kgi_alpha_value = kgi_alpha_value
        self.kgi_alpha_trainable = kgi_alpha_trainable
        self.kgi_embedding_file = kgi_embedding_file
        
        # Store input_dim for later initialization
        self.input_dim = input_dim
        self.initialized = False
        
        # Initialize model if input_dim is provided
        if input_dim is not None:
            self._initialize_model(input_dim)
            self.initialized = True
    
    def _initialize_model(self, input_dim):
        """Initialize the model architecture with the given dimensions"""
        # Input projection
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_dim)
        
        # INIZIALIZZIAMO IL DKI CONDIVISO
        if self.use_kgi:
            import os
            import pickle
            from models.saits.kgi_layer import DynamicKnowledgeInjector
            from models.saits.kgi_layer import KGIFusionLayer
            
            # The injector retrieves embeddings
            self.shared_kgi = DynamicKnowledgeInjector(text_embed_dim=768, hidden_dim=self.hidden_dim)
            
            # For DKI mode, we use a single fusion layer at the input
            if self.kgi_mode == 'dki':
                self.input_fusion = KGIFusionLayer(hidden_dim=self.hidden_dim)
            
            if not os.path.exists(self.kgi_embedding_file):
                print(f"Warning: {self.kgi_embedding_file} not found. DKI will wait for dictionary assignment.")
                self.medbert_dict = {}
            else:
                with open(self.kgi_embedding_file, "rb") as f:
                    self.medbert_dict = pickle.load(f)
            
            # Aligned ItemIDs for the 55 features
            self.kgi_itemids_full = [
                220045, 220179, 225310, 220181, 220210, 220277, 223762, # Vitals
                220074, 220059, 220060, 220061, 228368,                 # Hemodynamics
                225664, 50813, 220615, 225690, 227457, 220546, 223835, 223830, # Labs 1
                220224, 220235, 227442, 220645, 220602, 220635, 225625, 50804, # Labs 2
                220587, 220644, 220650, 227456, 227429, 227444, 220228, 220545, # Labs 3
                51279, 227465, 227466, 227467, 227443, 51006,                  # Labs 4
                226755, 228096,                                                # Neuro
                223834, 220339, 224686, 224687, 224697, 224695, 224696,        # Resp
                9991, 9992, 9993, 9994 # Urine, Norepi, Fluid, Vent (Grounded)
            ]
        else:
            self.shared_kgi = None
            self.medbert_dict = None

        # CREATE LAYERS: If DGI, use custom KGITransformerLayer, else standard Transformer blocks
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            if self.use_kgi and "dgi" in self.kgi_mode:
                layer = KGITransformerLayer(
                    hidden_dim=self.hidden_dim, 
                    nhead=self.nhead, 
                    dim_feedforward=self.hidden_dim * 4, 
                    dropout=self.dropout, 
                    kgi_injector=self.shared_kgi,
                    mask_aware=self.mask_aware
                )
            else:
                # Standard Transformer Layer (for Vanilla or DKI)
                layer = nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=self.nhead,
                    dim_feedforward=self.hidden_dim * 4,
                    dropout=self.dropout,
                    batch_first=True
                )
            self.layers.append(layer)
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1)
        )
        
        if self.task_type == 'classification':
            self.output_activation = nn.Identity() # Was Sigmoid
        else:
            self.output_activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the transformer model
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size,)
        """
        # Project input to hidden dimension
        x_proj = self.input_proj(x)
        
        # Add positional encoding
        x_proj = self.pos_encoder(x_proj)
        
        # Maschere per il KGI
        feature_dim = x.shape[-1] // 2
        surviving_mask = x[:, :, feature_dim:].bool() if self.use_kgi else None
        kgi_itemids = self.kgi_itemids_full[:feature_dim] if self.use_kgi else None
        
        # 3. KGI EARLY INJECTION (DKI Mode)
        if self.use_kgi and self.kgi_mode == 'dki':
            x_proj = self.input_fusion(x_proj, surviving_mask, self.medbert_dict, kgi_itemids)
        
        # 4. Passage through layers
        for layer in self.layers:
            if isinstance(layer, KGITransformerLayer):
                # DGI Mode (Injection inside each layer)
                x_proj = layer(
                    x_proj, 
                    surviving_mask=surviving_mask, 
                    medbert_dict=self.medbert_dict, 
                    kgi_itemids=kgi_itemids
                )
            else:
                # Vanilla or DKI Mode (Standard Transformer Layers)
                x_proj = layer(x_proj)
        
        # Take the output from the last timestep
        x_last = x_proj[:, -1, :]
        
        # Pass through output layers
        logits = self.output_layer(x_last)
        
        # Apply activation function
        probs = self.output_activation(logits)
        
        return probs.squeeze()
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            batch_size: int = 32, 
            epochs: int = 10, 
            learning_rate: float = 0.001,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            pos_weight: Optional[float] = None,
            save_path: Optional[str] = None):
        """
        Train the Transformer model
        """
        # ... rest of init logic ...
        # Initialize model if not already done
        if not self.initialized:
            self.input_dim = X.shape[2]  # Get input dimension from data
            self._initialize_model(self.input_dim)
            self.initialized = True
        
        self.to(self.device)
        
        # Convert input data to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Prepare validation data if provided
        val_dataloader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        if self.task_type == 'classification':
            if pos_weight is not None:
                pw_tensor = torch.tensor([pos_weight], device=self.device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
            else:
                criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.MSELoss()
        
        # Training loop
        self.train()
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation if data provided
            log_suffix = ""
            if self.use_kgi:
                # We can monitor the average gate weight if needed later
                pass

            if val_dataloader is not None:
                val_loss = self._validate(val_dataloader, criterion)
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}, Val Loss: {val_loss:.4f}{log_suffix}')
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                    patience_counter = 0
                    if save_path:
                        torch.save(best_model_state, save_path)
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    if best_model_state is not None:
                        self.load_state_dict(best_model_state)
                        self.to(self.device)
                    break
            elif (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}{log_suffix}')
    
        if val_dataloader is None and best_model_state is None:
            # no validation set, just keep the last epoch
            pass
        elif best_model_state is not None:
            self.load_state_dict(best_model_state)
            self.to(self.device)
    
    def _validate(self, val_dataloader, criterion):
        """Run validation and return validation loss"""
        self.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
        
        self.train()
        return total_val_loss / len(val_dataloader)
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Generate predictions using the trained model
        
        Args:
            X: Input features of shape (n_samples, n_timesteps, n_features)
            batch_size: Batch size for prediction
            
        Returns:
            Predictions array of shape (n_samples,)
        """
        self.to(self.device)
        self.eval()
        
        # Convert input to PyTorch tensor
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        predictions = []
        with torch.no_grad():
            for batch_X, in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self(batch_X)
                
                if self.task_type == 'classification':
                    outputs = torch.sigmoid(outputs)
                    
                # Ensure outputs are properly shaped before converting to numpy
                if outputs.ndim == 0:
                    outputs = outputs.unsqueeze(0)
                    
                # Convert to numpy and append to predictions
                batch_preds = outputs.cpu().numpy()
                
                # Ensure batch_preds is always at least 1D
                if batch_preds.ndim == 0:
                    batch_preds = np.array([batch_preds.item()])
                    
                predictions.append(batch_preds)
        
        # Concatenate all batch predictions
        if predictions:
            return np.concatenate(predictions)
        else:
            return np.array([]) 