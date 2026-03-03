import torch
import torch.nn as nn
import math
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional, List, Union

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
                 kgi_embedding_file: str = "data/embeddings/medbert_relation_embeddings_sepsis.pkl"):
        super().__init__()
        self.task_type = task_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.use_kgi = use_kgi
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
        
        if self.use_kgi:
            import os
            import pickle
            from models.saits.kgi_layer import DynamicKnowledgeInjector
            self.step_kgi = DynamicKnowledgeInjector(text_embed_dim=768, hidden_dim=self.hidden_dim)
            if not os.path.exists(self.kgi_embedding_file):
                print(f"Warning: {self.kgi_embedding_file} not found. DKI will wait for dictionary assignment.")
                self.medbert_dict = {}
            else:
                with open(self.kgi_embedding_file, "rb") as f:
                    self.medbert_dict = pickle.load(f)
            
            # MIMIC-Sepsis 22 features (17 UMLS aligned + 5 aggregates without direct relational embeddings)
            self.kgi_itemids = [
                220045, 220210, 220277, 223762, 220179, 220180, 220052, # HR to MBP
                50931, 50813, 50912, 50885, 51265, 51301, 223835, 50820, 50821, 50818, # Labs
                -1, -2, -3, -4, -5 # GCS, Urine, Norepi, Fluid, Vent
            ]

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.nhead,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.num_layers)
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1)
        )
        
        if self.task_type == 'classification':
            self.output_activation = nn.Sigmoid()
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
        
        # Dynamic Knowledge Injection
        if self.use_kgi:
            # We assume input_dim is D_feature + D_mask
            feature_dim = x.shape[-1] // 2
            surviving_mask = x[:, :, feature_dim:].bool() # Extract masks as bool
            
            kgi_context = self.step_kgi(
                query_hidden=x_proj,
                surviving_mask=surviving_mask,
                precomputed_embeddings=self.medbert_dict,
                variable_indices=self.kgi_itemids
            )
            # Augment numeric token trajectory with textual relationship vectors
            x_proj = x_proj + kgi_context

        
        # Pass through transformer encoder
        x_out = self.transformer_encoder(x_proj)
        
        # Take the output from the last timestep
        x_last = x_out[:, -1, :]
        
        # Pass through output layers
        logits = self.output_layer(x_last)
        
        # Apply activation function
        probs = self.output_activation(logits)
        
        return probs.squeeze()
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            batch_size: int = 32, 
            epochs: int = 10, 
            learning_rate: float = 0.001,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Train the Transformer model
        
        Args:
            X: Input features of shape (n_samples, n_timesteps, n_features)
            y: Target values of shape (n_samples,)
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            validation_data: Optional tuple of (X_val, y_val) for validation
        """
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
        criterion = nn.BCELoss() if self.task_type == 'classification' else nn.MSELoss()
        
        # Training loop
        self.train()
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
            if val_dataloader is not None:
                val_loss = self._validate(val_dataloader, criterion)
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}, Val Loss: {val_loss:.4f}')
            elif (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
    
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