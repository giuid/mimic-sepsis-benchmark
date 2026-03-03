import torch
import torch.nn as nn
from typing import Dict, Tuple
import math


class DynamicKnowledgeInjector(nn.Module):
    """
    Recupera gli embedding testuali UMLS e li aggrega dinamicamente
    usando l'attenzione guidata dallo stato numerico corrente (query_hidden),
    risolvendo il problema del "collasso semantico" della media matematica.
    
    Rispetto al TextualKnowledgeInjector originale (mean pooling):
    - La query è lo stato nascosto del modello al timestep corrente
    - Le relazioni UMLS vengono pesate in base alla loro rilevanza contestuale
    - Il risultato è già in hidden_dim (nessuna proiezione separata necessaria)
    
    Args:
        text_embed_dim: Dimensione degli embedding testuali (es. 768 per MedBERT)
        hidden_dim:     Dimensione del layer nascosto del modello
        top_k:          Se impostato, usa sparse attention — mantiene solo le
                        top-K relazioni per punteggio Q·K^T prima del softmax.
                        Default: None (soft attention su tutte le relazioni).
                        Valore consigliato: 28 (≈ C(8,2), mediana vitali ICU).
    """
    def __init__(self, text_embed_dim: int, hidden_dim: int, top_k: int = None):
        super().__init__()
        self.text_embed_dim = text_embed_dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        
        # 1. Layer di adattamento iniziale per il testo: text_embed_dim -> hidden_dim
        self.text_adapter = nn.Linear(text_embed_dim, hidden_dim)
        
        # 2. Proiezioni per il meccanismo di Attenzione Dinamica
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        
        # Fattore di scaling per stabilizzare i gradienti (1 / sqrt(d_k))
        self.scale = math.sqrt(hidden_dim)

    def forward(self, 
                query_hidden: torch.Tensor,
                surviving_mask: torch.Tensor, 
                precomputed_embeddings: Dict[Tuple[int, int], torch.Tensor],
                variable_indices: list) -> torch.Tensor:
        """
        Args:
            query_hidden:           [Batch, Time, Hidden_Dim] — stato numerico del modello
            surviving_mask:         [Batch, Time, Features]   — maschera delle osservazioni
            precomputed_embeddings: Dict{(id_a, id_b) -> [text_embed_dim]}
            variable_indices:       Indici globali (itemids)
            
        Returns:
            out_batch: [Batch, Time, Hidden_Dim] — contesto medico pesato per step
        """
        B, T, num_features = surviving_mask.shape
        device = surviving_mask.device
        
        # 1. Recupero/Inizializzazione Metadati Relazioni (Caching per velocità)
        # Identifichiamo quali coppie di feature hanno una relazione nel dizionario
        if not hasattr(self, "_rel_cache") or self._rel_cache["indices"] != variable_indices:
            f_idx_i, f_idx_j, embs = [], [], []
            for i in range(num_features):
                for j in range(i + 1, num_features):
                    key = tuple(sorted([variable_indices[i], variable_indices[j]]))
                    if key in precomputed_embeddings:
                        f_idx_i.append(i)
                        f_idx_j.append(j)
                        embs.append(precomputed_embeddings[key])
            
            if not embs:
                return torch.zeros(B, T, self.hidden_dim, device=device)
                
            self._rel_cache = {
                "indices": list(variable_indices),
                "embeddings": torch.stack(embs).to(device),
                "f_i": torch.tensor(f_idx_i, device=device),
                "f_j": torch.tensor(f_idx_j, device=device)
            }

        rel_embs = self._rel_cache["embeddings"] # [N_rel, text_embed_dim]
        f_i, f_j = self._rel_cache["f_i"], self._rel_cache["f_j"]
        
        # 2. Maschera delle Relazioni Attive: una relazione esiste se ambo le feature sono presenti
        # surviving_mask: [B, T, Features] -> [B, T, N_rel]
        rel_mask = surviving_mask[:, :, f_i] & surviving_mask[:, :, f_j]
        
        # 3. Proiezioni Vettorizzate
        # Adattiamo il testo: [N_rel, H]
        text_adapted = self.text_adapter(rel_embs)
        
        # Proiezioni Attention: query=[B,T,H], keys/values=[N_rel,H]
        Q = self.W_q(query_hidden)
        K = self.W_k(text_adapted)
        V = self.W_v(text_adapted)
        
        # Punteggi di affinità: [B, T, N_rel]
        # (Batch, Time, Hidden) x (N_rel, Hidden) -> (Batch, Time, N_rel)
        scores = torch.einsum('bth, kh -> btk', Q, K) / self.scale
        
        # Applichiamo la maschera (True=disponibile, False=-inf)
        scores = scores.masked_fill(~rel_mask, float('-inf'))
        
        # 4. Softmax / Sparse Attention
        # Gestione step vuoti: se non ci sono relazioni, softmax restituisce NaN.
        # Identifichiamo dove c'è almeno una relazione attiva
        any_rel_active = rel_mask.any(dim=-1, keepdim=True) # [B, T, 1]
        
        if self.top_k is not None and scores.shape[-1] > self.top_k:
            # Sparse Attention: solo top-K punteggi
            topk_vals, topk_idx = torch.topk(scores, min(self.top_k, scores.shape[-1]), dim=-1)
            sparse_scores = torch.full_like(scores, float('-inf'))
            sparse_scores.scatter_(-1, topk_idx, topk_vals)
            # Softmax sicuro (evita NaN su righe di soli -inf)
            final_scores = torch.where(any_rel_active, sparse_scores, torch.zeros_like(sparse_scores))
            attn_weights = torch.softmax(final_scores, dim=-1)
        else:
            # Soft Attention standard
            final_scores = torch.where(any_rel_active, scores, torch.zeros_like(scores))
            attn_weights = torch.softmax(final_scores, dim=-1)
            
        # Azzeriamo i pesi dove non c'è attività per sicurezza
        attn_weights = attn_weights * any_rel_active.float()
        
        # 5. Output: media pesata dei "valori" UMLS
        # weights=[B,T,N_rel] @ values=[N_rel,H] -> [B,T,H]
        out = torch.einsum('btk, kh -> bth', attn_weights, V)
        
        return out


class KGIFusionLayer(nn.Module):
    """
    Inietta conoscenza UMLS nello stato nascosto del modello.
    
    Pipeline:
        1. DynamicKnowledgeInjector: produce un contesto [B, T, H] dalle
           relazioni UMLS pesate dall'attenzione sullo stato corrente.
        2. Residual + LayerNorm: il contesto viene sommato allo stato originale
           per mantenere coerenza con il flusso di informazione del modello.

    Nota: non si usa un secondo strato MHA poiché DynamicKnowledgeInjector
    esegue già cross-attention query(numeric) -> key/value(text).
    """
    def __init__(self, hidden_dim: int, text_embed_dim: int = 768):
        super().__init__()
        self.text_injector = DynamicKnowledgeInjector(text_embed_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, 
                query_hidden: torch.Tensor, 
                surviving_mask: torch.Tensor, 
                precomputed_text_dict: dict, 
                variable_itemids: list):
        """
        Args:
            query_hidden:        [B, T, Hidden] — stato codificato della serie temporale
            surviving_mask:      [B, T, Features] — maschera delle osservazioni reali
            precomputed_text_dict: Dict di embedding UMLS pre-calcolati
            variable_itemids:    Lista di itemids corrispondenti alle feature
            
        Returns:
            fused: [B, T, Hidden] — stato arricchito con conoscenza medica
        """
        # DynamicKnowledgeInjector produce il contesto già in hidden_dim
        # usando query_hidden come guida per selezionare le relazioni rilevanti
        kgi_context = self.text_injector(
            query_hidden, surviving_mask, precomputed_text_dict, variable_itemids
        )
        
        # Residual + Norm: aggiunge il contesto allo stato originale
        fused = self.norm(query_hidden + kgi_context)
        
        # Salviamo il contesto per eventuale visualizzazione/analisi
        self.last_kgi_context = kgi_context.detach()
        
        return fused
