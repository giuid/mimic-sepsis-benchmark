# Enriched Graph Training Configuration Report

## 1. Overview
This report details the configuration of the 4 active training sessions currently running in the `benchmark` tmux session. These experiments are designed to validate the impact of **Enriched Structure Learning** and **Robust Normalization** on ICU time-series imputation.

**Session Status**:
- **GPU 4**: `prior_nullo` (Control: Random Adjacency)
- **GPU 5**: `sapbert_cignn` (Proposed: SapBERT + CI-GNN)
- **GPU 6**: `vanilla_saits` (Baseline: No Graph)
- **GPU 7**: `sssd` (Diffusion: Meta-Path Prior)

---

## 2. The Enriched Knowledge Graph
We have constructed a domain-specific knowledge graph linking the 17 MIMIC-IV clinical features used in our models.

### 2.1 Concepts (Nodes)
The graph nodes correspond to the 17 input features, mapped to their **UMLS CUI** (Concept Unique Identifier).
- **Vital Signs**: Heart Rate (C0018810), SBP (C0871470), DBP (C0428883), Respiration (C0231832), O2 Sat (C0483415), Temp (C0039476).
- **Labs**: Glucose (C0017725), Potassium (C0032821), Sodium (C0037473), Chloride (C0008203), Creatinine (C0010294), BUN (C0005845), WBC (C0023516), Platelets (C0005821), Hgb (C0019046), Hct (C0018935), Lactate (C0376261).

### 2.2 Connections (Edges)
The graph is built by querying the **UMLS Metathesaurus** (MRREL.RRF) for relationships between these concepts.
- **Enriched Filtering**: We recently expanded the valid Semantic Types (TUIs) to include:
    - *Clinical Attributes* (T201)
    - *Findings* (T033)
    - *Chemicals* (T109, T196)
    - *Biologic Functions* (T038)
- **Connectivity**: 
    - **Relational Prior**: 136/136 pairs connected (Full Coverage).
    - **Meta-Path Prior**: 51/136 pairs connected via "Hub" concepts (indirect relations).

---

## 3. Modeling Connections & Integration

Each model uses the graph differently during the **Generation Phase** (Imputation).

### 3.1 Prior Nullo (Control)
- **Concept**: Tests if *any* graph structure helps, or if the benefit comes purely from finding patterns in data.
- **Graph Usage**: **Random**.
    - The adjacency matrix $A$ is initialized randomly.
    - **No Graph Loss**: The model is NOT penalized for deviating from medical knowledge.
- **Integration**: 
    - **Spatial Attention**: $Attention(Q, K) = Softmax(\frac{QK^T}{\sqrt{d}} + A_{random})$
    - The model effectively learns its own arbitrary relationships from scratch.

### 3.2 SapBERT + CI-GNN (Proposed)
- **Concept**: Injects medical knowledge as a strong prior but allows the model to adapt based on data, constrained to be a Directed Acyclic Graph (DAG) for causality.
- **Graph Usage**: **Relational Prior $P$**.
    - $P_{ij} = \frac{1}{1 + dist(i, j)}$, where $dist$ is the weighted shortest path in UMLS.
    - **Adjacency Initialization**: $A_{learn}$ is initialized with $P$.
    - **Graph Loss**: $L_{graph} = ||A_{learn} - P||_F + \lambda_{DAG} \cdot Tr(e^{A_{learn}}) - d$.
- **Integration**:
    - **Semantic Initialization**: Feature Embeddings are initialized with **SapBERT** vectors (projected to $d_{model}$), injecting semantic meaning before the first layer.
    - **Adaptive Spatial Attention**: $Attention(Q, K) = Softmax(\frac{QK^T}{\sqrt{d}} + A_{learn})$
    - **Generation Phase**: The model uses $A_{learn}$ to bias attention. For example, when imputing *Lactate*, it explicitly attends more to *Oxygen Saturation* and *pH* (if available) because $A_{learn}$ has a high weight there, enforcing medically plausible imputations.

### 3.3 Vanilla SAITS (Baseline)
- **Concept**: Pure self-attention on time-series data.
- **Graph Usage**: **None**.
- **Integration**:
    - Uses standard DMSA (Diagonally Masked Self-Attention).
    - Learns correlations purely from the training data covariance.

### 3.4 SSSD (Diffusion)
- **Concept**: Generative diffusion model conditioned on observed data.
- **Graph Usage**: **Meta-Path Prior $M$**.
    - $M_{ij}$ counts paths like *Heart Rate* $\to$ *Cardiovascular System* $\to$ *Blood Pressure*.
- **Integration**:
    - **Conditioning**: The Diffusion model receives the observed data $x_{obs}$ and the Meta-Path matrix $M$ as conditional inputs.
    - **Generation Phase**: 
        - The model starts with noise $\epsilon$.
        - At each denoising step $t$, it predicts the noise $\epsilon_\theta(x_t, t, x_{obs}, M)$.
        - The Meta-Path matrix $M$ acts as a **structural bias** in the internal layers (via Cross-Attention or Concat), guiding the diffusion process to respect indirect clinical relationships (e.g., forcing chemically related labs to co-vary).

---

## 4. Summary Table

| Model | GPU | Graph Artefact | Init | Graph Loss | Integration Mechanism |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Prior Nullo** | 4 | None (Random) | Random | None | Random Bias in Spatial Attn |
| **SapBERT+CI-GNN** | 5 | Relational Prior $P$ | $P$ | $||A-P|| + DAG$ | SapBERT Emb + Learned Bias $A$ |
| **Vanilla SAITS** | 6 | None | N/A | None | Standard Self-Attention |
| **SSSD** | 7 | Meta-Path Prior $M$ | N/A | None | Conditional Input (Concat/Attn) |

---

## 5. Expected Outcome
We hypothesize that **SapBERT+CI-GNN** will outperform **Vanilla SAITS** and **Prior Nullo** by leveraging the enriched graph to make medically consistent imputations, especially when data is sparse (where correlation is hard to learn). **SSSD** should see improved stability and plausibility due to the Meta-Path conditioning.
