# Approcci Baseline per Imputazione di Serie Temporali Cliniche
## Report Tecnico Dettagliato — SAITS & SSSD

---

## Indice

1. [Introduzione e Contesto](#1-introduzione-e-contesto)
2. [Il Problema dell'Imputazione nelle Time Series](#2-il-problema-dellimputazione-nelle-time-series)
3. [Mascheramento e Valutazione](#3-mascheramento-e-valutazione)
4. [SAITS — Self-Attention-based Imputation](#4-saits--self-attention-based-imputation)
5. [SSSD — Structured State Space Diffusion](#5-sssd--structured-state-space-diffusion)
6. [Confronto tra i Due Approcci](#6-confronto-tra-i-due-approcci)
7. [Riferimenti](#7-riferimenti)

---

## 1. Introduzione e Contesto

In ambiente ICU (Intensive Care Unit), i dati dei pazienti sono raccolti sotto forma di **serie temporali multivariate**: frequenza cardiaca, pressione sanguigna, SpO₂, valori di laboratorio, ecc. Questi dati sono intrinsecamente **incompleti** — i sensori possono staccarsi, i campioni di laboratorio si fanno solo ogni poche ore, e gli infermieri registrano i valori vitali a intervalli variabili.

L'**imputazione** è il processo di stima dei valori mancanti a partire da quelli osservati, ed è un passo fondamentale prima di poter addestrare qualsiasi modello predittivo (mortalità, sepsi, ecc.) sui dati ICU.

In questo repository implementiamo **due approcci stato dell'arte** che affrontano il problema con paradigmi fondamentalmente diversi:

| Aspetto | SAITS | SSSD |
|---------|-------|------|
| **Paradigma** | Deterministico (self-attention) | Generativo (diffusione) |
| **Paper** | Du et al., ESWA 2023 | Alcaraz & Strodthoff, TMLR 2023 |
| **Output** | Un singolo valore imputato | Distribuzione di possibili imputazioni |
| **Formazione** | Ottimizzazione diretta della ricostruzione | Predizione del rumore (denoising) |
| **Velocità** | Veloce (singolo forward pass) | Lento (1000 step di denoising per campione) |

---

## 2. Il Problema dell'Imputazione nelle Time Series

### Formulazione Matematica

Abbiamo un tensore di dati con forma **(N, T, D)** dove:
- **N** = numero di degenze ICU (pazienti)
- **T** = 48 timestep (prime 48 ore, risoluzione 1h)
- **D** = 9 variabili (HR, SBP, DBP, RR, SpO₂, Creatinina, Lattato, Glucosio, BUN)

Per ogni paziente, il dato osservato è **sparso**: solo circa il 50% delle celle (t, d) hanno un valore registrato.

### Le Due Maschere

Il sistema di mascheramento è la base dell'intera pipeline e presenta un concetto fondamentale da capire:

```
Maschera Originale (orig_mask):     Dati reali con valori mancanti del dataset
                                     1 = valore presente, 0 = valore mancante

Maschera Artificiale (artificial_mask): Valori che NASCONDIAMO per l'addestramento
                                         1 = nascosto artificialmente, 0 = invariato

Maschera di Input (input_mask):     Ciò che il modello effettivamente VEDE
                                     input_mask = orig_mask - artificial_mask
```

**Perché due maschere?** Perché non abbiamo il "ground truth" per i valori realmente mancanti. Quindi:
1. Prendiamo i valori **osservati** (orig_mask = 1)
2. Ne **nascondiamo** artificialmente una parte (artificial_mask = 1)
3. Addestriamo il modello a **ricostruire** i valori nascosti
4. Calcoliamo l'errore solo sui punti artificialmente mascherati

Questo è il cuore dell'approccio: il modello impara a ricostruire valori che conosciamo, e poi lo applichiamo ai valori che non conosciamo.

---

## 3. Mascheramento e Valutazione

### 3.1 Random Masking (RM)

Lo scenario più semplice: ogni valore osservato viene nascosto con probabilità *p*.

```python
class RandomMaskGenerator:
    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, orig_mask, rng=None):
        random_vals = rng.random(orig_mask.shape)
        # Maschera SOLO dove c'è un valore osservato
        artificial_mask = ((random_vals < self.p) & (orig_mask == 1)).astype(np.float32)
        return artificial_mask
```

**Nota chiave**: la condizione `orig_mask == 1` garantisce che non mascheriamo mai un valore già mancante. Senza questa condizione, il modello potrebbe "imparare" che i valori mancanti sono sempre zero.

### 3.2 Block Masking (BM)

Simula **interruzioni temporanee del monitoraggio** — un monitor si stacca per 10 ore consecutive e tutte le variabili diventano mancanti simultaneamente.

```python
class BlockMaskGenerator:
    def __init__(self, block_len=10, n_blocks=2, mask_all_features=True):
        ...

    def __call__(self, orig_mask, rng=None):
        for i in range(N):
            for _ in range(self.n_blocks):
                start = rng.integers(0, max(0, T - self.block_len))
                end = min(start + self.block_len, T)
                if self.mask_all_features:
                    # Maschera TUTTE le variabili nel blocco temporale
                    artificial_mask[i, start:end, :] = orig_mask[i, start:end, :]
```

Questo scenario è **molto più difficile** del random masking perché il modello non ha informazioni locali — deve "estrapolare" da ciò che viene prima e dopo il buco.

### 3.3 Feature-wise Masking (FW)

Simula il **guasto di un singolo sensore** per un periodo prolungato, ad esempio SpO₂ mancante per il 50% del tempo.

```python
class FeatureMaskGenerator:
    def __init__(self, feature_idx=-1, p_time=0.5):
        ...

    def __call__(self, orig_mask, rng=None):
        feat_idx = rng.integers(0, D) if self.feature_idx == -1 else self.feature_idx
        block_len = int(T * self.p_time)
        start = rng.integers(0, max(0, T - block_len) + 1)
        # Maschera solo UNA variabile per un blocco lungo
        artificial_mask[i, start:end, feat_idx] = orig_mask[i, start:end, feat_idx]
```

In questo scenario il modello deve sfruttare le **correlazioni tra variabili** — ad esempio, inferire SpO₂ dalla frequenza respiratoria e dalla frequenza cardiaca.

---

## 4. SAITS — Self-Attention-based Imputation

### 4.1 Idea Fondamentale

SAITS affronta l'imputazione come un problema di **ricostruzione diretta**: dato un input con valori mancanti, il modello produce direttamente i valori ricostruiti. L'intuizione è che il meccanismo di **self-attention** dei Transformer è perfettamente adatto per questo compito, perché può catturare relazioni sia **temporali** (tra diversi timestep) che **inter-variabile** (tra diversi sensori).

### 4.2 Architettura Completa

L'architettura di SAITS segue questo flusso:

```
Input: [dati con valori mancanti (B,T,D) | maschera (B,T,D)]
   │
   ▼
┌─────────────────────────┐
│ Input Projection (2D→64)│  Concatena valori + maschera, proietta in d_model=64
│ Positional Encoding     │  Aggiunge informazione temporale sinusoidale
└────────────┬────────────┘
             │
   ▼─────────┘
┌─────────────────────────┐
│    DMSA Block 1         │  3 layer di attenzione con maschera diagonale
│    → Imputazione X̂₁    │  Produce una prima stima dei valori mancanti
└────────────┬────────────┘
             │
   ▼─────────┘
   Sostituisce i valori mancanti con X̂₁
             │
   ▼─────────┘
┌─────────────────────────┐
│    DMSA Block 2         │  3 layer di attenzione (stessa architettura)
│    → Imputazione X̂₂    │  Seconda stima, più raffinata
└────────────┬────────────┘
             │
   ▼─────────┘
┌─────────────────────────┐
│  Combinazione Pesata    │  X̂₃ = combine(X̂₁, X̂₂)
│  → Output Finale X̂₃   │  Mantiene valori osservati, imputa solo i mancanti
└─────────────────────────┘
```

### 4.3 Componente Chiave: Diagonally-Masked Self-Attention (DMSA)

Il componente più innovativo di SAITS è la **maschera diagonale** nell'attenzione. In un Transformer standard, ogni posizione può "guardare" tutte le altre posizioni, **inclusa se stessa**. In SAITS, questo è esplicitamente proibito:

```python
class DiagonallyMaskedMultiHeadAttention(nn.Module):
    def forward(self, Q, K, V):
        # Proiezioni lineari standard (come in un Transformer)
        q = self.W_Q(Q).view(B, T, n_heads, d_k).transpose(1, 2)
        k = self.W_K(K).view(B, T, n_heads, d_k).transpose(1, 2)
        v = self.W_V(V).view(B, T, n_heads, d_v).transpose(1, 2)

        # Calcolo scores di attenzione: quanto ogni timestep "guarda" gli altri
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # ★ MASCHERA DIAGONALE: impedisce l'auto-attenzione ★
        diag_mask = torch.eye(T, device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(diag_mask, float("-inf"))
        #                              ↑
        #   Pone -∞ sulla diagonale → dopo softmax = 0
        #   Risultato: il timestep t=5 NON può copiare il proprio valore

        attn_weights = F.softmax(scores, dim=-1)  # Normalizza
        context = torch.matmul(attn_weights, v)    # Applica attenzione
```

**Perché è importante?** Senza questa maschera, il modello potrebbe semplicemente "copiare" il valore di input a se stesso. Con la maschera diagonale, è **forzato** a inferire ogni valore dal contesto circostante (altri timestep, altre variabili). Questo è esattamente ciò che vogliamo per l'imputazione: il modello deve **estrapolare** dai dati osservati, non copiarli.

### 4.4 Il Flusso a Due Blocchi (Imputation Iterativa)

SAITS usa **due blocchi DMSA in cascata** per un'imputazione iterativa:

```python
class SAITSModule(pl.LightningModule):
    def forward(self, batch):
        data = batch["data"]               # (B, T, D) i dati (0 dove mancano)
        input_mask = batch["input_mask"]    # (B, T, D) 1=visibile al modello

        # ═════════ BLOCCO 1: Prima Stima ═════════
        x_input = data * input_mask                    # Azzera i valori nascosti
        x_concat = torch.cat([x_input, input_mask], dim=-1)  # (B, T, 2D)
        x = self.input_proj(x_concat)                  # (B, T, d_model=64)
        x = self.pos_encoding(x)                       # Aggiunge info temporale

        hidden_1, imputed_1, attns_1 = self.dmsa_block_1(x)
        # imputed_1: (B, T, D) → prima stima dei valori

        # ═════════ BLOCCO 2: Raffinamento ═════════
        # Usa i valori originali dove osservati, la stima del blocco 1 dove mancano
        x_replaced = data * input_mask + imputed_1 * (1 - input_mask)
        x_concat_2 = torch.cat([x_replaced, input_mask], dim=-1)
        x2 = self.input_proj(x_concat_2)
        x2 = self.pos_encoding(x2)

        hidden_2, imputed_2, attns_2 = self.dmsa_block_2(x2)

        # ═════════ COMBINAZIONE FINALE ═════════
        imputed_3 = (imputed_1 + imputed_2) / 2.0       # Media
        imputed_3 = self.combining_weight(imputed_3)     # Pesi imparati
        imputed_3 = data * input_mask + imputed_3 * (1 - input_mask)
        # ↑ Mantiene i valori osservati intatti, imputa solo i mancanti
```

L'idea è che il **Blocco 1** produce un'imputazione approssimativa, poi il **Blocco 2** riceve un input "più completo" (perché i buchi sono stati riempiti dalla prima stima) e può quindi produrre imputazioni migliori. È un processo di **raffinamento iterativo**.

### 4.5 Loss Function: MIT + ORT

SAITS usa una **loss congiunta** composta da due termini:

```python
def _compute_loss(self, batch, outputs):
    target = batch["target"]               # Ground truth
    artificial_mask = batch["artificial_mask"]  # 1 = nascosto per training
    input_mask = batch["input_mask"]        # 1 = visibile al modello

    # ★ MIT: Masked Imputation Task ★
    # Errore sui valori NASCOSTI artificialmente
    # Questo è il compito principale: "quanto bene ricostruisci i buchi?"
    loss_mit_1 = mae_torch(outputs["imputed_1"], target, artificial_mask)
    loss_mit_2 = mae_torch(outputs["imputed_2"], target, artificial_mask)
    loss_mit_3 = mae_torch(outputs["imputed_3"], target, artificial_mask)
    loss_mit = (loss_mit_1 + loss_mit_2 + loss_mit_3) / 3.0

    # ★ ORT: Observed Reconstruction Task ★
    # Errore sui valori OSSERVATI (quelli che il modello può già vedere)
    # Questo agisce come "regolarizzazione": il modello deve anche
    # ricostruire correttamente ciò che ha visto
    loss_ort_1 = mae_torch(outputs["imputed_1"], target, input_mask)
    loss_ort_2 = mae_torch(outputs["imputed_2"], target, input_mask)
    loss_ort = (loss_ort_1 + loss_ort_2) / 2.0

    # ★ COMBINAZIONE con peso α = 0.8 ★
    # 80% MIT + 20% ORT → l'imputazione è la priorità
    loss = self.alpha * loss_mit + (1 - self.alpha) * loss_ort
```

**Perché anche ORT?** Senza il termine ORT, il modello potrebbe produrre buone imputazioni ma distorcere i valori osservati. ORT agisce come una **regolarizzazione** che mantiene il modello "onesto" — deve ricostruire fedelmente anche ciò che ha visto.

### 4.6 Iperparametri SAITS

| Parametro | Valore | Motivazione |
|-----------|--------|-------------|
| d_model | 64 | Dimensione embedding (dal paper) |
| d_inner | 128 | FFN hidden dim = 2 × d_model |
| n_heads | 4 | Teste di attenzione |
| d_k, d_v | 16 | d_model / n_heads |
| n_layers | 3 | Layer encoder per blocco DMSA |
| n_dmsa_blocks | 2 | Due blocchi per imputazione iterativa |
| dropout | 0.1 | Regolarizzazione |
| α (alpha) | 0.8 | Peso MIT nella loss |
| lr | 1e-3 | Learning rate (Adam) |
| Parametri totali | ~201K | Modello relativamente leggero |

---

## 5. SSSD — Structured State Space Diffusion

### 5.1 Idea Fondamentale

SSSD affronta l'imputazione con un approccio radicalmente diverso: è un **modello generativo** basato sulla **diffusione**. Invece di predire direttamente i valori mancanti, SSSD:

1. **Aggiunge rumore** progressivamente ai dati (processo forward)
2. **Impara a togliere quel rumore** (processo reverse, il denoising)
3. **A test time**, parte da rumore puro e progressivamente lo "pulisce" per produrre valori imputati

Questo approccio è lo stesso delle **Denoising Diffusion Probabilistic Models (DDPM)** usate per generare immagini (come DALL-E 2, Stable Diffusion), ma adattato alle serie temporali con l'uso di **Structured State Spaces (S4)** al posto delle U-Net convoluzionali.

### 5.2 Cos'è il Processo di Diffusione?

#### Forward Process (aggiunta di rumore)

Partendo dai dati puliti **x₀**, aggiungiamo progressivamente rumore gaussiano in **T = 1000 step**:

```
x₀ (dati puliti) → x₁ (poco rumore) → x₂ → ... → x₁₀₀₀ (rumore puro ≈ N(0,I))
```

Matematicamente, per ogni step *t* vale:

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$$

dove ε ~ N(0, I) è rumore gaussiano e ᾱ_t è un parametro della "noise schedule" che controlla quanto rumore c'è al timestep *t*.

```python
class DiffusionSchedule(nn.Module):
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        # Schedule lineare: β cresce linearmente da 1e-4 a 0.02
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)  # ᾱ_t = ∏ αᵢ

        # Pre-computa tutto per efficienza
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1 - alpha_bars))

    def q_sample(self, x_0, t, noise=None):
        """Forward process: campiona x_t da q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_0)

        # x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε
        x_t = self.sqrt_alpha_bars[t] * x_0 + self.sqrt_one_minus_alpha_bars[t] * noise
        return x_t, noise
```

#### Conditional Forward (D1 Setup) — La Chiave per l'Imputazione

A differenza della diffusione standard, in SSSD il rumore viene aggiunto **solo** alle posizioni da imputare. I valori osservati restano **intatti**:

```python
def conditional_q_sample(x_0, observed, mask, schedule, t, noise=None):
    """
    Diffusione condizionale: rumore SOLO dove mask=0 (da imputare).
    I punti osservati (mask=1) restano fissi.
    """
    x_t, noise = schedule.q_sample(x_0, t, noise)  # Applica rumore ovunque

    # ★ CONDIZIONAMENTO: mantieni i valori osservati ★
    x_t = mask * observed + (1 - mask) * x_t
    #     ↑ dove osservato    ↑ dove da imputare
    #     tieni il valore     usa il dato rumoroso
    #     originale

    return x_t, noise
```

Questo è il **setup D1** del paper SSSD: il modello sa sempre esattamente quali sono i valori veri (osservati) e deve solo "denoizzare" le posizioni mancanti.

#### Reverse Process (denoising iterativo)

A test time, partiamo da **rumore puro** nelle posizioni da imputare e percorriamo la catena al contrario (da t=1000 a t=0):

```python
@torch.no_grad()
def p_sample_loop(model_fn, shape, schedule, observed, mask, device, n_samples=1):
    """
    Loop completo di reverse diffusion per imputazione.

    T=1000: parte da rumore → rimuove rumore step by step → valori imputati
    """
    for _ in range(n_samples):
        # Inizia con rumore puro dove mancano i valori
        x = torch.randn(shape, device=device)
        x = mask * observed + (1 - mask) * x  # Condizionamento iniziale

        for t_idx in reversed(range(schedule.T)):  # Da 999 a 0
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)

            # Il modello predice il rumore aggiunto
            pred_noise = model_fn(x, t, observed, mask)

            # Calcola la media della distribuzione p(x_{t-1} | x_t)
            mean, log_var = schedule.p_mean_variance(pred_noise, x, t)

            if t_idx > 0:
                noise = torch.randn_like(x)
                x = mean + torch.exp(0.5 * log_var) * noise  # Campiona
            else:
                x = mean  # Ultimo step: prendi la media

            # ★ Ri-applica condizionamento ad ogni step ★
            x = mask * observed + (1 - mask) * x
```

> [!IMPORTANT]
> Il condizionamento `x = mask * observed + (1 - mask) * x` viene ri-applicato **ad ogni step** del reverse process. Questo è fondamentale per il D1 setup: i valori osservati non vengono mai corrotti durante l'inferenza.

### 5.3 Il Backbone: Structured State Spaces (S4)

Mentre SAITS usa **self-attention** per catturare le dipendenze temporali, SSSD usa i **Structured State Space Models (S4)**, un'alternativa che modella le sequenze come sistemi dinamici a tempo continuo.

#### Il Modello Continuo

S4 definisce un sistema lineare a tempo continuo:

```
x'(t) = A · x(t) + B · u(t)    ← evoluzione dello stato
y(t)  = C · x(t) + D · u(t)    ← output
```

dove:
- **u(t)** è l'input (il nostro segnale temporale)
- **x(t)** è lo stato nascosto (dimensione N = 128)
- **y(t)** è l'output del layer
- **A** è la matrice di transizione dello stato

La matrice **A** è inizializzata con la **HiPPO** (High-order Polynomial Projection Operator), che è specificamente progettata per catturare la storia del segnale. Nella nostra implementazione usiamo una **approssimazione diagonale** per efficienza:

```python
class S4Layer(nn.Module):
    def __init__(self, d_model, state_dim=128, ...):
        # A è parametrizzata come log-space + parte immaginaria
        # Questo garantisce stabilità (parte reale sempre negativa)
        self.log_A_real = nn.Parameter(torch.randn(d_model, state_dim) * 0.5)
        self.A_imag = nn.Parameter(torch.randn(d_model, state_dim) * 0.1)

        # B e C sono complessi
        self.B_real = nn.Parameter(torch.randn(d_model, state_dim) / state_dim)
        self.B_imag = nn.Parameter(torch.randn(d_model, state_dim) / state_dim)
        self.C_real = nn.Parameter(torch.randn(d_model, state_dim) / state_dim)
        self.C_imag = nn.Parameter(torch.randn(d_model, state_dim) / state_dim)

        # D: skip connection diretta input → output
        self.D = nn.Parameter(torch.randn(d_model))

        # Step di discretizzazione (imparabile)
        self.log_dt = nn.Parameter(torch.randn(d_model) * 0.01 - 1.0)
```

#### Convoluzione nel Dominio delle Frequenze (FFT)

Il vantaggio chiave di S4 è che il sistema lineare discretizzato equivale a una **convoluzione** — e le convoluzioni si calcolano efficientemente con la **Fast Fourier Transform (FFT)**:

```python
def _compute_kernel(self, L):
    """Calcola il kernel convolutivo K[k] = C · Ā^k · B per k=0..L-1"""
    dt = torch.exp(self.log_dt)

    # A continua, diagonale complessa
    A = -torch.exp(self.log_A_real) + 1j * self.A_imag
    # Discretizzazione: Ā = exp(A · Δt)
    A_bar = torch.exp(A * dt.unsqueeze(-1))

    B = self.B_real + 1j * self.B_imag
    C = self.C_real + 1j * self.C_imag

    # Calcolo delle potenze: Ā^0, Ā^1, ..., Ā^(L-1)
    powers = torch.arange(L)
    A_powers = A_bar.unsqueeze(-1) ** powers

    # Kernel = Σ_n C[n] · Ā^k[n] · B[n]
    kernel = ((C * B).unsqueeze(-1) * A_powers).sum(dim=1).real * dt.unsqueeze(-1)
    return kernel  # (d_model, L)

def forward(self, x):
    kernel = self._compute_kernel(L)

    # ★ Convoluzione via FFT: O(L log L) invece di O(L²) ★
    fft_len = 2 * L  # Padding per evitare artefatti circolari
    x_fft = torch.fft.rfft(x, n=fft_len, dim=-1)
    k_fft = torch.fft.rfft(kernel, n=fft_len, dim=-1)
    y = torch.fft.irfft(x_fft * k_fft, n=fft_len, dim=-1)[..., :L]

    # Skip connection
    y = y + self.D * x
```

> [!NOTE]
> La convoluzione via FFT ha complessità **O(L log L)**, rispetto all'**O(L²)** del self-attention. Per sequenze lunghe, S4 è significativamente più efficiente. Per le nostre sequenze corte (T=48), la differenza è minima, ma diventa importante se si aumenta la risoluzione temporale.

#### S4Block: Due Layer S4 per Blocco Residuale

Ogni blocco residuale di SSSD contiene un **S4Block** composto da due S4 layer:

```python
class S4Block(nn.Module):
    def __init__(self, d_model, state_dim=128, ...):
        self.s4_1 = S4Layer(d_model, state_dim, ...)
        self.s4_2 = S4Layer(d_model, state_dim, ...)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):  # (B, C, L)
        x = self.s4_1(x)               # Primo layer S4
        x = self.activation(x)         # Non-linearità GELU
        x = self.norm(x)               # Normalizzazione
        x = self.s4_2(x)               # Secondo layer S4
        return x
```

### 5.4 La Rete di Denoising Completa

Il **SSSDDenoiser** prende come input i dati rumorosi, i valori osservati, la maschera, e il timestep di diffusione, e predice il rumore ε:

```python
class SSSDDenoiser(nn.Module):
    def __init__(self, d_feature=9, residual_layers=36, 
                 residual_channels=256, skip_channels=256, ...):

        # Input: concatena (dati rumorosi, osservati, maschera) → 3 × D canali
        self.input_proj = nn.Conv1d(d_feature * 3, residual_channels, kernel_size=1)

        # Embedding del timestep di diffusione
        self.diffusion_embedding = DiffusionStepEmbedding(embed_dim=256)

        # 36 layer residuali con S4
        self.residual_layers = nn.ModuleList([
            SSSDResidualLayer(...) for _ in range(36)
        ])

        # Output: dai skip channels → predizione del rumore per ogni feature
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, skip_channels),
            nn.GELU(),
            nn.Conv1d(skip_channels, skip_channels, 1),
            nn.GELU(),
            nn.Conv1d(skip_channels, d_feature, 1),  # → D canali
        )
        # ★ Inizializzazione a zero: il modello parte predicendo "nessun rumore"
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)
```

Il denoiser ha **tre tipi di condizionamento**:
1. **Condizionamento sull'input**: il dato rumoroso è concatenato con i valori osservati e la maschera
2. **Condizionamento sul timestep**: l'embedding del timestep t viene addizionato ad ogni layer residuale
3. **Condizionamento strutturale**: la maschera dice al modello dove sono i valori veri e dove deve denoisare

#### Layer Residuale con Skip Connection

```python
class SSSDResidualLayer(nn.Module):
    def forward(self, x, diffusion_emb):
        h = self.norm(x)                          # GroupNorm

        # Aggiungi informazione sul timestep di diffusione
        d = self.diffusion_proj(diffusion_emb)    # (B, C)
        h = h + d.unsqueeze(-1)                   # Broadcast su L

        h = self.s4_block(h)                      # Processamento S4

        res = self.res_proj(h)                    # Per il prossimo layer
        skip = self.skip_proj(h)                  # Per l'output finale

        res = (res + x) / (2 ** 0.5)             # Residual + normalizzazione
        return res, skip
```

Le **skip connections** sono accumulate da tutti i 36 layer e poi processate insieme per l'output finale — questa architettura è ispirata a **WaveNet** e a **DiffWave**:

```
Layer 1 ──┬── residual ──→ Layer 2 ──┬── residual ──→ ... ──→ Layer 36
           │                           │
           └── skip ──────┐            └── skip ──────┐
                          │                            │
                          ▼                            ▼
                        Σ (accumula tutti gli skip)
                          │
                          ▼
                    Output Projection → Rumore Predetto ε_θ
```

### 5.5 Training di SSSD

Il training è sorprendentemente semplice:

```python
class SSSDModule(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        data = batch["data"]               # (B, T, D) ground truth
        input_mask = batch["input_mask"]   # 1 = visibile al modello
        observed = data * input_mask

        # 1. Campiona un timestep t casuale per ogni sample del batch
        t = torch.randint(0, 1000, (B,), device=self.device)

        # 2. Aggiungi rumore condizionale (solo dove mask=0)
        x_t, noise = conditional_q_sample(
            x_0=data, observed=observed, mask=input_mask,
            schedule=self.schedule, t=t
        )

        # 3. La rete predice quale rumore è stato aggiunto
        predicted_noise = self.denoiser(x_t, t, observed, input_mask)

        # 4. Loss: MSE tra rumore vero e rumore predetto
        #    Calcolata SOLO sulle posizioni artificialmente mascherate
        loss_mask = batch["artificial_mask"]
        loss = ((predicted_noise - noise)**2 * loss_mask).sum() / loss_mask.sum()
```

> [!TIP]
> A differenza di SAITS che minimizza l'errore di ricostruzione direttamente (MAE sui valori imputati), SSSD minimizza l'**errore di predizione del rumore** (MSE su ε). La qualità dell'imputazione emerge indirettamente dal fatto che, se il modello predice bene il rumore, il processo di denoising produrrà buone ricostruzioni.

### 5.6 Embedding del Timestep di Diffusione

Il modello deve sapere a quale livello di rumore sta operando. Questo è comunicato tramite un **embedding sinusoidale** (lo stesso principio del positional encoding nei Transformer):

```python
class DiffusionStepEmbedding(nn.Module):
    def __init__(self, embed_dim=256):
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),               # Swish activation
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t):
        # Encoding sinusoidale del timestep intero t
        half_dim = self.embed_dim // 2
        emb = torch.exp(torch.arange(half_dim) * -(log(10000) / (half_dim - 1)))
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([sin(emb), cos(emb)], dim=-1)

        return self.fc(emb)  # (B, 256)
```

### 5.7 Iperparametri SSSD

| Parametro | Valore | Motivazione |
|-----------|--------|-------------|
| residual_layers | 36 | Dal paper originale |
| residual_channels | 256 | Canali per layer |
| skip_channels | 256 | Canali skip connection |
| S4 state_dim | 128 | Dimensione dello stato S4 |
| S4 dropout | 0.2 | Regolarizzazione S4 |
| T (diffusion steps) | 1000 | Step di diffusione (standard DDPM) |
| β_start | 1e-4 | Noise schedule inizio |
| β_end | 0.02 | Noise schedule fine |
| lr | 2e-4 | Learning rate (più basso di SAITS) |
| inference_samples | 5 | Campioni mediati a test time |
| Parametri totali | ~20M+ | Modello molto grande |

---

## 6. Confronto tra i Due Approcci

### 6.1 Architettura e Paradigma

| Aspetto | SAITS | SSSD |
|---------|-------|------|
| **Tipo** | Deterministico | Generativo (probabilistico) |
| **Backbone** | Transformer (self-attention) | S4 (state space) |
| **Output** | 1 valore per posizione | Distribuzione → media di N campioni |
| **Complessità temporale** | O(T²) per l'attenzione | O(T log T) per S4 + 1000 step diffusione |
| **N. parametri** | ~201K | ~20M+ |
| **Speed training** | ~27 batch/s | ~1-2 batch/s |
| **Speed inference** | 1 forward pass | 1000 × forward pass (× N campioni) |

### 6.2 Punti di Forza e Debolezza

#### SAITS
**Punti di forza:**
- ✅ **Velocità**: Una singola forward pass produce l'imputazione
- ✅ **Semplicità**: Architettura chiara e facilmente interpretabile
- ✅ **Attenzione interpretabile**: I pesi di attenzione mostrano quali timestep/variabili il modello sta usando
- ✅ **Leggero**: Solo ~200K parametri

**Punti deboli:**
- ❌ **Deterministico**: Non fornisce incertezza sulla stima
- ❌ **Quadratico**: L'attenzione scala come O(T²), limitante per sequenze molto lunghe
- ❌ **Bias di copia**: Senza la maschera diagonale, tenderebbe a copiare l'input

#### SSSD
**Punti di forza:**
- ✅ **Generativo**: Produce una distribuzione → si può calcolare l'incertezza
- ✅ **Block masking**: Eccelle nel ricostruire lunghi segmenti mancanti
- ✅ **S4 backbone**: Cattura dipendenze a lungo raggio efficientemente
- ✅ **Flessibile**: La stessa architettura funziona per imputazione, forecasting, e generazione

**Punti deboli:**
- ❌ **Lento**: 1000 step di denoising × N campioni per ogni inferenza
- ❌ **Pesante**: ~20M parametri, richiede più GPU memory
- ❌ **Meno interpretabile**: Difficile capire "cosa" il modello ha imparato
- ❌ **Instabilità**: I modelli di diffusione possono essere instabili in training

### 6.3 Quando Usare Quale

| Scenario | Modello Consigliato | Motivazione |
|----------|-------------------|-------------|
| Missing casuale (sensori) | SAITS | Più veloce, performance simili |
| Blocchi di dati mancanti | SSSD | Meglio nel catturare contesto lungo |
| Stima con incertezza | SSSD | Produce distribuzioni |
| Tempo reale / bassa latenza | SAITS | Inferenza istantanea |
| Ricerca (confronto con CSDI, CoSTI) | SSSD | Stessa famiglia di modelli |
| Baseline rapida per nuovi dataset | SAITS | Setup più semplice |

### 6.4 Connessione ai Futuri Sviluppi

Entrambi i modelli sono stati progettati per essere **estensibili**:

- **GNN + UMLS conditioning**: Le relazioni tra variabili mediche (es. "creatinina → funzione renale → pressione") possono essere codificate come un grafo UMLS e integrate come conditioning aggiuntivo. Per SSSD, questo si inserisce naturalmente come ulteriore input al denoiser. Per SAITS, come bias nell'attenzione.

- **Consistency Models (CoSTI)**: Questi modelli accelerano il processo di diffusione da 1000 step a 1-2 step. La pipeline dati e il framework di valutazione rimangono identici — cambia solo il modello.

---

## 7. Riferimenti

1. **Du, W., Côté, D., & Liu, Y.** (2023). "SAITS: Self-Attention-based Imputation for Time Series." *Expert Systems with Applications*, 219, 119619.

2. **Alcaraz, J.M.L., & Strodthoff, N.** (2023). "Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models." *Transactions on Machine Learning Research (TMLR)*.

3. **Gu, A., Goel, K., & Ré, C.** (2022). "Efficiently Modeling Long Sequences with Structured State Spaces." *ICLR 2022*.

4. **Ho, J., Jain, A., & Abbeel, P.** (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS 2020*.

5. **Vaswani, A., et al.** (2017). "Attention Is All You Need." *NeurIPS 2017*.
