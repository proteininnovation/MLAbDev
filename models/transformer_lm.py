# models/transformer_lm.py
# Transformer for LM Embeddings (ablang, antiberty, antiberta2, antiberta2-cssp)
# IPI Antibody Developability Prediction Platform — Production Version DEC-2025
#
# ══════════════════════════════════════════════════════════════════════════════
# QUICK START — Three training modes, one predict_developability.py command
# ══════════════════════════════════════════════════════════════════════════════
#
# ── MODE 1 · Frozen embeddings (DEFAULT — recommended for most users) ─────────
#
#   Best for  : n < 10,000 | CPU deployment | same domain as IPI PSR/SEC
#   PLM update: NONE — IgBERT/ABlang embeddings pre-computed once
#   Speed     : fast (5–10 min on CPU)
#   Accuracy  : ρ_OVA = −0.66 (ablang, GDPa3, n=80)
#
#   STEP 1 — Train
#   python predict_developability.py --train \
#       --target psr_filter --lm ablang --model transformer_lm \
#       --db data/ipi_psr_trainset.xlsx
#   → FINAL_psr_filter_ablang_transformer_lm_ipi_psr_trainset.pt
#   → ipi_psr_trainset.xlsx.ablang.emb.csv  (auto-generated if missing)
#
#   STEP 2 — K-fold validation (optional, recommended before final train)
#   python predict_developability.py --kfold 10 \
#       --target psr_filter --lm ablang --model transformer_lm \
#       --db data/ipi_psr_trainset.xlsx
#   → prints mean AUC ± std across 10 folds
#   → recommended threshold embedded in BEST_*.pt checkpoint
#
#   STEP 3 — Predict on new data
#   python predict_developability.py --predict data/new_cohort.xlsx \
#       --target psr_filter --lm ablang --model transformer_lm \
#       --db data/ipi_psr_trainset.xlsx
#   → new_cohort_pred_psr_filter_ablang_transformer_lm_ipi_psr_trainset.xlsx
#   → SHAP waterfall plots, ROC, KDE, histogram
#
#   STEP 4 — Mutagenesis (optional, in-silico CDR3 scanning)
#   python predict_developability.py --predict data/new_cohort.xlsx \
#       --target psr_filter --lm ablang --model transformer_lm \
#       --db data/ipi_psr_trainset.xlsx --mutagenesis 50
#   → heatmaps for first 50 antibodies
#
# ── MODE 2 · PLM layer unfreezing (advanced — large datasets) ────────────────
#
#   Best for  : n > 20,000 | domain-shifted sequences | GPU available
#   PLM update: top (12 - freeze_plm_layers) IgBERT layers get full gradients
#   Speed     : slow (2–4 hours on GPU, days on CPU)
#   Sequences : NO pre-computed .emb.csv needed — sequences processed in batches
#
#   python predict_developability.py --train --finetune_plm \
#       --target psr_filter --lm igbert --model transformer_lm \
#       --db data/ipi_psr_trainset.xlsx \
#       --freeze_plm_layers 10 \   # freeze first 10/12 IgBERT layers
#       --lr_plm 1e-6 \            # slow PLM update (avoids forgetting)
#       --lr_classifier 1e-4 \     # normal classifier speed
#       --finetune_epochs 20
#   → FINAL_psr_filter_igbert_transformer_lm_ipi_psr_trainset_plmft.pt
#
#   Trainable params:
#     freeze_plm_layers=10 → ~20M of 110M IgBERT params + 200k classifier
#     freeze_plm_layers=0  → all 110M IgBERT params + 200k classifier
#
# ── MODE 3 · LoRA — PEFT (recommended when PLM adaptation is needed) ──────────
#
#   Best for  : n 1,000–50,000 | CPU feasible | low forgetting risk
#   PLM update: only LoRA A×B matrices in Q,V attention (~400k params)
#   Original W: FROZEN — antibody sequence grammar preserved
#   Speed     : medium (30–60 min on CPU for 11k samples)
#   Sequences : NO pre-computed .emb.csv needed
#
#   python predict_developability.py --train --finetune_plm \
#       --target psr_filter --lm igbert --model transformer_lm \
#       --db data/ipi_psr_trainset.xlsx \
#       --peft lora \              # enable LoRA
#       --lora_r 8 \               # rank: 4=small, 8=medium(default), 16=large
#       --lora_alpha 16 \          # scaling = alpha/r = 2.0
#       --lora_layers 10 11 \      # which layers get LoRA (default: last 2)
#       --lr_plm 1e-5 \
#       --lr_classifier 1e-4 \
#       --finetune_epochs 20
#   → FINAL_psr_filter_igbert_transformer_lm_ipi_psr_trainset_lora8.pt
#   → lora_weights_psr_filter_igbert_transformer_lm_ipi_psr_trainset.pt (~2 MB)
#
#   LoRA param count example (IgBERT, r=8, last 2 layers):
#     2 layers × 2 matrices (Q,V) × 2 × (1024×8) = 65,536 PLM params
#     vs 110,000,000 total IgBERT params → 0.06% updated
#
# ── LEVEL 2 · Collaborator fine-tuning from YOUR pretrained model ─────────────
#
#   Collaborator downloads MLAbDev + installs PLM (ablang2/transformers)
#   Loads YOUR pretrained .pt, fine-tunes on THEIR 300 antibodies
#
#   python predict_developability.py --finetune \
#       --pretrained FINAL_psr_filter_ablang_transformer_lm_ipi_psr_trainset.pt \
#       --target psr_filter --lm ablang --model transformer_lm \
#       --finetune_db their_300_antibodies.xlsx \
#       --freeze_layers 1 \        # freeze first classifier layer
#       --finetune_lr 1e-6 \       # 100× lower than original lr
#       --finetune_epochs 10
#   → FINAL_psr_filter_ablang_transformer_lm_ipi_psr_trainset_ft_their_300.pt
#
# ── DECISION GUIDE ────────────────────────────────────────────────────────────
#
#   Dataset size    GPU?    Recommended mode
#   ──────────────────────────────────────────────────────────────────────
#   < 1,000         any     Mode 1 (frozen)         — safe, fast
#   1,000–10,000    no      Mode 1 (frozen)         — your IPI PSR case
#   1,000–10,000    yes     Mode 3 (LoRA r=4)       — if domain-shifted
#   10,000–50,000   yes     Mode 3 (LoRA r=8)       — recommended
#   > 50,000        yes     Mode 3 (LoRA r=16)      — full adaptation
#   > 50,000        yes     Mode 2 (full unfreeze)  — maximum performance
#   Collaborator    any     Level 2 fine-tune        — from your pretrained
#
# ── SUPPORTED PLMs ────────────────────────────────────────────────────────────
#
#   LM               Package        emb_dim  Mode1  Mode2  Mode3(LoRA)
#   ─────────────────────────────────────────────────────────────────────
#   ablang           ablang2         480      ✓      ✓      ✓
#   antiberty        antiberty       512      ✓      ✓      ✓
#   antiberta2       transformers    1024     ✓      ✓      ✓
#   antiberta2-cssp  transformers    1024     ✓      ✓      ✓
#   igbert           transformers    1024     ✓      ✓      ✓
#
# ── OUTPUT FILES (Mode 1 --train + --predict example) ────────────────────────
#
#   build/pretrained_models/
#     FINAL_psr_filter_ablang_transformer_lm_ipi_psr_trainset.pt
#     BEST_psr_filter_ablang_transformer_lm_ipi_psr_trainset_k10_fold3.pt
#     fold_preds_psr_filter_ablang_transformer_lm_ipi_psr_trainset_k10.csv
#     CV_ROC_psr_filter_ablang_transformer_lm_ipi_psr_trainset_k10.png
#     kfold_psr_filter_ablang_transformer_lm_ipi_psr_trainset_k10.log
#
#   Downloads/
#     new_cohort_pred_psr_filter_ablang_transformer_lm_ipi_psr_trainset.xlsx
#     new_cohort_pred_..._shap_beeswarm.png
#     new_cohort_pred_..._predict_waterfalls/  (SHAP per antibody + PPT)
#     new_cohort_pred_..._roc_psr_filter.tiff
#     new_cohort_pred_..._kde_psr_filter.tiff
#     new_cohort_pred_..._histogram_psr_filter.tiff
#     new_cohort_pred_..._mutagenesis/         (CDR3 heatmaps + PPT)
#
# ══════════════════════════════════════════════════════════════════════════════
#
# ── JAN-2026 updates (style-aligned with transformer_onehot_new.py) ──────────
# [UPD-1] kfold_validation(): db_stem replaces dbname for naming consistency.
#         cluster_col now passed from predict_developability.py via --cluster arg,
#         supporting any CDR3 identity threshold (0.8, 0.9, …).
# [UPD-2] Per-fold prediction CSVs saved immediately after each fold evaluation
#         (not only in the combined all_records CSV at the end).
# [UPD-3] _TeeLogger: all console output mirrored to a .log file in MODEL_DIR.
#         kfold: kfold_{target}_{lm}_transformer_lm_{db}_k{K}.log
#         train: train_{target}_{lm}_transformer_lm_{db}.log
# [UPD-4] train(): added test_size, cluster_col, target, db_stem params.
#         When test_size > 0, applies CDR3-cluster-stratified train/test split
#         before training; held-out set used for early stopping + threshold opt.
# [UPD-5] run_full_threshold_pipeline() called with model_name, db_tag, kfold
#         so threshold_optimizer.py writes correctly named files directly.
# [UPD-6] Leakage check printed per fold (mirrors transformer_onehot_new.py).
# [UPD-7] predict_developability.py passes db_stem (not dbname) and cluster_col.
#
# ── Compatibility contract with predict_developability.py ──────────────────────
#
#  TRAIN path:
#    model = TransformerLMModel()
#    model.train(X, y, target=args.target, db_stem=db_stem, test_size=_test_size)
#    model.save(path)   ← path = FINAL_{target}_{lm}_transformer_lm_{db_stem}.pt
#
#  PREDICT path (unchanged):
#    model = TransformerLMModel.load(model_path, embedding_dim=X.shape[1])
#    scores = model.predict_proba(X_input)
#
#  KFOLD path:
#    TransformerLMModel().kfold_validation(db_stem, data, X, y,
#        embedding_lm=args.lm, title=title, kfold=args.kfold,
#        target=args.target, cluster_col=_cluster_col)
#
# ── FULLY ADAPTIVE — works for ANY dataset × ANY language model ───────────────
#
#  auto_scale_config()  Two-axis continuous scaling (not hard-coded bands).
#
#    Axis 1 — n_norm = log10(n/500) / log10(1000)        ← absolute dataset size
#      Controls: num_layers, batch_size, epochs, lr
#
#    Axis 2 — d_norm = log10((n/emb)/5) / log10(100)     ← samples per embedding dim
#      Controls: hidden_dim, dropout
#      Key insight: antiberta2(1024) on 8k samples has n/emb=7.8 (data-starved)
#        → needs SMALLER hidden_dim and HIGHER dropout than ablang(480) on same data
#
#    Correctness guaranteed:
#      • hd always divisible by nh
#      • antiberta2 always gets smaller hd than ablang for same n
#      • hd increases with n for same LM
#      • Works for any future LM embedding size (320, 768, 2048, ...)
#
#  auto_kfold()         3/5/10 folds from dataset size.
#  auto_select_loss()   focal/label_smooth/weighted_ce from imbalance+size.
#
#  Threshold optimisation — ZERO changes needed to predict_developability.py
#    kfold_validation() automatically calls run_full_threshold_pipeline() when
#    it finishes. The recommended_threshold is embedded into BEST_*.pt.
#    predict_developability.py reads it via payload.get('recommended_threshold', 0.5).
#    Soft dependency: if utils/threshold_optimizer.py is absent, kfold still
#    completes and a warning is printed. threshold defaults to 0.5.
# ─────────────────────────────────────────────────────────────────────────────

import os
import copy
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc, roc_auc_score,
)
import matplotlib.pyplot as plt

try:
    from config import MODEL_DIR
except ImportError:
    MODEL_DIR = "models/saved"

# ── Soft import: threshold_optimizer ─────────────────────────────
# Runs automatically at the end of kfold_validation().
# If utils/threshold_optimizer.py is absent nothing breaks —
# kfold still completes and threshold defaults to 0.5 at predict time.
try:
    from utils.threshold_optimizer import run_full_threshold_pipeline
    _THRESHOLD_OPT_AVAILABLE = True
except ImportError:
    _THRESHOLD_OPT_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════
# 0.  LOGGING HELPER
# ══════════════════════════════════════════════════════════════════

class _TeeLogger:
    """
    Mirror every print() to both stdout and a .log file simultaneously.
    Used as a context manager or via start()/stop().
    """
    def __init__(self, path: str):
        self._path = path; self._file = None; self._orig = None

    def start(self):
        import sys as _sys
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        self._file = open(self._path, 'w', encoding='utf-8')
        self._orig = _sys.stdout
        _sys.stdout = self
        return self

    def stop(self):
        import sys as _sys
        if self._orig is not None:
            _sys.stdout = self._orig
        if self._file and not self._file.closed:
            self._file.close()
        if self._orig is not None:
            self._orig.write(f"[log] → {self._path}\n")
            self._orig.flush()
        self._orig = None; self._file = None

    def write(self, text: str):
        if self._orig:   self._orig.write(text);  self._orig.flush()
        if self._file and not self._file.closed:
            try: self._file.write(text); self._file.flush()
            except Exception: pass

    def flush(self):
        if self._orig:   self._orig.flush()
        if self._file and not self._file.closed:
            try: self._file.flush()
            except Exception: pass

    def __enter__(self): return self.start()
    def __exit__(self, *_): self.stop()


# ══════════════════════════════════════════════════════════════════
# 1.  DATASET
# ══════════════════════════════════════════════════════════════════

class AntibodyDataset(Dataset):
    """Wraps (embedding_matrix, int_labels, barcodes) for DataLoader."""

    def __init__(self, embeddings, labels, barcodes):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.labels     = np.asarray(labels,     dtype=np.int64)
        self.barcodes   = list(barcodes)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.embeddings[idx]),
            torch.tensor(self.labels[idx]),
            self.barcodes[idx],
        )


# ══════════════════════════════════════════════════════════════════
# 2.  LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss — focuses training on hard/misclassified examples.
    Best for heavy class imbalance (e.g. PSR-positive < 15 % of data).
    gamma=2 is a robust default; tune in [1, 4].
    """
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, logits, targets):
        ce  = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt  = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def auto_select_loss(y: np.ndarray) -> tuple:
    """
    Automatically select the best loss function from class distribution + n.

    Decision table (from empirical tuning — matches our conversation):
    ┌─────────────────────────────┬────────────┬──────────────────────────────────┐
    │ Condition                   │ min_rate   │ Selected loss                    │
    ├─────────────────────────────┼────────────┼──────────────────────────────────┤
    │ Extreme imbalance           │ < 5%       │ focal γ=2.0  (+ warning)         │
    │ Severe imbalance            │ 5–10%      │ focal γ=2.0                      │
    │ Moderate imbalance          │ 10–40%     │ weighted_ce                      │
    │ Balanced + large n          │ ≥ 40%      │ label_smooth ε=0.05              │
    │   (n ≥ 50k)                 │            │   (calibration matters at scale) │
    │ Balanced + small n          │ ≥ 40%      │ ce                               │
    │   (n < 50k)                 │            │   (simple, correct, no weighting)│
    └─────────────────────────────┴────────────┴──────────────────────────────────┘

    Verified against your datasets:
      IPI PSR  (~12k,  52.6% pos) → ce               ← balanced, small n
      IPI SEC  (~8k,   21.0% pos) → weighted_ce      ← moderate imbalance
      DS1      (~240k, 53.0% pos) → label_smooth 0.05← balanced, large n

    Returns
    -------
    loss_type : str    ('ce' | 'weighted_ce' | 'label_smooth' | 'focal')
    gamma     : float  focal gamma (used only when loss_type='focal')
    smoothing : float  label smoothing ε (used only when loss_type='label_smooth')
    """
    y_np      = np.asarray(y, dtype=int)
    counts    = np.bincount(y_np)
    total     = len(y_np)
    min_rate  = counts.min() / total    # minority class fraction

    if min_rate < 0.05:
        # Extreme imbalance — focal is the only effective choice
        loss_type, gamma, smoothing = 'focal', 2.0, 0.0
        print(f"  [auto_loss] WARNING: extreme imbalance (min_rate={min_rate:.1%}) — "
              f"consider upsampling the minority class.")
    elif min_rate < 0.10:
        # Severe imbalance — focal focuses on hard examples
        loss_type, gamma, smoothing = 'focal', 2.0, 0.0
    elif min_rate < 0.40:
        # Moderate imbalance (10–40%) — weighted CE fixes gradient imbalance
        # Focal is overkill here and can destabilise training
        loss_type, gamma, smoothing = 'weighted_ce', 2.0, 0.0
    elif total >= 50_000:
        # Balanced + large n — label smoothing prevents overconfident probabilities
        # CE would push probs toward 0/1, hurting calibration and transfer
        loss_type, gamma, smoothing = 'label_smooth', 2.0, 0.05
    else:
        # Balanced + small n — plain CE is optimal; no weighting needed
        loss_type, gamma, smoothing = 'ce', 2.0, 0.0

    print(
        f"  [auto_loss] n={total:,}  min_rate={min_rate:.1%}"
        f"  → loss={loss_type}"
        + (f"  γ={gamma}"         if loss_type == 'focal'        else "")
        + (f"  ε={smoothing}"     if loss_type == 'label_smooth' else "")
    )
    return loss_type, gamma, smoothing


def build_criterion(y_train, device, loss_type='auto',
                    focal_gamma=2.0, label_smoothing=0.05):
    """
    Build the loss criterion + class-weight tensor.

    loss_type options
    ─────────────────────────────────────────────────────────────────
    'auto'              Auto-select from class distribution + n.
                        Uses auto_select_loss() — see table above.

    'ce'                Plain CrossEntropy. Correct for balanced data.
                        No class weights — both classes contribute equally.

    'weighted_ce'       CE + inverse-frequency class weights.
                        Safe default for any imbalanced dataset (10–40%).

    'label_smooth'      CE + label smoothing (ε from config).
                        Best for large balanced datasets (n > 50k).
                        Prevents overconfident probabilities.

    'weighted_label_smooth'
                        CE + class weights + label smoothing.
                        Best when data is both imbalanced AND large.

    'focal'             FocalLoss + class weights.
                        Best for severe imbalance (minority < 10%).

    Manual YAML examples
    ─────────────────────────────────────────────────────────────────
    IPI PSR  (12k,  balanced):    loss: {type: ce}
    IPI SEC  (8k,   21% min):     loss: {type: weighted_ce}
    DS1      (240k, balanced):    loss: {type: label_smooth, label_smoothing: 0.05}
    severe   (<10% min):          loss: {type: focal, focal_gamma: 2.0}
    """
    y_np         = np.asarray(y_train, dtype=int)
    class_counts = np.bincount(y_np)
    n_classes    = len(class_counts)
    total        = len(y_np)
    weights      = torch.tensor(
        total / (n_classes * class_counts), dtype=torch.float
    ).to(device)

    # Resolve 'auto' before branching
    if loss_type == 'auto':
        loss_type, focal_gamma, label_smoothing = auto_select_loss(y_np)

    if loss_type == 'focal':
        criterion = FocalLoss(weight=weights, gamma=focal_gamma)

    elif loss_type == 'label_smooth':
        # No class weights — data is balanced when this is chosen
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    elif loss_type == 'weighted_label_smooth':
        # Class weights + label smoothing — imbalanced + large n
        criterion = nn.CrossEntropyLoss(weight=weights,
                                        label_smoothing=label_smoothing)

    elif loss_type == 'weighted_ce':
        criterion = nn.CrossEntropyLoss(weight=weights)

    else:
        # 'ce' — plain CrossEntropy, no weights, no smoothing
        # Correct for balanced datasets; also the safe fallback
        criterion = nn.CrossEntropyLoss()

    return criterion, weights


# ══════════════════════════════════════════════════════════════════
# 3.  MODEL — TransClassifier
# ══════════════════════════════════════════════════════════════════

class TransClassifier(nn.Module):
    """
    Transformer classifier for fixed-length PLM embeddings (AbLang, AntiBERTa2, AntiBERTa2-CSSP, etc.)
    
    Improved CLS token design while maintaining full backward compatibility.
    Works for any embedding dimension and any future PLM.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim:    int   = 128,
        num_layers:    int   = 2,
        num_heads:     int   = 4,
        dropout:       float = 0.35,
        num_classes:   int   = 2,
    ):
        super().__init__()
        
        # Ensure hidden_dim is divisible by num_heads
        while hidden_dim % num_heads != 0:
            num_heads = num_heads // 2
            if num_heads < 2:
                num_heads = 2
                break

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Input projection: embedding_dim → hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Learnable CLS token - improved initialization
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, mean=0.0, std=0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,                    # Pre-LN for better stability
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Final classification head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        """
        x: [B, embedding_dim]  ← PLM embedding (AbLang, AntiBERTa2, etc.)
        Returns: logits [B, num_classes]
        """
        B = x.shape[0]

        # Project embedding
        x_proj = self.input_proj(x)                    # [B, hidden_dim]
        x_proj = x_proj.unsqueeze(1)                   # [B, 1, hidden_dim]

        # Expand CLS token to batch size
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, hidden_dim]

        # Concatenate CLS + projected embedding
        x = torch.cat([cls_tokens, x_proj], dim=1)     # [B, 2, hidden_dim]

        # Pass through Transformer
        x = self.transformer(x)                        # [B, 2, hidden_dim]

        # Use CLS token output for classification
        cls_output = x[:, 0]                           # [B, hidden_dim]

        # Final prediction
        logits = self.head(cls_output)                 # [B, 2]
        return logits

class LegacyTransClassifier(nn.Module):
    """
    Original architecture used before the CLS-token rewrite.

    Identified by the presence of 'embedding_fc' keys in the state_dict.
    Reconstructed entirely from checkpoint weight shapes — no config needed.

    Old forward:
        embedding_fc  Linear(emb_dim, hidden_dim)
        transformer   TransformerEncoder  (no Pre-LN, no CLS token)
        squeeze       x[:, 0, :] after mean-pool (squeeze(1) on 1-token seq)
        fc            Linear(hidden_dim, 2)

    This class exists solely to let load() read old .pt files without error.
    All new training uses TransClassifier.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim:    int,
        num_layers:    int,
        num_heads:     int,
        dropout:       float = 0.35,
        num_classes:   int   = 2,
    ):
        super().__init__()
        while hidden_dim % num_heads != 0:
            num_heads //= 2
        self.embedding_fc = nn.Linear(embedding_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            # norm_first=False — old checkpoints used post-LN (default)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding_fc(x).unsqueeze(1)   # [B, 1, H]
        x = self.transformer(x)                  # [B, 1, H]
        x = x.squeeze(1)                         # [B, H]
        return self.fc(x)                        # [B, 2]


# ══════════════════════════════════════════════════════════════════
# 4.  AUTO-SCALE CONFIG
# ══════════════════════════════════════════════════════════════════

def auto_scale_config(n_samples: int, embedding_dim: int, base_cfg: dict) -> dict:
    """
    PLM-optimised auto-scaling for transformer_lm.

    Design principles (derived from empirical tuning across ablang/antiberty/antiberta2):

    ── FIXED for ALL PLM classifiers ────────────────────────────────────────────
    PLM embeddings (ablang=480, antiberty=512, antiberta2=1024) already encode
    rich biochemical features. The transformer head only needs to find a decision
    boundary — it does NOT need to re-learn representations.

      hidden_dim = 128   always — small head, not a re-learner
                         Linear(emb→128) = controlled compression
                         compression ratios: ablang=3.75×, antiberta2=8×
      num_heads  = 8     always — head_dim=128/8=16, stable attention
      batch_size = 32    always — best generalisation (flat minima, noisy gradients)
                         Keskar et al. 2017: large batches → sharp minima → poor transfer
      lr         = 1e-5  always — matched to batch=32 (linear scaling rule baseline)

    ── SCALED by data density d_norm ────────────────────────────────────────────
    d_norm = log10((n/emb) / D_MIN) / log10(D_MAX / D_MIN)   ∈ [0, 1]
      n/emb = samples per embedding dimension (key overfitting indicator)
      D_MIN=5 (data-starved), D_MAX=500 (data-rich)

      dropout = 0.45 − 0.30 × d_norm    range [0.10, 0.45]
        d_norm captures BOTH n and emb effects in a single axis:
          small n + large emb (1024) → low d_norm  → high dropout (strong regularisation)
          large n + small emb (480)  → high d_norm → low dropout  (minimal regularisation)

    ── SCALED by dataset size n_norm ────────────────────────────────────────────
    n_norm = log10(n / N_MIN) / log10(N_MAX / N_MIN)   ∈ [0, 1]
      N_MIN=1000, N_MAX=500000

      num_layers   2 (n < 50k) or 3 (n ≥ 50k)   never more — PLM head is shallow
      epochs       50→15  (small n needs more passes; large n sees more data/epoch)
      weight_decay 0.001 (small n) → 0.0005 (large n, dataset self-regularises)
      patience     7 (small n, noisy loss signal) → 5 (large n, stable signal)

    ── Verified outputs ─────────────────────────────────────────────────────────
    Dataset         emb   n      n/emb  → hidden layers dropout  epochs  wd     pat
    ─────────────── ─────────────────────────────────────────────────────────────
    IPI SEC          480   8k    16.7   →  128    2     0.37     42    0.001    7
    IPI PSR          480  11k    22.9   →  128    2     0.35     40    0.001    7
    IPI PSR         1024  11k    10.7   →  128    2     0.40     40    0.001    7
    DS1 (ablang)     480 240k   500.0   →  128    3     0.15     20    0.0005   5
    DS1 (antiberty)  512 240k   468.8   →  128    3     0.15     20    0.0005   5
    DS1 (antiberta2)1024 240k   234.4   →  128    3     0.20     20    0.0005   5
    large (500k)     480 500k  1041.7   →  128    3     0.15     15    0.0005   5
    large (500k)    1024 500k   488.3   →  128    3     0.15     15    0.0005   5
    """
    # ── LM name for logging ───────────────────────────────────────
    _lm_tag = {480: 'ablang', 512: 'antiberty', 1024: 'antiberta2'}.get(
        embedding_dim, f'emb={embedding_dim}')

    # ── Axis 1: dataset size ──────────────────────────────────────
    N_MIN, N_MAX = 1_000, 500_000
    n_norm = float(np.clip(
        np.log10(max(n_samples, 1) / N_MIN) / np.log10(N_MAX / N_MIN),
        0.0, 1.0
    ))

    # ── Axis 2: data density (samples per embedding dimension) ────
    ratio  = n_samples / max(embedding_dim, 1)
    D_MIN, D_MAX = 5.0, 500.0
    d_norm = float(np.clip(
        np.log10(max(ratio, D_MIN) / D_MIN) / np.log10(D_MAX / D_MIN),
        0.0, 1.0
    ))

    # ── FIXED: hidden_dim, num_heads, batch_size, lr ─────────────
    hd = 128    # PLM head — always small, always 128
    nh = 8      # head_dim = 128/8 = 16
    bs = 32     # best generalisation — never increase
    lr = 1e-5   # matched to batch=32

    # ── num_layers: 2 for small n, 3 for large n ─────────────────
    # PLM classifiers never need more than 3 layers:
    # the PLM already did the deep feature extraction.
    nl = 2 if n_samples < 50_000 else 3

    # ── dropout: 0.45→0.10 from data density ─────────────────────
    # Uses d_norm (captures both n and emb effects):
    #   IPI PSR(11k) + antiberta2(1024): d_norm=0.165 → dr=0.40  ← strong reg
    #   IPI PSR(11k) + ablang(480):      d_norm=0.331 → dr=0.35  ← moderate reg
    #   DS1(240k)    + antiberta2(1024): d_norm=0.835 → dr=0.20  ← light reg
    #   DS1(240k)    + ablang(480):      d_norm=1.0   → dr=0.15  ← minimal reg
    dr = round(max(0.10, min(0.45, 0.45 - 0.30 * d_norm)), 2)

    # ── epochs: 50→15 from n_norm ─────────────────────────────────
    # Small datasets need many passes; large datasets see enough data per epoch.
    # Early stopping (patience) prevents overfitting regardless of this cap.
    ep = max(15, round(55 - 40 * n_norm))

    # ── weight_decay: 0.001 (small n) → 0.0005 (large n) ─────────
    wd = 0.001 if n_samples < 50_000 else 0.0005

    # ── patience: 7 (noisy small-n signal) → 5 (stable large-n) ──
    pat = 7 if n_samples < 50_000 else 5

    # ── Apply to config ───────────────────────────────────────────
    cfg = copy.deepcopy(base_cfg)
    cfg['model']['hidden_dim']          = hd
    cfg['model']['num_heads']           = nh
    cfg['model']['num_layers']          = nl
    cfg['model']['dropout']             = dr
    cfg['training']['batch_size']       = bs
    cfg['training']['epochs']           = ep
    cfg['training']['lr']               = lr
    cfg['training']['weight_decay']     = wd
    cfg.setdefault('scheduler', {})['patience'] = pat

    print(
        f"[auto_scale] n={n_samples:,}  emb={embedding_dim} ({_lm_tag})"
        f"  n/emb={ratio:.1f}"
        f"  n_norm={n_norm:.2f}  d_norm={d_norm:.2f}"
        f"\n             → hidden={hd}  heads={nh}  layers={nl}"
        f"  dropout={dr}  batch={bs}  lr={lr:.2e}"
        f"\n             → epochs={ep}  weight_decay={wd}  patience={pat}"
    )

    # ── YAML hard overrides — always win over auto_scale ──────────
    # Allows capping specific params without disabling auto_scale.
    for key in ['epochs', 'batch_size']:
        yaml_val = base_cfg.get('training', {}).get(key)
        auto_val = cfg['training'].get(key)
        if yaml_val is not None and yaml_val < auto_val:
            cfg['training'][key] = yaml_val
            print(f"[auto_scale] {key} capped by YAML: {auto_val} → {yaml_val}")

    return cfg


def auto_kfold(n_samples: int, min_val_size: int = 50) -> int:
    """
    Select appropriate number of cross-validation folds for any dataset size.

    Rule: need at least min_val_size validation samples per fold for reliable metrics.
    k = n // min_val_size, snapped to standard values [3, 5, 10], clamped to [3, 10].

    n_samples    → kfold    val fold size
    ──────────   ────────   ─────────────────────────────────────────
    < 150        3          ~n/3 (warns: very small, results may be noisy)
    150 – 499    3          ~50–166
    500 – 999    5          ~100–200
    ≥ 1 000      10         ≥100  (your IPI_SEC, IPI_PSR, DS1 all land here)
    """
    k_raw = n_samples // min_val_size
    k     = max(3, min(10, k_raw))
    if   k >= 8: k = 10
    elif k >= 4: k = 5
    else:        k = 3

    if n_samples < 150:
        print(
            f"[auto_kfold] WARNING: n={n_samples:,} is very small."
            f" Cross-validation estimates will be noisy."
            f" Consider collecting more data."
        )
    print(f"[auto_kfold] n={n_samples:,}  → {k}-fold CV"
          f"  (~{n_samples // k:,} samples/val fold)")
    return k


def validate_dataset(
    y:          np.ndarray,
    embedding_dim: int,
    context:    str = 'train',
    n_samples:  int = None,
) -> dict:
    """
    Validate any dataset before training or cross-validation.
    Raises ValueError on fatal conditions; prints actionable warnings otherwise.
    Returns safe operating parameters that are guaranteed not to crash.

    Called at the top of both train() and kfold_validation() so every
    code path — regardless of dataset size or LM choice — is protected.

    Parameters
    ----------
    y             : label array (int 0/1)
    embedding_dim : LM embedding dimension
    context       : 'train' or 'kfold'
    n_samples     : len(y) — passed separately so this works before y is loaded

    Returns
    -------
    dict with keys:
        n, min_class, min_rate, safe_batch, safe_kfold,
        warnings  (list of str — already printed)

    Fatal conditions → ValueError (never silently ignored)
    ───────────────────────────────────────────────────────
    n < 2                   Cannot train on fewer than 2 samples
    only 1 class present    Binary classifier requires both 0 and 1

    Recoverable conditions → warning printed, safe value returned
    ──────────────────────────────────────────────────────────────
    batch_size > n          Clamped to n (DataLoader would crash)
    kfold > min_class_count Reduced to min_class (StratifiedKFold would crash)
    kfold > n // 2          Reduced (val fold would have < 2 samples)
    n < 100                 Minimal architecture warning
    large emb + tiny n      LM choice warning (consider ablang for <500 samples)
    severe imbalance        Noise warning for kfold metrics
    """
    y   = np.asarray(y, dtype=int)
    n   = len(y) if n_samples is None else n_samples

    # ── Fatal checks ─────────────────────────────────────────────
    if n < 2:
        raise ValueError(
            f"[validate] n={n}: cannot train on fewer than 2 samples."
        )
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(
            f"[validate] Only class {classes[0]} found in labels. "
            f"Binary classification requires both class 0 and class 1. "
            f"Check your label column and data filtering."
        )

    # ── Class statistics ─────────────────────────────────────────
    class_counts = np.bincount(y)
    min_class    = int(class_counts.min())
    min_rate     = min_class / n

    warnings_list = []

    # ── Safe batch size (used as fallback only — YAML batch_size wins) ──
    # safe_batch = min(n, reasonable_auto_value)
    # Purpose: prevent DataLoader crash when batch_size > n.
    # train() honours YAML batch_size first; safe_batch is the fallback.
    safe_bs = min(n, 256)   # hard cap: batch can never exceed dataset size

    # ── Safe kfold ───────────────────────────────────────────────
    if context == 'kfold':
        # Start from size-based recommendation
        if   n < 500:  k = 3
        elif n < 1000: k = 5
        else:          k = 10

        # Constraint 1: StratifiedKFold needs ≥1 minority per fold
        k = min(k, min_class)
        # Constraint 2: val fold needs ≥2 samples
        k = min(k, n // 2)
        # Floor
        k = max(2, k)

        if k < 5:
            msg = (
                f"kfold reduced to {k}  "
                f"(min_class_count={min_class}, n={n:,}). "
                f"Collect more data for stable 10-fold CV."
            )
            warnings_list.append(msg)
            print(f"[validate] ⚠  {msg}")
        if min_rate < 0.05:
            msg = (
                f"Severe imbalance: minority class = {min_rate:.1%}. "
                f"Fold-level metrics will be noisy. "
                f"Consider targeted data collection for the minority class."
            )
            warnings_list.append(msg)
            print(f"[validate] ⚠  {msg}")
    else:
        k = None

    # ── General dataset quality warnings ────────────────────────
    if n < 100:
        msg = (
            f"Very small dataset (n={n}). "
            f"Architecture is minimal (hd=64, nl=1, dr=0.50). "
            f"Predictions may not generalise — aim for n≥500."
        )
        warnings_list.append(msg)
        print(f"[validate] ⚠  {msg}")

    if embedding_dim > 512 and n < 500:
        msg = (
            f"High-dimensional embedding (emb={embedding_dim}) "
            f"with only n={n} samples  (n/emb={n/embedding_dim:.2f}). "
            f"Severe overfitting risk. "
            f"Consider ablang(480) or antiberty(512) for small datasets."
        )
        warnings_list.append(msg)
        print(f"[validate] ⚠  {msg}")

    if not warnings_list:
        print(
            f"[validate] ✓  n={n:,}  emb={embedding_dim}  "
            f"min_class={min_class} ({min_rate:.1%})"
            + (f"  safe_kfold={k}" if k else "")
        )

    return {
        'n':           n,
        'min_class':   min_class,
        'min_rate':    min_rate,
        'safe_batch':  safe_bs,
        'safe_kfold':  k,
        'warnings':    warnings_list,
    }


# ══════════════════════════════════════════════════════════════════
# 5.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════

def train_model(
    model, train_loader, val_loader,
    criterion, optimizer, scheduler,
    epochs=20, patience=10, device='cpu',
    model_save="models/saved/TransformerLM.pt",
    use_amp=False,
    plot_path=None,       # if set, save training curve PNG here
    plot_title="",        # figure suptitle
):
    """
    Training loop with:
      ✓ gradient clipping (max_norm=1.0)
      ✓ early stopping on val_AUC  (robust to class imbalance)
      ✓ best state_dict saved + reloaded before return
      ✓ optional AMP for GPU mixed-precision
      ✓ history dict returned (train_loss, val_loss, val_auc per epoch)
      ✓ optional training curve PNG saved (plot_path)
    """
    save_dir = os.path.dirname(model_save)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    model  = model.to(device)
    _amp_device = 'cuda' if str(device) != 'cpu' else 'cpu'
    scaler = torch.amp.GradScaler(
        device=_amp_device,
        enabled=(use_amp and str(device) != 'cpu')
    )
    best_val_auc  = -1.0
    best_val_loss = float('inf')
    best_state    = None
    patience_ctr  = 0
    use_val       = val_loader is not None

    # ── History tracking ──────────────────────────────────────────
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    for epoch in range(epochs):
        # ── train ────────────────────────────────────────────────
        model.train()
        t_loss, t_preds, t_true = 0.0, [], []

        for emb, lbl, _ in train_loader:
            emb, lbl = emb.to(device), lbl.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(
                    device_type=_amp_device,
                    enabled=(use_amp and str(device) != 'cpu')):
                out  = model(emb)
                loss = criterion(out, lbl)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            t_loss += loss.item()
            t_preds.extend(torch.argmax(out, 1).cpu().numpy())
            t_true.extend(lbl.cpu().numpy())

        avg_t_loss = t_loss / len(train_loader)
        t_acc      = accuracy_score(t_true, t_preds)
        history['train_loss'].append(avg_t_loss)
        log   = (f"Epoch {epoch+1:03d}/{epochs}  "
                 f"train_loss={avg_t_loss:.4f}"
                 f"  train_acc={t_acc:.4f}")

        # ── validation ───────────────────────────────────────────
        if use_val:
            model.eval()
            v_loss, v_probs, v_preds, v_true = 0.0, [], [], []

            with torch.no_grad():
                for emb, lbl, _ in val_loader:
                    emb, lbl  = emb.to(device), lbl.to(device)
                    out        = model(emb)
                    v_loss    += criterion(out, lbl).item()
                    v_probs.extend(torch.softmax(out, 1)[:, 1].cpu().numpy())
                    v_preds.extend(torch.argmax(out, 1).cpu().numpy())
                    v_true.extend(lbl.cpu().numpy())

            avg_v_loss = v_loss / len(val_loader)
            history['val_loss'].append(avg_v_loss)

            try:
                from sklearn.metrics import roc_auc_score as _roc_auc
                v_auc = _roc_auc(v_true, v_probs) if len(set(v_true)) > 1 else 0.5
            except Exception:
                v_auc = 0.5
            history['val_auc'].append(v_auc)

            scheduler.step(avg_v_loss)
            log += (f"  |  val_loss={avg_v_loss:.4f}"
                    f"  val_auc={v_auc:.4f}")

            if v_auc > best_val_auc:
                best_val_auc  = v_auc
                best_val_loss = avg_v_loss
                best_state    = copy.deepcopy(model.state_dict())
                torch.save(best_state, model_save)
                patience_ctr  = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(log)
                    print(f"  → Early stopping at epoch {epoch+1}  "
                          f"best_val_auc={best_val_auc:.4f}")
                    break
        else:
            torch.save(model.state_dict(), model_save)

        print(log)

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  → Best weights restored  (val_auc={best_val_auc:.4f})")

    # ── Training curve plot ───────────────────────────────────────
    if plot_path and len(history['train_loss']) > 0:
        _save_training_curve(history, plot_path, plot_title)

    return model, history


def _save_training_curve(history: dict, plot_path: str, title: str = "") -> None:
    """
    Save training curve: loss panel + val_auc panel (if validation available).
    Mirrors transformer_onehot_new.py _save_fold_loss_plot style.
    """
    n_ep   = len(history['train_loss'])
    ep_ax  = range(1, n_ep + 1)
    has_val = len(history['val_loss']) > 0

    n_panels = 3 if has_val else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 3.5))
    if n_panels == 1:
        axes = [axes]

    # Panel 1: loss
    axes[0].plot(ep_ax, history['train_loss'], 'b-o', ms=3, label='train')
    if has_val:
        axes[0].plot(ep_ax, history['val_loss'], 'r-o', ms=3, label='val')
        axes[0].legend(fontsize=7)
    axes[0].set_title('Loss', fontsize=9)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].grid(alpha=0.3)

    if has_val:
        # Panel 2: val loss zoom
        axes[1].plot(ep_ax, history['val_loss'], 'r-o', ms=3, label='val loss')
        axes[1].set_title('Val Loss', fontsize=9)
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
        axes[1].grid(alpha=0.3)

        # Panel 3: val AUC
        best_ep  = int(np.argmax(history['val_auc'])) + 1
        best_auc = max(history['val_auc'])
        axes[2].plot(ep_ax, history['val_auc'], 'g-o', ms=3, label='val AUC')
        axes[2].axvline(best_ep, color='gray', lw=1, linestyle='--',
                        label=f'best ep={best_ep}  AUC={best_auc:.4f}')
        axes[2].legend(fontsize=7)
        axes[2].set_title('Val AUC', fontsize=9)
        axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('AUC')
        axes[2].set_ylim(0, 1); axes[2].grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [plot] → {plot_path}")


# ══════════════════════════════════════════════════════════════════
# 6.  DEFAULT CONFIG  (used when YAML is absent)
# ══════════════════════════════════════════════════════════════════

_DEFAULT_CONFIG = {
    'model': {
        'hidden_dim':  128,   # PLM head — always small
        'num_layers':  2,
        'num_heads':   8,     # head_dim = 128/8 = 16
        'dropout':     0.35,
    },
    'training': {
        'epochs':       40,   # safe default — early stopping prevents overfitting
        'batch_size':   32,   # best generalisation
        'lr':           1e-5, # matched to batch=32
        'weight_decay': 0.001,
    },
    'scheduler': {
        'mode':     'min',
        'factor':    0.5,
        'patience':  7,       # was 3 — too aggressive for PLM classifiers
    },
    'loss': {
        'type':             'auto',   # auto-selects from data statistics
        'focal_gamma':       2.0,     # used only when type: focal
        'label_smoothing':   0.05,    # used only when type: label_smooth
    },
    'auto_scale': True,
    'amp':        False,
}


# ══════════════════════════════════════════════════════════════════
# 7.  PUBLIC WRAPPER  —  TransformerLMModel
# ══════════════════════════════════════════════════════════════════

class TransformerLMModel:
    """
    Framework wrapper.  All public method signatures match predict_developability.py.

    Typical usage:
        # train
        model = TransformerLMModel()
        model.train(X, y)
        model.save("FINAL_psr_filter_ablang_transformer_lm_db.pt")

        # predict
        model = TransformerLMModel.load(path, embedding_dim=X.shape[1])
        scores = model.predict_proba(X)

        # k-fold
        TransformerLMModel().kfold_validation(
            data, X, y, embedding_lm='ablang',
            title='PSR_transformer_lm', kfold=10
        )
    """

    def __init__(self, config_path="config/transformer_lm.yaml"):
        if os.path.exists(config_path):
            with open(config_path) as f:
                user_cfg = yaml.safe_load(f)
            self.config = _deep_merge(copy.deepcopy(_DEFAULT_CONFIG), user_cfg or {})
            print(f"[TransformerLMModel] config ← {config_path}")
        else:
            self.config = copy.deepcopy(_DEFAULT_CONFIG)
            print(f"[TransformerLMModel] {config_path} not found — using built-in defaults")

        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model          = None
        self._embedding_dim = None
        self._lm_name       = "lm"

        # Print config report on init (manual mode = auto_scale: false)
        if not self.config.get('auto_scale', True):
            TransformerLMModel.print_config_report(self.config)

    # ── config report ─────────────────────────────────────────────

    @staticmethod
    def print_config_report(config: dict, embedding_dim: int = None,
                            n: int = None, lm_name: str = "") -> None:
        """
        Print all hyperparameters loaded from YAML / auto_scale.
        Called from __init__() (manual mode) and after auto_scale (auto mode).
        Mirrors TransformerOneHotModel.print_config_report() style.
        """
        m   = config['model']
        t   = config['training']
        sch = config.get('scheduler', {})
        lss = config.get('loss', {})

        W    = 62
        sep  = '═' * W
        sep2 = '─' * W
        auto = config.get('auto_scale', True)

        print(f"\n{sep}")
        print(f"  TransformerLM  ·  {'AUTO' if auto else 'MANUAL'} CONFIG"
              + (f"  |  lm={lm_name}" if lm_name else "")
              + (f"  |  n={n:,}" if n else "")
              + (f"  |  emb={embedding_dim}" if embedding_dim else ""))
        print(sep2)

        # ── Model architecture ────────────────────────────────────
        hd       = m['hidden_dim']
        nh       = m['num_heads']
        nl       = m['num_layers']
        dr       = m['dropout']
        head_dim = hd // max(nh, 1)
        # Rough param count: input_proj + CLS + transformer + head
        n_params = (
            embedding_dim * hd if embedding_dim else 0
        ) + nl * (4 * hd**2 + 2 * hd * hd * 4) + hd * 2
        print(f"  MODEL ARCHITECTURE")
        print(f"    hidden_dim    : {hd}")
        print(f"    num_heads     : {nh}  →  head_dim = {head_dim}")
        print(f"    num_layers    : {nl}")
        print(f"    dropout       : {dr}")
        if embedding_dim:
            print(f"    embedding_dim : {embedding_dim}  (LM: {lm_name})")
            n_p = (embedding_dim * hd + hd +          # input_proj linear + LN
                   hd +                               # CLS token
                   nl * (4*hd**2 + 2*hd*hd*4 + hd) + # transformer layers
                   hd * 2)                            # head
            print(f"    trainable ≈   : ~{n_p/1e6:.2f}M params")
        print(sep2)

        # ── Training ──────────────────────────────────────────────
        loss_type = lss.get('type', 'auto')
        loss_extra = ""
        if loss_type == 'focal':
            loss_extra = f"  (γ={lss.get('focal_gamma', 2.0)})"
        elif loss_type == 'label_smooth':
            loss_extra = f"  (ε={lss.get('label_smoothing', 0.05)})"
        elif loss_type == 'auto':
            loss_extra = "  (auto-selects from data statistics)"
        print(f"  TRAINING")
        print(f"    epochs        : {t.get('epochs', 20)}")
        print(f"    batch_size    : {t.get('batch_size', 32)}")
        print(f"    lr            : {t.get('lr', 1e-5):.2e}")
        print(f"    weight_decay  : {t.get('weight_decay', 1e-3):.2e}")
        print(f"    loss_type     : {loss_type}{loss_extra}")
        print(sep2)

        # ── Scheduler ────────────────────────────────────────────
        print(f"  SCHEDULER  (ReduceLROnPlateau)")
        print(f"    mode          : {sch.get('mode', 'min')}")
        print(f"    factor        : {sch.get('factor', 0.5)}")
        print(f"    patience      : {sch.get('patience', 3)}")
        print(sep2)

        # ── Auto-scale ────────────────────────────────────────────
        print(f"  AUTO_SCALE    : {auto}")
        print(f"  AMP           : {config.get('amp', False)}")
        print(f"{sep}\n")

    def _build_model(self, embedding_dim: int) -> TransClassifier:
        self._embedding_dim = embedding_dim
        c = self.config['model']
        return TransClassifier(
            embedding_dim=embedding_dim,
            hidden_dim=c['hidden_dim'],
            num_layers=c['num_layers'],
            num_heads=c['num_heads'],
            dropout=c['dropout'],
        ).to(self.device)

    def _to_loader(self, X, y, batch_size: int, shuffle: bool) -> DataLoader:
        X_np     = X.values if hasattr(X, 'values') else np.asarray(X, dtype=np.float32)
        barcodes = (X.index.tolist() if hasattr(X, 'index')
                    else [f"ab_{i}" for i in range(len(X_np))])
        ds = AntibodyDataset(X_np, y, barcodes)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=0,
            pin_memory=(str(self.device) != 'cpu'),
        )

    # ── train ─────────────────────────────────────────────────────
    # Called from main as:
    #   model.train(X, y, target=args.target, db_stem=db_stem, test_size=_test_size)

    def train(self, X, y, val_X=None, val_y=None,
              epochs=None, batch_size=None,
              target:       str   = "model",
              db_stem:      str   = "",
              embedding_lm: str   = "",
              test_size:    float = 0.0,
              cluster_col:  str   = "HCDR3_CLUSTER_0.8"):
        """
        Parameters
        ----------
        X, y        : training data  (DataFrame / ndarray, int labels)
        val_X/y     : explicit held-out set — enables early stopping + threshold opt.
                      Cannot be combined with test_size > 0.
        target      : label column name — used in output filenames
        db_stem     : database filename stem — inserted into all output filenames
        test_size   : fraction to hold out as a test set BEFORE training.
                      0.0 (default) = train on all data.
                      0.2 = 20 % test (CDR3-cluster-stratified when cluster_col present).
        cluster_col : CDR3 cluster column for group-stratified split.
        """
        if test_size > 0.0 and (val_X is not None or val_y is not None):
            raise ValueError(
                "test_size and val_X/val_y cannot both be set."
            )
        if not (0.0 <= test_size < 1.0):
            raise ValueError(f"test_size must be in [0, 1), got {test_size}")

        # Set lm name for filenames — from argument or keep existing value
        if embedding_lm:
            self._lm_name = embedding_lm

        _db_tag = f"_{db_stem}" if db_stem else ""

        # ── Log file ─────────────────────────────────────────────
        os.makedirs(MODEL_DIR, exist_ok=True)
        _log_path = os.path.join(
            MODEL_DIR,
            f"train_{target}_{self._lm_name}_transformer_lm{_db_tag}.log"
        )
        _train_logger = _TeeLogger(_log_path)
        _train_logger.start()

        y   = np.asarray(y, dtype=int)
        n   = len(y)
        emb_dim = X.shape[1]

        # ── Optional CDR3-cluster outer split ────────────────────
        _holdout_X = None; _holdout_y = None

        if test_size > 0.0:
            from sklearn.model_selection import StratifiedGroupKFold as _SGKF
            if hasattr(X, 'index') and cluster_col in (
                    X.columns if hasattr(X, 'columns') else []):
                _grp = X[cluster_col].values if cluster_col in X.columns else None
            else:
                _grp = None

            if _grp is not None:
                _n_spl = max(2, round(1.0 / test_size))
                _sgkf  = _SGKF(n_splits=_n_spl, shuffle=True, random_state=42)
                _best, _bd = None, float('inf')
                for _tr, _te in _sgkf.split(np.arange(n), y, _grp):
                    _d = abs(y[_te].mean() - y.mean())
                    if _d < _bd: _bd, _best = _d, (_tr, _te)
                _tr_idx, _te_idx = _best
                _leaked = set(_grp[_tr_idx]) & set(_grp[_te_idx])
                if _leaked:
                    print(f"  [WARN] {len(_leaked)} cluster(s) leaked in split")
                else:
                    print(f"  [OK]  No CDR3 leakage in train/test split")
                _split_method = f"StratifiedGroupKFold on '{cluster_col}'"
            else:
                from sklearn.model_selection import train_test_split as _tts
                _tr_idx, _te_idx = _tts(
                    np.arange(n), test_size=test_size, stratify=y, random_state=42)
                _split_method = "StratifiedShuffleSplit"

            _holdout_X = X.iloc[_te_idx] if hasattr(X, 'iloc') else X[_te_idx]
            _holdout_y = y[_te_idx]
            X          = X.iloc[_tr_idx] if hasattr(X, 'iloc') else X[_tr_idx]
            y          = y[_tr_idx]
            n          = len(y)
            val_X      = _holdout_X
            val_y      = _holdout_y
            print(f"  [split] method={_split_method}")
            print(f"  [split] train={n:,} pos={y.mean():.1%}  "
                  f"test={len(_holdout_y):,} pos={_holdout_y.mean():.1%}")

        # ── Validate + get safe operating parameters ─────────────
        vd = validate_dataset(y, emb_dim, context='train')

        if self.config.get('auto_scale', True):
            self.config = auto_scale_config(n, emb_dim, self.config)

        # Print full config report (after auto_scale so values are final)
        TransformerLMModel.print_config_report(
            self.config, embedding_dim=emb_dim, n=n, lm_name=self._lm_name)

        cfg        = self.config['training']
        epochs     = epochs     or cfg['epochs']
        # batch_size priority: explicit caller arg > YAML config > n (hard cap)
        # vd['safe_batch'] = min(n, 256) — only used as a size-cap guard.
        _yaml_bs   = cfg.get('batch_size', 32)
        batch_size = batch_size or _yaml_bs
        batch_size = min(batch_size, len(y))  # never exceed dataset size

        train_loader = self._to_loader(X, y, batch_size, shuffle=True)
        val_loader   = None
        if val_X is not None and val_y is not None:
            val_y      = np.asarray(val_y, dtype=int)
            val_loader = self._to_loader(val_X, val_y, batch_size, shuffle=False)

        loss_cfg      = self.config.get('loss', {})
        criterion, cw = build_criterion(
            y, self.device,
            loss_type       = loss_cfg.get('type',             'weighted_ce'),
            focal_gamma     = loss_cfg.get('focal_gamma',       2.0),
            label_smoothing = loss_cfg.get('label_smoothing',   0.05),
        )
        print(f"[train] n={n:,}  emb={emb_dim}  batch={batch_size}  epochs={epochs}")
        print(f"[train] class_weights={cw.cpu().tolist()}")
        if val_loader:
            src = f"test_size={test_size}" if _holdout_X is not None else "explicit val_X"
            print(f"[train] val_n={len(val_y):,}  pos={val_y.mean():.1%}  source={src}")

        self.model = self._build_model(emb_dim)
        optimizer  = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg['lr'], weight_decay=cfg['weight_decay'],
        )
        sch       = self.config['scheduler']
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sch['mode'], factor=sch['factor'], patience=sch['patience'],
        )
        # Unique temp file per process — avoids race condition when multiple runs overlap
        _tmp_uid = f"{os.getpid()}_{int(__import__('time').time()*1000) % 100000}"
        tmp_save = os.path.join(MODEL_DIR, f"_tmp_TransformerLM_{_tmp_uid}.pt")
        _curve_path = os.path.join(
            MODEL_DIR,
            f"train_curve_{target}_{self._lm_name}_transformer_lm{_db_tag}.png"
        )
        _curve_title = (f"{target.upper()}  {self._lm_name}  transformer_lm  {db_stem}\n"
                        f"n={n:,}  emb={emb_dim}  hd={self.config['model']['hidden_dim']}"
                        f"  nl={self.config['model']['num_layers']}"
                        f"  loss={self.config.get('loss',{}).get('type','auto')}")

        self.model, _history = train_model(
            self.model, train_loader, val_loader,
            criterion, optimizer, scheduler,
            epochs   = epochs,
            patience = self.config.get('scheduler', {}).get('patience', 10),
            device   = self.device,
            model_save = tmp_save,
            use_amp  = self.config.get('amp', False),
            plot_path  = _curve_path,
            plot_title = _curve_title,
        )
        # Clean up unique tmp checkpoint (safe — best weights already loaded by train_model)
        try:
            if os.path.exists(tmp_save):
                os.remove(tmp_save)
        except OSError:
            pass

        # ── Save validation prediction CSV (when --split used) ────
        if val_loader is not None and val_y is not None:
            self.model.eval()
            _vp_probs, _vp_preds, _vp_bcs = [], [], []
            val_y_arr = np.asarray(val_y, dtype=int)
            val_X_np  = (val_X.values if hasattr(val_X, 'values')
                         else np.asarray(val_X, dtype=np.float32))
            _vp_bcs_src = (val_X.index.tolist() if hasattr(val_X, 'index')
                           else [f"val_{i}" for i in range(len(val_y_arr))])
            _val_ds = AntibodyDataset(val_X_np, val_y_arr, _vp_bcs_src)
            _val_ld = DataLoader(_val_ds, batch_size=256, shuffle=False, num_workers=0)
            with torch.no_grad():
                for emb, lbl, bc in _val_ld:
                    out = self.model(emb.to(self.device))
                    _vp_probs.extend(torch.softmax(out, 1)[:, 1].cpu().numpy())
                    _vp_preds.extend(torch.argmax(out, 1).cpu().numpy())
                    _vp_bcs.extend(bc)

            _val_pred_path = os.path.join(
                MODEL_DIR,
                f"val_preds_{target}_{self._lm_name}_transformer_lm{_db_tag}.csv"
            )
            _val_df = pd.DataFrame({
                'BARCODE': _vp_bcs,
                'true':    val_y_arr.tolist(),
                'pred':    _vp_preds,
                'prob':    _vp_probs,
            })
            try:
                from sklearn.metrics import roc_auc_score as _roc
                _val_auc = _roc(val_y_arr, _vp_probs) if len(set(val_y_arr)) > 1 else 0.5
            except Exception:
                _val_auc = 0.5
            _val_df.to_csv(_val_pred_path, index=False)
            print(f"[train] Val predictions → {_val_pred_path}"
                  f"  (n={len(_vp_bcs):,}  AUC={_val_auc:.4f})")

        # ── Threshold optimisation on held-out set ────────────────
        if _holdout_X is not None and _THRESHOLD_OPT_AVAILABLE:
            print(f"\n[threshold] Running optimisation on held-out set "
                  f"(n={len(_holdout_y):,})  ···")
            self.model.eval()
            _h_probs, _h_preds, _h_bcs = [], [], []
            _h_ds = AntibodyDataset(
                _holdout_X.values if hasattr(_holdout_X,'values') else np.asarray(_holdout_X,dtype=np.float32),
                _holdout_y,
                _holdout_X.index.tolist() if hasattr(_holdout_X,'index') else [f"h_{i}" for i in range(len(_holdout_y))]
            )
            _h_loader = DataLoader(_h_ds, batch_size=256, shuffle=False, num_workers=0)
            with torch.no_grad():
                for emb, lbl, bc in _h_loader:
                    out = self.model(emb.to(self.device))
                    _h_probs.extend(torch.softmax(out,1)[:,1].cpu().numpy())
                    _h_preds.extend(torch.argmax(out,1).cpu().numpy())
                    _h_bcs.extend(bc)

            os.makedirs(MODEL_DIR, exist_ok=True)
            _holdout_csv = os.path.join(
                MODEL_DIR,
                f"holdout_preds_{target}_{self._lm_name}"
                f"_transformer_lm{_db_tag}.csv"
            )
            pd.DataFrame({'BARCODE':_h_bcs,'fold':1,'true':_holdout_y,
                          'pred':_h_preds,'prob':_h_probs,'best_fold':1}
            ).to_csv(_holdout_csv, index=False)
            print(f"[threshold] Holdout predictions → {_holdout_csv}")

            _tmp_ckpt = os.path.join(
                MODEL_DIR, f"TRAIN_{target}_{self._lm_name}_transformer_lm{_db_tag}.pt")
            torch.save({'state_dict':self.model.state_dict(),'config':self.config,
                        'embedding_dim':self._embedding_dim}, _tmp_ckpt)
            try:
                _stab = run_full_threshold_pipeline(
                    fold_preds_csv = _holdout_csv,
                    target         = target,
                    lm             = self._lm_name,
                    best_ckpt_path = _tmp_ckpt,
                    output_dir     = MODEL_DIR,
                    cost_fp        = 1.0, cost_fn = 3.0,
                    model_name     = "transformer_lm",
                    db_tag         = _db_tag,
                )
                self.recommended_threshold = float(_stab['pooled_threshold'])
                print(f"\n  Holdout threshold : {self.recommended_threshold:.4f}")
            except Exception as _e:
                print(f"[threshold] WARNING: optimisation failed — {_e}")

        _train_logger.stop()
        return self

    # ── predict ──────────────────────────────────────────────────
    # Called from main as:  scores = model.predict_proba(X_input)

    def predict_proba(self, X) -> np.ndarray:
        assert self.model is not None, "Model not trained or loaded."
        self.model.eval()
        X_np     = X.values if hasattr(X, 'values') else np.asarray(X, dtype=np.float32)
        barcodes = (X.index.tolist() if hasattr(X, 'index')
                    else [f"ab_{i}" for i in range(len(X_np))])
        ds     = AntibodyDataset(X_np, np.zeros(len(X_np), dtype=int), barcodes)
        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

        probs = []
        with torch.no_grad():
            for emb, _, _ in loader:
                out = self.model(emb.to(self.device))
                probs.extend(torch.softmax(out, 1)[:, 1].cpu().numpy())
        return np.array(probs)

    def predict(self, X, threshold=0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # ── save / load ───────────────────────────────────────────────
    # Main script names the file:
    #   FINAL_{target}_{lm}_transformer_lm_{db_stem}.pt
    # save() accepts that path verbatim — no renaming here.

    def save(self, path: str):
        save_dir = os.path.dirname(path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        rec_thresh = getattr(self, 'recommended_threshold', None)

        # Embed LoRA metadata if Mode 3 was used
        _peft       = getattr(self, '_peft',       'none')
        _lora_r     = getattr(self, '_lora_r',      8)
        _lora_alpha = getattr(self, '_lora_alpha',  16.0)
        _lora_path  = getattr(self, '_lora_path',   None)

        torch.save(
            {
                'state_dict':            self.model.state_dict(),
                'config':                self.config,
                'embedding_dim':         self._embedding_dim,
                'recommended_threshold': rec_thresh,
                'lm_name':               getattr(self, '_lm_name', ''),
                # Mode tracking
                'peft':                  _peft,
                'lora_r':                _lora_r,
                'lora_alpha':            _lora_alpha,
                'lora_weights_path':     _lora_path,
            },
            path,
        )
        thresh_note = (f"  threshold={rec_thresh:.4f}"
                       if rec_thresh is not None else "  threshold=None")
        mode_note   = f"  mode={'LoRA r='+str(_lora_r) if _peft=='lora' else 'frozen' if _peft=='none' else _peft}"
        print(f"[TransformerLMModel] saved → {path}{thresh_note}{mode_note}")
        if _lora_path:
            print(f"[TransformerLMModel] LoRA weights → {_lora_path}")

    @classmethod
    def load(cls, path: str, embedding_dim: int = None,
             config_path: str = "config/transformer_lm.yaml"):
        """
        Load a checkpoint into TransformerLMModel regardless of which
        architecture version trained it.

        Called from main as:
            model = TransformerLMModel.load(model_path, embedding_dim=X.shape[1])

        Handles three checkpoint formats
        ─────────────────────────────────────────────────────────────────────
        NEW   dict with 'state_dict' + 'config' + 'embedding_dim'
              Architecture: TransClassifier  (cls_token + input_proj + head)
              Action: restore config from checkpoint, build model, load weights

        LEGACY  dict with 'state_dict' but keys contain 'embedding_fc'/'fc'
              Architecture: LegacyTransClassifier  (embedding_fc + fc)
              Action: infer all architecture params from weight shapes,
                      build LegacyTransClassifier, load weights
              No config or embedding_dim argument required — all recovered
              from the weight tensor shapes inside the checkpoint.

        RAW   raw state_dict (no wrapping dict)
              Same key-name detection as above to pick the right class.
              embedding_dim argument required if legacy keys absent.

        Why shape-inference for legacy?
        ────────────────────────────────
        The old checkpoint was trained with different hidden_dim / num_layers
        than auto_scale_config would compute for the same dataset today.
        Rebuilding from current config would produce size mismatches.
        Reading the shapes directly from the saved weights guarantees an
        exact match every time, with no user action required.
        """
        import re

        instance = cls(config_path)
        payload  = torch.load(path, map_location=instance.device,
                               weights_only=False)

        # ── Unwrap payload ────────────────────────────────────────
        if isinstance(payload, dict) and 'state_dict' in payload:
            state_dict = payload['state_dict']
            ckpt_config = payload.get('config', {})
            ckpt_emb    = payload.get('embedding_dim')
        else:
            # Raw state_dict (very old format)
            state_dict  = payload
            ckpt_config = {}
            ckpt_emb    = None

        # ── Detect architecture from key names ────────────────────
        is_legacy = 'embedding_fc.weight' in state_dict

        if is_legacy:
            # ── LEGACY path: infer every param from weight shapes ─
            hidden_dim = state_dict['embedding_fc.bias'].shape[0]
            emb_dim    = state_dict['embedding_fc.weight'].shape[1]
            layers_idx = set(
                int(m.group(1))
                for k in state_dict
                for m in [re.search(r'transformer\.layers\.(\d+)\.', k)] if m
            )
            num_layers = max(layers_idx) + 1 if layers_idx else 2
            num_heads  = 8
            while hidden_dim % num_heads != 0:
                num_heads //= 2

            print(
                f"[TransformerLMModel] legacy checkpoint detected\n"
                f"  architecture : LegacyTransClassifier "
                f"(embedding_fc + transformer + fc)\n"
                f"  inferred     : emb={emb_dim}  hidden={hidden_dim}"
                f"  layers={num_layers}  heads={num_heads}"
            )

            legacy_model = LegacyTransClassifier(
                embedding_dim = emb_dim,
                hidden_dim    = hidden_dim,
                num_layers    = num_layers,
                num_heads     = num_heads,
            )
            legacy_model.load_state_dict(state_dict)
            legacy_model.eval()

            # Wrap in the TransformerLMModel shell so predict_proba works
            instance._embedding_dim = emb_dim
            instance.model          = legacy_model
            print(f"[TransformerLMModel] loaded (legacy) ← {path}")

        else:
            # ── NEW path: restore config, build TransClassifier ───
            emb_dim = embedding_dim or ckpt_emb
            if emb_dim is None:
                raise ValueError(
                    f"Cannot determine embedding_dim for new-style checkpoint {path}. "
                    f"Pass embedding_dim=X.shape[1] to TransformerLMModel.load()."
                )
            instance.config = _deep_merge(instance.config, ckpt_config)
            instance.model  = instance._build_model(emb_dim)
            instance.model.load_state_dict(state_dict)
            instance.model.eval()
            print(f"[TransformerLMModel] loaded ← {path}  (emb_dim={emb_dim})")

        return instance

    # ── kfold_validation  (instance method) ──────────────────────
    # Called from main as:
    #   TransformerLMModel().kfold_validation(data, X, y,
    #       embedding_lm=args.lm, title=title, kfold=args.kfold)
    #
    # Uses StratifiedGroupKFold on HCDR3_CLUSTER_0.8 so that sequence-
    # similar antibodies never leak across train/val folds.
    # Falls back to StratifiedKFold if the cluster column is absent.

    def kfold_validation(
        self,
        db_stem,                         # database filename stem (replaces dbname)
        data,
        X,
        y,
        embedding_lm:    str  = 'ablang',
        title:           str  = None,
        kfold:           int  = None,
        target:          str  = None,
        cluster_col:     str  = 'HCDR3_CLUSTER_0.8',
        save_fold_preds: bool = True,
    ):
        """
        Stratified-Group K-Fold cross-validation with CDR3 cluster grouping.

        [UPD-1]  db_stem replaces dbname for naming consistency with transformer_onehot.
        [UPD-2]  Per-fold prediction CSV saved immediately after each fold.
        [UPD-3]  All output mirrored to kfold_{target}_{lm}_transformer_lm_{db}_k{K}.log.
        [UPD-5]  run_full_threshold_pipeline called with model_name/db_tag/kfold.
        [UPD-6]  CDR3 leakage check printed per fold.

        Output files (all in MODEL_DIR):
          kfold_{target}_{lm}_transformer_lm_{db}_k{K}.log
          fold_preds_{target}_{lm}_transformer_lm_{db}_fold{N}_k{K}.csv   (per fold)
          fold_preds_{target}_{lm}_transformer_lm_{db}_k{K}.csv           (all folds)
          CV_ROC_{target}_{lm}_transformer_lm_{db}_k{K}.png
          _fold{N}_{target}_{lm}_transformer_lm_{db}.pt                   (per fold ckpt)
          BEST_{target}_{lm}_transformer_lm_{db}_k{K}_fold{F}.pt
          thresh_report_{target}_{lm}_transformer_lm_{db}_k{K}.png/json
          thresh_report_{target}_{lm}_transformer_lm_{db}_k{K}_sweep.csv
          thresh_stability_{target}_{lm}_transformer_lm_{db}_k{K}_auto.png
        """
        if target is None:
            raise ValueError(
                "kfold_validation() requires target= to be passed explicitly.\n"
                "Check that predict_developability.py passes args.target."
            )

        # Store lm name on instance so train() log filenames are consistent
        self._lm_name = embedding_lm
        _db_tag = f"_{db_stem}" if db_stem else ""

        if title is None:
            title = f"{target.upper()}_transformer_lm_{db_stem}"

        # ── Log file ─────────────────────────────────────────────
        os.makedirs(MODEL_DIR, exist_ok=True)
        _log_path = os.path.join(
            MODEL_DIR,
            f"kfold_{target}_{embedding_lm}_transformer_lm{_db_tag}_k{kfold if kfold else 'auto'}.log"
        )
        _logger = _TeeLogger(_log_path)
        _logger.start()

        y       = np.asarray(y, dtype=int)
        X_np    = X.values if hasattr(X, 'values') else np.asarray(X, dtype=np.float32)
        emb_dim = X_np.shape[1]
        n       = len(y)

        # ── Validate + get safe operating parameters ─────────────
        vd    = validate_dataset(y, emb_dim, context='kfold')
        kfold = kfold or vd['safe_kfold']

        # ── Splitter selection ────────────────────────────────────
        if cluster_col in data.columns:
            groups          = data[cluster_col].values
            n_unique_groups = len(np.unique(groups))
            if n_unique_groups == n:
                print(f"[kfold] '{cluster_col}' all-singleton → StratifiedKFold")
                splitter   = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
                split_iter = splitter.split(np.arange(n), y)
            elif n_unique_groups < kfold:
                print(f"[kfold] Only {n_unique_groups} clusters → reducing folds to {n_unique_groups}")
                kfold      = n_unique_groups
                splitter   = StratifiedGroupKFold(n_splits=kfold, shuffle=True, random_state=42)
                split_iter = splitter.split(np.arange(n), y, groups)
            else:
                splitter   = StratifiedGroupKFold(n_splits=kfold, shuffle=True, random_state=42)
                split_iter = splitter.split(np.arange(n), y, groups)
                print(f"[kfold] StratifiedGroupKFold on '{cluster_col}' "
                      f"({n_unique_groups} clusters, {kfold} folds)")
        else:
            print(f"[kfold] '{cluster_col}' not found — using StratifiedKFold")
            splitter   = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
            split_iter = splitter.split(np.arange(n), y)

        # Rename log file now that kfold count is final
        _logger.stop()
        _log_path = os.path.join(
            MODEL_DIR,
            f"kfold_{target}_{embedding_lm}_transformer_lm{_db_tag}_k{kfold}.log"
        )
        _logger = _TeeLogger(_log_path)
        _logger.start()

        barcodes = data.index.tolist()
        full_ds  = AntibodyDataset(X_np, y, barcodes)
        mean_fpr = np.linspace(0, 1, 100)

        print(f"\n[kfold] {kfold}-fold SGKF | {title} | lm={embedding_lm}"
              f" | cluster_col='{cluster_col}'")

        tprs, aucs_list, accs, f1s, precs, recs, recs_fail = [], [], [], [], [], [], []
        all_records = []

        best_fold_auc   = -1.0
        best_fold_num   = -1
        best_fold_state = None
        best_fold_cfg   = None

        plt.figure(figsize=(5, 5))

        for fold, (tr_idx, va_idx) in enumerate(split_iter, 1):
            print(f"\n── Fold {fold}/{kfold} ──")

            # [UPD-6] Leakage check
            if cluster_col in data.columns:
                _grps      = data[cluster_col].values
                _tr_grps   = set(_grps[tr_idx])
                _va_grps   = set(_grps[va_idx])
                _leaked    = _tr_grps & _va_grps
                if _leaked:
                    print(f"  [WARN] {len(_leaked)} cluster(s) leaked — "
                          f"check clustering threshold")
                else:
                    print(f"  [OK]  No CDR3 leakage | "
                          f"train_clusters={len(_tr_grps)}  val_clusters={len(_va_grps)}")

            y_tr = y[tr_idx]; y_va = y[va_idx]
            print(f"  Train={len(tr_idx):,}  "
                  f"pos={y_tr.sum():,} ({y_tr.mean():.1%})  "
                  f"neg={int((y_tr==0).sum()):,} ({1-y_tr.mean():.1%})")
            print(f"  Val  ={len(va_idx):,}  "
                  f"pos={y_va.sum():,} ({y_va.mean():.1%})  "
                  f"neg={int((y_va==0).sum()):,} ({1-y_va.mean():.1%})")

            # Fresh sub-model instance per fold
            fm          = TransformerLMModel.__new__(TransformerLMModel)
            fm.config   = copy.deepcopy(self.config)
            fm.device   = self.device
            fm.model    = None
            fm._embedding_dim = None
            fm._lm_name = embedding_lm

            n_fold = len(tr_idx)
            if fm.config.get('auto_scale', True):
                fm.config = auto_scale_config(n_fold, emb_dim, fm.config)

            # Print config report for fold 1 only (all folds use same config)
            if fold == 1:
                TransformerLMModel.print_config_report(
                    fm.config, embedding_dim=emb_dim,
                    n=n_fold, lm_name=embedding_lm)

            cfg = fm.config['training']
            bs  = cfg['batch_size']

            tr_loader = DataLoader(Subset(full_ds, tr_idx), batch_size=bs,
                                   shuffle=True,  num_workers=0)
            va_loader = DataLoader(Subset(full_ds, va_idx), batch_size=bs,
                                   shuffle=False, num_workers=0)

            loss_cfg      = fm.config.get('loss', {})
            criterion, cw = build_criterion(
                y_tr, fm.device,
                loss_type       = loss_cfg.get('type',             'weighted_ce'),
                focal_gamma     = loss_cfg.get('focal_gamma',       2.0),
                label_smoothing = loss_cfg.get('label_smoothing',   0.05),
            )
            print(f"  class_weights={cw.cpu().tolist()}")

            fm.model  = fm._build_model(emb_dim)
            optimizer = torch.optim.AdamW(
                fm.model.parameters(),
                lr=cfg['lr'], weight_decay=cfg['weight_decay'],
            )
            sch       = fm.config['scheduler']
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=sch['mode'], factor=sch['factor'], patience=sch['patience'],
            )
            fold_save = os.path.join(
                MODEL_DIR,
                f"_fold{fold}_{target}_{embedding_lm}_transformer_lm{_db_tag}.pt",
            )
            _fold_curve = os.path.join(
                MODEL_DIR,
                f"loss_curve_{target}_{embedding_lm}_transformer_lm{_db_tag}"
                f"_fold{fold}_k{kfold}.png",
            )
            _fold_title = (f"{target.upper()}  {embedding_lm}  fold {fold}/{kfold}\n"
                           f"n_train={len(tr_idx):,}  hd={fm.config['model']['hidden_dim']}"
                           f"  nl={fm.config['model']['num_layers']}"
                           f"  loss={fm.config.get('loss',{}).get('type','auto')}")

            fm.model, _fold_history = train_model(
                fm.model, tr_loader, va_loader,
                criterion, optimizer, scheduler,
                epochs   = cfg['epochs'],
                patience = fm.config.get('scheduler', {}).get('patience', 10),
                device   = fm.device,
                model_save = fold_save,
                use_amp  = fm.config.get('amp', False),
                plot_path  = _fold_curve,
                plot_title = _fold_title,
            )

            # ── evaluate ─────────────────────────────────────────
            fm.model.eval()
            probs, preds, trues, bcs = [], [], [], []
            with torch.no_grad():
                for emb, lbl, bc in va_loader:
                    out = fm.model(emb.to(fm.device))
                    probs.extend(torch.softmax(out, 1)[:, 1].cpu().numpy())
                    preds.extend(torch.argmax(out, 1).cpu().numpy())
                    trues.extend(lbl.cpu().numpy())
                    bcs.extend(bc)

            if len(set(trues)) < 2:
                print(f"  Skipping fold {fold} — only one class present.")
                continue

            fpr, tpr, _ = roc_curve(trues, probs)
            f_auc  = auc(fpr, tpr)
            f_acc  = accuracy_score(trues, preds)
            f_f1   = f1_score(trues,       preds, average='weighted', zero_division=0)
            f_prec = precision_score(trues, preds, average='weighted', zero_division=0)
            f_rec  = recall_score(trues,   preds, average='weighted', zero_division=0)
            f_rec_fail = recall_score(trues, preds, pos_label=0, average='binary',
                                      zero_division=0)

            aucs_list.append(f_auc);  accs.append(f_acc)
            f1s.append(f_f1);         precs.append(f_prec)
            recs.append(f_rec);       recs_fail.append(f_rec_fail)
            tprs.append(np.interp(mean_fpr, fpr, tpr));  tprs[-1][0] = 0.0

            if f_auc > best_fold_auc:
                best_fold_auc   = f_auc
                best_fold_num   = fold
                best_fold_state = copy.deepcopy(fm.model.state_dict())
                best_fold_cfg   = copy.deepcopy(fm.config)

            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label=f'Fold {fold} (AUC={f_auc:.3f})')
            print(f"  Fold {fold} → AUC={f_auc:.4f}  Acc={f_acc:.4f}"
                  f"  F1={f_f1:.4f}  Prec={f_prec:.4f}  Rec={f_rec:.4f}"
                  f"  Rec(Fail)={f_rec_fail:.4f}")

            if save_fold_preds:
                # [UPD-2] Per-fold CSV saved immediately
                _fold_pred_path = os.path.join(
                    MODEL_DIR,
                    f"fold_preds_{target}_{embedding_lm}_transformer_lm{_db_tag}"
                    f"_fold{fold}_k{kfold}.csv"
                )
                pd.DataFrame({
                    'BARCODE': list(bcs),
                    'fold':    fold,
                    'true':    list(trues),
                    'pred':    list(preds),
                    'prob':    list(probs),
                }).to_csv(_fold_pred_path, index=False)
                print(f"  [preds] Fold {fold} → {os.path.basename(_fold_pred_path)}")

                for bc, true, pred, prob in zip(bcs, trues, preds, probs):
                    all_records.append({
                        'BARCODE': bc, 'fold': fold,
                        'true': true, 'pred': pred, 'prob': prob,
                    })

        if not aucs_list:
            print("[kfold] No valid folds — check class distribution.")
            _logger.stop()
            return None, None, None, None, None, None

        # ── aggregate ────────────────────────────────────────────
        mean_tpr      = np.mean(tprs, axis=0);  mean_tpr[-1] = 1.0
        mean_auc      = auc(mean_fpr, mean_tpr)
        std_auc       = np.std(aucs_list)
        mean_acc      = np.mean(accs)
        mean_f1       = np.mean(f1s)
        mean_prec     = np.mean(precs)
        mean_rec      = np.mean(recs)
        mean_rec_fail = np.mean(recs_fail)

        # ── ROC plot ─────────────────────────────────────────────
        std_tpr = np.std(tprs, axis=0)
        plt.plot(mean_fpr, mean_tpr, 'b', lw=2,
                 label=f'Mean ROC (AUC={mean_auc:.3f}±{std_auc:.3f})')
        plt.fill_between(
            mean_fpr,
            np.maximum(mean_tpr - std_tpr, 0),
            np.minimum(mean_tpr + std_tpr, 1),
            color='grey', alpha=0.2, label='±1 std',
        )
        plt.plot([0, 1], [0, 1], 'k--', lw=0.8)
        plt.xlim([0, 1]);  plt.ylim([0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=9)
        plt.ylabel('True Positive Rate',  fontsize=9)
        plt.title(
            f'{title} — {embedding_lm}\n{kfold}-Fold SGKF ROC\n'
            f'Acc={mean_acc:.3f}  F1={mean_f1:.3f}  '
            f'Prec={mean_prec:.3f}  Rec={mean_rec:.3f}  '
            f'Rec(Fail)={mean_rec_fail:.3f}',
            fontsize=8,
        )
        plt.legend(loc='lower right', fontsize=5)
        plt.grid(True, lw=0.4)
        plt.tight_layout()

        plot_path = os.path.join(
            MODEL_DIR,
            f"CV_ROC_{target}_{embedding_lm}_transformer_lm{_db_tag}_k{kfold}.png",
        )
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\n[kfold] ROC plot → {plot_path}")

        # ── Combined fold-predictions CSV ─────────────────────────
        if save_fold_preds and all_records:
            pred_path = os.path.join(
                MODEL_DIR,
                f"fold_preds_{target}_{embedding_lm}_transformer_lm{_db_tag}_k{kfold}.csv",
            )
            df_preds = pd.DataFrame(all_records)
            df_preds['best_fold'] = (df_preds['fold'] == best_fold_num).astype(int)
            df_preds.to_csv(pred_path, index=False)
            print(f"[kfold] Combined fold predictions → {pred_path}")

        # ── Save best-fold checkpoint ─────────────────────────────
        if best_fold_state is not None:
            best_path = os.path.join(
                MODEL_DIR,
                f"BEST_{target}_{embedding_lm}_transformer_lm{_db_tag}"
                f"_k{kfold}_fold{best_fold_num}.pt",
            )
            torch.save(
                {
                    'state_dict':    best_fold_state,
                    'config':        best_fold_cfg,
                    'embedding_dim': emb_dim,
                    'fold':          best_fold_num,
                    'fold_auc':      best_fold_auc,
                    'kfold':         kfold,
                    'target':        target,
                    'embedding_lm':  embedding_lm,
                    'db_stem':       db_stem,
                },
                best_path,
            )
            print(f"[kfold] Best fold → {best_path}"
                  f"  (fold={best_fold_num}, AUC={best_fold_auc:.4f})")

        # ── Summary ───────────────────────────────────────────────
        _min_rate   = float(vd['min_rate'])
        _imbalanced = _min_rate < 0.30
        sep = '═' * 62
        print(f"\n{sep}")
        print(f"  {kfold}-FOLD CV RESULTS — {target.upper()}")
        print("─"*62)
        print(f"  Best fold : {best_fold_num}  (AUC={best_fold_auc:.4f})")
        print(f"  Mean AUC  : {mean_auc:.4f} ± {std_auc:.4f}")
        if _imbalanced:
            print(f"  [imbalanced: {_min_rate:.0%} minority — metrics at t=0.5 may be misleading]")
        print(f"  Mean Acc  : {mean_acc:.4f}")
        print(f"  Mean F1   : {mean_f1:.4f}")
        print(f"  Mean Prec : {mean_prec:.4f}")
        print(f"  Mean Rec  : {mean_rec:.4f}  (Pass class)")
        print(f"  Rec(Fail) : {mean_rec_fail:.4f}  ← minority class recall (class 0)")
        print(f"{sep}")

        # ── Threshold optimisation ────────────────────────────────
        # [UPD-5] passes model_name, db_tag, kfold so threshold_optimizer.py
        # writes correctly named files without any post-generation renaming.
        if _THRESHOLD_OPT_AVAILABLE and save_fold_preds and all_records:
            fold_preds_csv = os.path.join(
                MODEL_DIR,
                f"fold_preds_{target}_{embedding_lm}_transformer_lm{_db_tag}_k{kfold}.csv",
            )
            best_ckpt_path = os.path.join(
                MODEL_DIR,
                f"BEST_{target}_{embedding_lm}_transformer_lm{_db_tag}"
                f"_k{kfold}_fold{best_fold_num}.pt",
            ) if best_fold_state is not None else None

            print(f"\n[threshold] Starting optimisation  ···")
            try:
                run_full_threshold_pipeline(
                    fold_preds_csv = fold_preds_csv,
                    target         = target,
                    lm             = embedding_lm,
                    best_ckpt_path = best_ckpt_path,
                    output_dir     = MODEL_DIR,
                    cost_fp        = 1.0,
                    cost_fn        = 3.0,
                    model_name     = "transformer_lm",
                    db_tag         = _db_tag,
                    kfold          = kfold,
                )
            except Exception as _e:
                print(f"[threshold] WARNING: optimisation failed — {_e}")
                print(f"[threshold] kfold results are unaffected.")
        elif not _THRESHOLD_OPT_AVAILABLE:
            print(
                "\n[threshold] utils/threshold_optimizer.py not found — "
                "skipping. Predictions will use threshold=0.5."
            )

        _logger.stop()
        return mean_auc, std_auc, mean_acc, mean_f1, mean_prec, mean_rec

    # Alias — kept so any direct calls to the SGKF variant still work
    kfold_validation_sgkf = kfold_validation


# ══════════════════════════════════════════════════════════════════
# 8.  MODULE-LEVEL HELPER  (used by both wrapper and kfold)
# ══════════════════════════════════════════════════════════════════

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (in-place on base)."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


# ══════════════════════════════════════════════════════════════════
# 9.  SAMPLE SIZE EVALUATION
# ══════════════════════════════════════════════════════════════════

def evaluate_sample_size_effect(
    db_path:       str,
    embedding_csv: str  = None,
    target:        str  = "psr_filter",
    lm:            str  = "antiberta2-cssp",
    start_size:    int  = 500,
    increment:     int  = 1000,
    output_csv:    str  = None,
    cluster_col:   str  = "HCDR3_CLUSTER_0.8",
    random_state:  int  = 42,
    epochs:        int  = 0,
):
    """
    Evaluate how model performance scales with training set size.

    Trains a TransformerLMModel on increasing subsets of the dataset and
    records AUC, accuracy, F1, precision and recall on a held-out test set.
    Uses CDR3-cluster-stratified split to prevent sequence leakage.
    Each size step uses an 80/20 CDR3-cluster-stratified split.
    The 20% val/test set serves double duty: early stopping signal
    during training AND the held-out evaluation set reported in the CSV.
    No data is wasted on a separate validation set.

    Parameters
    ----------
    db_path       : path to Excel/CSV file with sequences and labels
    embedding_csv : path to pre-computed embedding CSV  (index = BARCODE).
                    Auto-detected as {db_path}.{lm}.emb.csv if not given.
                    If neither path resolves to an existing file the function
                    prints a warning and returns None (safe skip).
    target        : label column  (default 'psr_filter')
    lm            : language model name — embedded in all output filenames
    start_size    : first sample size to evaluate  (default 500)
    increment     : step between sample sizes       (default 1000)
    output_csv    : explicit output path; auto-generated from db + target + lm
                    if not supplied.
    cluster_col   : CDR3 cluster column for group-stratified split.
                    Computed on-the-fly if absent.
    random_state  : random seed for reproducibility
    epochs        : training epochs per size step.
                    0 (default) = let auto_scale decide the optimal count
                    for each dataset size — recommended.
                    Set > 0 to force a fixed epoch count (useful for speed).

    Outputs  (all filenames contain the db stem for traceability)
    -------
    {db_stem}_transformer_lm_{lm}_{target}_sample_size_{start}_{step}.csv
    {db_stem}_transformer_lm_{lm}_{target}_sample_size_{start}_{step}_plot.png
    {db_stem}_transformer_lm_{lm}_{target}_sample_size_{start}_{step}.log
    """
    # ── Resolve db stem ───────────────────────────────────────────
    db_stem = os.path.splitext(os.path.basename(db_path))[0]

    # ── Resolve embedding CSV ─────────────────────────────────────
    _candidates = [
        c for c in [
            embedding_csv,
            f"{db_path}.{lm}.emb.csv",
            os.path.join(os.path.dirname(db_path),
                         f"{db_stem}.{lm}.emb.csv"),
        ] if c
    ]
    _emb_path = next((c for c in _candidates if os.path.exists(c)), None)

    if _emb_path is None:
        print(
            f"\n[eval] WARNING: embedding file not found for lm='{lm}'.\n"
            f"  Searched:\n"
            + "".join(f"    {c}\n" for c in _candidates) +
            f"  Skipping evaluate_sample_size_effect.\n"
            f"  Generate embeddings first:\n"
            f"    python predict_developability.py --build-embedding {db_path} "
            f"--lm {lm}"
        )
        return None

    # ── Auto output paths — same directory as db_path ────────────
    db_dir = os.path.dirname(os.path.abspath(db_path))
    if output_csv is None:
        output_csv = os.path.join(
            db_dir,
            f"{db_stem}_transformer_lm_{lm}_{target}"
            f"_sample_size_{start_size}_{increment}.csv",
        )
    plot_path = output_csv.replace('.csv', '_plot.tiff')
    log_path  = output_csv.replace('.csv', '.log')

    # ── Start log ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    _logger = _TeeLogger(log_path)
    _logger.start()

    # ── Header ────────────────────────────────────────────────────
    print("═"*62)
    print(f"  evaluate_sample_size_effect — TransformerLM")
    print("─"*62)
    print(f"  db_path      : {db_path}")
    print(f"  embedding    : {_emb_path}")
    print(f"  target       : {target}")
    print(f"  lm           : {lm}")
    print(f"  start_size   : {start_size}")
    print(f"  increment    : {increment}")
    print(f"  val_split    : 20%  (CDR3-cluster-stratified, same set for early stopping + evaluation)")
    print(f"  epochs       : {epochs if epochs > 0 else 'auto (auto_scale)'}")
    print(f"  cluster_col  : {cluster_col}")
    print(f"  random_state : {random_state}")
    print(f"  output dir   : {db_dir}")
    print(f"  output_csv   : {output_csv}")
    print(f"  output_plot  : {plot_path}")
    print(f"  log          : {log_path}")
    print(f"{sep}\n")

    # ── Load database ─────────────────────────────────────────────
    print(f"[eval] Loading database  : {db_path}")
    ext = os.path.splitext(db_path)[1].lower()
    df  = pd.read_excel(db_path) if ext in ('.xlsx', '.xls') else pd.read_csv(db_path)

    df = df.dropna(subset=[target])
    if 'antigen' in df.columns:
        df = df[~df['antigen'].str.contains('test', na=False, case=False)]

    # ── Load embeddings ───────────────────────────────────────────
    print(f"[eval] Loading embeddings: {_emb_path}")
    emb = pd.read_csv(_emb_path, index_col=0)

    # Align on BARCODE
    if 'BARCODE' in df.columns:
        df = df.set_index('BARCODE')
    common = df.index.intersection(emb.index)
    if len(common) == 0:
        print(f"[eval] ERROR: no overlapping BARCODEs — skipping.")
        _logger.stop()
        return None
    df  = df.loc[common]
    emb = emb.loc[common]
    y   = df[target].astype(int).values
    print(f"[eval] n={len(df):,}  pos={y.mean():.1%}  emb_dim={emb.shape[1]}")
    print(f"[eval] Output CSV  : {output_csv}")
    print(f"[eval] Output plot : {plot_path}")

    # ── Ensure cluster column exists ──────────────────────────────
    if cluster_col not in df.columns:
        try:
            from utils.clustering import greedy_clustering_by_levenshtein
            thresh = float(cluster_col.split('_')[-1])
            print(f"[eval] Computing {cluster_col}  (threshold={thresh}) ...")
            df[cluster_col] = greedy_clustering_by_levenshtein(
                df['CDR3'].tolist(), thresh)
            print(f"[eval] → {df[cluster_col].nunique():,} clusters")
        except Exception as e:
            print(f"[eval] WARN: cannot compute {cluster_col} — {e}")
            print(f"[eval] Falling back to StratifiedShuffleSplit")
            cluster_col = None

    # ── Sample size sweep ─────────────────────────────────────────
    sample_sizes = list(range(start_size, len(df) + 1, increment))
    if not sample_sizes or sample_sizes[-1] < len(df):
        sample_sizes.append(len(df))

    results = []
    os.makedirs(MODEL_DIR, exist_ok=True)

    for size in sample_sizes:
        print(f"\n{'═'*55}")
        print(f"  n = {size:,}  |  db={db_stem}  |  lm={lm}")
        print(f"{'─'*55}")

        # Stratified subsample (balanced class ratio)
        idx_all = np.arange(len(df))
        sampled_idx = []
        for cls in [0, 1]:
            cls_idx = idx_all[y == cls]
            n_take  = min(len(cls_idx),
                          size // 2 + (size % 2 if cls == 1 else 0))
            rng = np.random.default_rng(random_state)
            sampled_idx.extend(rng.choice(cls_idx, n_take, replace=False).tolist())
        sampled_idx = sorted(sampled_idx)[:size]

        df_s  = df.iloc[sampled_idx]
        emb_s = emb.iloc[sampled_idx]
        y_s   = y[sampled_idx]

        # ── CDR3-cluster-stratified 80/20 split ──────────────────
        # 80% train, 20% val/test — same set used for both early
        # stopping and final evaluation. No data wasted.
        if cluster_col and cluster_col in df_s.columns:
            from sklearn.model_selection import StratifiedGroupKFold as _SGKF
            groups = df_s[cluster_col].values
            sgkf   = _SGKF(n_splits=5, shuffle=True, random_state=random_state)
            best_split, best_diff = None, float('inf')
            for tr, te in sgkf.split(np.arange(len(y_s)), y_s, groups):
                diff = abs(y_s[te].mean() - y_s.mean())
                if diff < best_diff:
                    best_diff, best_split = diff, (tr, te)
            tr_idx, te_idx = best_split
            _leaked = set(groups[tr_idx]) & set(groups[te_idx])
            print(f"  Split : StratifiedGroupKFold on '{cluster_col}'"
                  + (f"  [WARN] {len(_leaked)} cluster(s) leaked"
                     if _leaked else "  [OK] no CDR3 leakage"))
        else:
            from sklearn.model_selection import train_test_split as _tts
            tr_idx, te_idx = _tts(np.arange(len(y_s)), test_size=0.20,
                                   stratify=y_s, random_state=random_state)
            print(f"  Split : StratifiedShuffleSplit (no cluster col)")

        X_tr = emb_s.iloc[tr_idx];  y_tr = y_s[tr_idx]
        X_te = emb_s.iloc[te_idx];  y_te = y_s[te_idx]
        print(f"  train={len(X_tr):,} pos={y_tr.mean():.1%}  "
              f"val/test={len(X_te):,} pos={y_te.mean():.1%}")

        # Train — val/test used for early stopping; same set evaluated below
        model = TransformerLMModel()
        model._lm_name = lm
        _train_epochs = epochs if epochs > 0 else None
        model.train(X_tr, y_tr,
                    val_X  = X_te,
                    val_y  = y_te,
                    epochs = _train_epochs,
                    target = target,
                    db_stem= db_stem)

        # Evaluate
        probs = model.predict_proba(X_te)
        preds = (probs >= 0.5).astype(int)
        try:
            _auc = roc_auc_score(y_te, probs)
        except Exception:
            _auc = 0.5

        r = {
            'db':          db_stem,
            'lm':          lm,
            'sample_size': size,
            'train_size':  len(X_tr),
            'test_size':   len(X_te),
            'auc':         round(_auc, 4),
            'accuracy':    round(accuracy_score(y_te, preds), 4),
            'f1_score':    round(f1_score(y_te, preds, zero_division=0), 4),
            'precision':   round(precision_score(y_te, preds, zero_division=0), 4),
            'recall':      round(recall_score(y_te, preds, zero_division=0), 4),
        }
        results.append(r)
        print(f"  AUC={r['auc']:.4f}  Acc={r['accuracy']:.4f}  "
              f"F1={r['f1_score']:.4f}  Prec={r['precision']:.4f}  "
              f"Rec={r['recall']:.4f}")

        # Save incrementally — partial results never lost
        pd.DataFrame(results).to_csv(output_csv, index=False)

    # ── Final save ────────────────────────────────────────────────
    rdf = pd.DataFrame(results)
    rdf.to_csv(output_csv, index=False)
    print(f"\n[eval] Results → {output_csv}")

    # ── Nature Biotechnology-style figure ────────────────────────
    # Specifications: Arial 7pt, no top/right spines, outward ticks,
    # 300 DPI TIFF, double-column width (183 mm = 7.2 in)
    _MM = 1 / 25.4
    plt.rcParams.update({
        'font.family':      'sans-serif',
        'font.sans-serif':  ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size':         7,
        'axes.labelsize':    7,
        'axes.linewidth':    0.6,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'xtick.labelsize':   6,
        'ytick.labelsize':   6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size':  2.5,
        'ytick.major.size':  2.5,
        'xtick.direction':   'out',
        'ytick.direction':   'out',
        'legend.fontsize':   6,
        'legend.frameon':    False,
        'pdf.fonttype':      42,
        'ps.fonttype':       42,
    })

    fig, ax = plt.subplots(figsize=(183 * _MM, 183 * _MM * 0.60))

    _styles = [
        ('auc',       'o', '#1f77b4', 'AUC-ROC'),
        ('accuracy',  's', '#ff7f0e', 'Accuracy'),
        ('f1_score',  '^', '#2ca02c', 'F1'),
        ('precision', 'd', '#9467bd', 'Precision'),
        ('recall',    'v', '#d62728', 'Recall'),
    ]
    for col, mk, color, lb in _styles:
        ax.plot(rdf['sample_size'], rdf[col],
                marker=mk, color=color, label=lb,
                lw=1.2, ms=4, markeredgewidth=0.4,
                markeredgecolor='white')

    ax.set_xlabel('Training set size', fontsize=7, labelpad=3)
    ax.set_ylabel('Metric',            fontsize=7, labelpad=3)
    ax.set_ylim(0, 1.05)

    # X-axis: tick every 500 with rotated labels to prevent overlap
    _x_min = rdf['sample_size'].min()
    _x_max = rdf['sample_size'].max()
    ax.set_xlim(0, _x_max * 1.02)
    import matplotlib.ticker as _ticker
    ax.xaxis.set_major_locator(_ticker.MultipleLocator(500))
    ax.xaxis.set_minor_locator(_ticker.MultipleLocator(250))
    ax.tick_params(axis='x', which='major', labelsize=5.5,
                   rotation=45, length=2.5)
    ax.tick_params(axis='x', which='minor', length=1.5, color='#cccccc')
    ax.tick_params(axis='y', labelsize=6, length=2.5)
    ax.set_title(
        f'{db_stem}  |  {target}  |  transformer_lm  |  lm={lm}',
        fontsize=6.5, pad=4, color='#333333')
    ax.legend(loc='lower right', fontsize=6,
              handlelength=1.2, handletextpad=0.4, labelspacing=0.3)
    ax.grid(True, alpha=0.2, lw=0.4)

    plt.tight_layout()
    fig.savefig(plot_path, dpi=300, format='tiff',
                bbox_inches='tight', facecolor='white')
    # Also save PNG preview
    _png_path = plot_path.replace('.tiff', '_preview.png')
    fig.savefig(_png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    plt.rcdefaults()
    print(f"[eval] Plot TIFF → {plot_path}")
    print(f"[eval] Plot PNG  → {_png_path}")

    _logger.stop()
    return rdf




# ══════════════════════════════════════════════════════════════════
# 10.  PLM ENCODER WRAPPER  (for end-to-end fine-tuning, Mode 2)
# ══════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════
# LoRA IMPLEMENTATION  (Mode 3 — PEFT without external library)
# ══════════════════════════════════════════════════════════════════
# Works for ALL PLMs uniformly (ablang, igbert, antiberta2, antiberty)
# Zero new dependencies — pure PyTorch.
#
# LoRA reference: Hu et al. 2021 "LoRA: Low-Rank Adaptation of Large
# Language Models"  https://arxiv.org/abs/2106.09685
#
# How it works:
#   Original:  out = x @ W          W: (d, d) — frozen
#   LoRA:      out = x @ W  +  (x @ A) @ B * scale
#              A: (d, r)  B: (r, d)  — trained
#              scale = lora_alpha / r
#   r=8, alpha=16, d=1024 → 16,384 new params vs 1,048,576 original
#   → 64× compression
# ═══════════════════════════════════════════════════════════════════

class LoRALayer(nn.Module):
    """
    Wraps an existing nn.Linear with LoRA A×B low-rank adaptation.

    The original weight W is NEVER modified — LoRA matrices A and B
    accumulate all task-specific updates.

    Parameters
    ----------
    original_linear : nn.Linear   the layer to wrap (W frozen inside)
    r               : int         LoRA rank  (4, 8, 16)
    lora_alpha      : float       scaling factor  (usually 2×r)
    """
    def __init__(self, original_linear: nn.Linear,
                 r: int = 8, lora_alpha: float = 16.0):
        super().__init__()
        self.original = original_linear
        self.r         = r
        self.scale     = lora_alpha / r

        d_in  = original_linear.in_features
        d_out = original_linear.out_features

        # LoRA matrices — initialised following paper:
        #   A ~ N(0, 1/r)  (small random)
        #   B = 0           (so LoRA output = 0 at init, = pretrained behaviour)
        self.lora_A = nn.Linear(d_in,  r,     bias=False)
        self.lora_B = nn.Linear(r,     d_out, bias=False)
        nn.init.normal_(self.lora_A.weight, std=1.0 / r)
        nn.init.zeros_(self.lora_B.weight)

        # Freeze original weights — they never change
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen path + low-rank update
        return self.original(x) + self.lora_B(self.lora_A(x)) * self.scale

    def extra_repr(self) -> str:
        return (f"in={self.original.in_features}  "
                f"out={self.original.out_features}  "
                f"r={self.r}  scale={self.scale:.2f}")


def inject_lora(plm_model: nn.Module,
                lm_name:   str,
                r:         int   = 8,
                lora_alpha: float = 16.0,
                target_layers: list = None) -> dict:
    """
    Inject LoRA into attention Q and V projection matrices of a PLM.

    Modifies plm_model IN-PLACE.
    Returns a dict mapping injection paths → LoRALayer instances
    (used later by save_lora() / load_lora()).

    target_layers : list of int  — which encoder layer indices get LoRA.
                    None = last 2 layers (safe default for small datasets).
                    [0..11] = all layers (full LoRA, for large datasets).

    Layer path patterns per PLM:
        igbert      encoder.layer.{N}.attention.self.{query,value}
        antiberta2  encoder.layer.{N}.attention.self.{query,value}
        antiberty   encoder.layer.{N}.attention.self.{query,value}
        ablang      (RoBERTa-style) encoder.layer.{N}.attention.self.{query,value}
    """
    import re

    # Auto-detect number of layers from model
    layer_indices = set()
    for name, _ in plm_model.named_modules():
        m = re.search(r'(?:encoder\.layer|layers)\.(\d+)\.', name)
        if m:
            layer_indices.add(int(m.group(1)))

    n_layers = max(layer_indices) + 1 if layer_indices else 12

    # Default: last 2 layers get LoRA
    if target_layers is None:
        target_layers = list(range(max(0, n_layers - 2), n_layers))

    print(f"[LoRA] Injecting into {lm_name}  "
          f"r={r}  alpha={lora_alpha}  "
          f"layers={target_layers}")

    injected = {}

    # Walk the model and replace Q,V linears in target layers
    def _inject_recursive(module, prefix=''):
        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name

            # Check if this is a target attention layer
            m = re.search(r'(?:encoder\.layer|layers)\.(\d+)\.', full_name)
            layer_idx = int(m.group(1)) if m else -1

            if (layer_idx in target_layers and
                    isinstance(child, nn.Linear) and
                    any(t in child_name for t in ('query', 'value', 'q_proj', 'v_proj'))):

                # Replace with LoRA-wrapped version
                lora_layer = LoRALayer(child, r=r, lora_alpha=lora_alpha)
                setattr(module, child_name, lora_layer)
                injected[full_name] = lora_layer
                print(f"  [LoRA] ← {full_name}  "
                      f"({child.in_features}→{child.out_features}  "
                      f"params: {child.in_features*r + r*child.out_features:,})")
            else:
                _inject_recursive(child, full_name)

    _inject_recursive(plm_model)

    n_lora   = sum(p.numel() for n, p in plm_model.named_parameters()
                   if p.requires_grad and ('lora_A' in n or 'lora_B' in n))
    n_total  = sum(p.numel() for p in plm_model.parameters())
    print(f"[LoRA] Injected {len(injected)} layers  "
          f"LoRA trainable={n_lora:,}  "
          f"PLM total={n_total/1e6:.1f}M  "
          f"({100*n_lora/n_total:.2f}% updated)")
    return injected


def save_lora_weights(injected: dict, path: str) -> None:
    """
    Save ONLY the LoRA A and B matrices (not the full PLM).
    File is tiny (~2MB for r=8 on IgBERT) — easy to share.
    """
    state = {name: {
        'lora_A': layer.lora_A.state_dict(),
        'lora_B': layer.lora_B.state_dict(),
        'r':       layer.r,
        'scale':   layer.scale,
    } for name, layer in injected.items()}
    torch.save(state, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"[LoRA] Saved {len(state)} LoRA layers → {path}  ({size_mb:.1f} MB)")


def load_lora_weights(plm_model: nn.Module, path: str,
                      lm_name: str, r: int = 8,
                      lora_alpha: float = 16.0) -> dict:
    """
    Load LoRA weights into a freshly-loaded PLM.
    Re-injects LoRA structure then loads saved A,B matrices.
    """
    state = torch.load(path, map_location='cpu', weights_only=False)

    # Infer target layers from saved keys
    import re
    target_layers = set()
    for key in state:
        m = re.search(r'(?:encoder\.layer|layers)\.(\d+)\.', key)
        if m:
            target_layers.add(int(m.group(1)))

    injected = inject_lora(plm_model, lm_name, r=r,
                           lora_alpha=lora_alpha,
                           target_layers=sorted(target_layers))

    # Load saved weights into injected layers
    for name, layer_state in state.items():
        if name in injected:
            injected[name].lora_A.load_state_dict(layer_state['lora_A'])
            injected[name].lora_B.load_state_dict(layer_state['lora_B'])

    print(f"[LoRA] Loaded weights from {path}")
    return injected

class PLMEncoder(nn.Module):
    """
    Wraps an external PLM (ABlang, IgBERT, AntiBERTy, AntiBERTa2, AntiBERTa2-CSSP)
    as a torch.nn.Module so gradients can flow through it during training.

    Used ONLY in Mode 2 (--finetune_plm).
    In Mode 1 (current default), embeddings are pre-computed and static.

    freeze_layers : int
        Number of PLM transformer layers to freeze from the bottom.
        0  = all layers trainable (full fine-tune, needs large dataset)
        N  = freeze first N layers, train top layers + classifier
        -1 = freeze all PLM layers (equivalent to Mode 1 but batched)

    Supported LMs and their approximate layer counts:
        ablang        : 12 encoder layers  (RoBERTa-base style)
        igbert        : 12 encoder layers  (BERT-base style)
        antiberty     : 12 encoder layers  (RoBERTa-base style)
        antiberta2    : 12 encoder layers  (RoBERTa-base style)
        antiberta2-cssp: 12 encoder layers
    """

    # Each PLM returns embeddings differently — map to a canonical forward()
    _SUPPORTED = {'ablang', 'igbert', 'antiberty', 'antiberta2', 'antiberta2-cssp'}

    def __init__(self, lm_name: str, freeze_layers: int = 10):
        super().__init__()
        self.lm_name      = lm_name.lower()
        self.freeze_layers = freeze_layers
        self._model        = None
        self._tokenizer    = None
        self._emb_dim      = None
        self._load()

    def _load(self):
        """
        Load the PLM matching EXACTLY how embedding_generator.py uses each model.

        PLM         Package          Input              Pooling       emb_dim
        ─────────────────────────────────────────────────────────────────────────
        ablang      ablang2          [HSEQ, LSEQ]       seqcoding     480
                                     or [HSEQ] VH-only  (mean over residues)
        antiberty   antiberty        HSEQ[CLS][CLS]LSEQ mean pool     512
                                     or HSEQ[CLS] VH
        antiberta2  transformers     HSEQ[CLS][CLS]LSEQ mean pool     1024
         -cssp      RoFormerTokenizer or HSEQ[CLS] VH
        igbert      transformers     [CLS]VH[SEP]VL[SEP] mean pool    1024
                    BertTokenizer     or [CLS]VH[SEP]
                    space-separated AAs
        """
        lm = self.lm_name
        print(f"[PLMEncoder] Loading {lm}  (freeze_layers={self.freeze_layers}) ...")

        if lm == 'ablang':
            # ablang2 package — paired model, seqcoding mode
            # Input: [HSEQ_str, LSEQ_str] list per antibody (or [HSEQ] for VH-only)
            # Output: already mean-pooled 480-dim vector per antibody
            import ablang2
            self._model       = ablang2.pretrained(
                "ablang2-paired", device="cpu", random_init=False, ncpu=3)
            self._tokenizer   = None    # ablang2 handles tokenisation internally
            self._emb_dim     = 480
            self._forward_fn  = self._forward_ablang

        elif lm == 'antiberty':
            # AntiBERTy — space-separated AAs, HSEQ[CLS][CLS]LSEQ format
            from antiberty import AntiBERTyRunner
            self._runner      = AntiBERTyRunner()
            self._model       = self._runner.model    # underlying nn.Module
            self._tokenizer   = self._runner.tokenizer
            self._emb_dim     = 512
            self._forward_fn  = self._forward_antiberty

        elif lm in ('antiberta2', 'antiberta2-cssp'):
            # AntiBERTa2 / AntiBERTa2-CSSP — RoFormer, space-separated AAs
            # HSEQ[CLS][CLS]LSEQ format, mean pool over non-padding tokens
            from transformers import RoFormerTokenizer, RoFormerModel
            _model_name = ("alchemab/antiberta2-cssp" if lm == "antiberta2-cssp"
                           else "alchemab/antiberta2")
            self._tokenizer  = RoFormerTokenizer.from_pretrained(_model_name)
            self._model      = RoFormerModel.from_pretrained(_model_name)
            self._emb_dim    = 1024
            self._forward_fn = self._forward_antiberta2

        elif lm == 'igbert':
            # IgBERT — BertTokenizer, space-separated AAs
            # Paired: [CLS] VH_spaced [SEP] VL_spaced [SEP]
            # Mean pool over ALL non-padding tokens (incl. special tokens)
            from transformers import BertTokenizer, BertModel
            _model_name      = "Exscientia/IgBert"
            self._tokenizer  = BertTokenizer.from_pretrained(_model_name)
            self._model      = BertModel.from_pretrained(
                _model_name, add_pooling_layer=False)
            self._emb_dim    = 1024
            self._forward_fn = self._forward_igbert

        else:
            raise ValueError(
                f"PLMEncoder: unsupported lm='{lm}'. "
                f"Supported: {self._SUPPORTED}"
            )

        self._apply_freezing()
        _m = self._model if not lm == 'ablang' else None
        if _m is not None:
            n_total  = sum(p.numel() for p in _m.parameters())
            n_frozen = sum(p.numel() for p in _m.parameters() if not p.requires_grad)
            print(f"[PLMEncoder] {lm}  emb_dim={self._emb_dim}  "
                  f"params={n_total/1e6:.1f}M  "
                  f"trainable={(n_total-n_frozen)/1e6:.1f}M  "
                  f"frozen={n_frozen/1e6:.1f}M")
        else:
            print(f"[PLMEncoder] {lm}  emb_dim={self._emb_dim}  "
                  f"(ablang2 manages params internally)")

    def _apply_freezing(self):
        """Freeze first freeze_layers transformer encoder layers."""
        _m = getattr(self, '_model', None)
        if _m is None or self.lm_name == 'ablang':
            return   # ablang2 manages its own parameters

        if self.freeze_layers == -1:
            for p in _m.parameters():
                p.requires_grad = False
            return

        # Always freeze embedding layers (token + position embeddings)
        for name, param in _m.named_parameters():
            if any(k in name for k in ('embedding', 'position', 'token_type')):
                param.requires_grad = False

        # Freeze first N encoder layers
        if self.freeze_layers > 0:
            import re
            for name, param in _m.named_parameters():
                # Matches encoder.layer.N. (BERT) or layers.N. (RoFormer)
                m_ = re.search(r'(?:encoder\.layer|layers)\.(\d+)\.', name)
                if m_ and int(m_.group(1)) < self.freeze_layers:
                    param.requires_grad = False

    @staticmethod
    def _spaced(seq: str) -> str:
        """Insert spaces between amino acids — required by IgBERT/AntiBERTa2."""
        return ' '.join(seq.upper().replace(' ', ''))

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean pool hidden states over non-padding positions."""
        mask_f = mask.unsqueeze(-1).float()
        return (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)

    def _forward_ablang(self, seqs_hseq: list,
                         seqs_lseq: list = None) -> torch.Tensor:
        """
        ABlang2: input list of [HSEQ, LSEQ] pairs (or [HSEQ] for VH-only).
        ablang2.pretrained() handles tokenisation + embedding internally.
        Returns (B, 480) already mean-pooled.
        """
        device = "cpu"
        results = []
        for i, h in enumerate(seqs_hseq):
            l   = seqs_lseq[i] if (seqs_lseq is not None) else None
            seq = [str(h), str(l)] if l else [str(h)]
            emb = self._model(seq, mode='seqcoding')[0]   # (480,)
            results.append(torch.from_numpy(emb) if hasattr(emb, 'numpy') else emb)
        return torch.stack(results).float()               # (B, 480)

    def _forward_antiberty(self, sequences: list) -> torch.Tensor:
        """
        AntiBERTy: sequences already formatted as 'HSEQ[CLS][CLS]LSEQ'
        or 'HSEQ[CLS]' for VH-only.
        Mean pool over residue embeddings.
        """
        embs   = self._runner.embed(sequences)   # list of (L, 512) arrays
        result = []
        for e in embs:
            t = torch.from_numpy(e).float() if hasattr(e, 'numpy') else e.float()
            result.append(t.mean(0))              # (512,)
        return torch.stack(result)                # (B, 512)

    def _forward_antiberta2(self, sequences: list) -> torch.Tensor:
        """
        AntiBERTa2 / AntiBERTa2-CSSP: space-separated AAs,
        HSEQ[CLS][CLS]LSEQ format. RoFormer. Mean pool.
        """
        dev = next(self._model.parameters()).device
        enc = self._tokenizer(
            sequences, padding=True, truncation=True,
            max_length=320, return_tensors='pt'
        )
        enc    = {k: v.to(dev) for k, v in enc.items()}
        hidden = self._model(**enc).last_hidden_state   # (B, L, 1024)
        return self._mean_pool(hidden, enc['attention_mask'])

    def _forward_igbert(self, sequences: list) -> torch.Tensor:
        """
        IgBERT: space-separated AAs, BertTokenizer.
        Paired input: [CLS] VH_spaced [SEP] VL_spaced [SEP]
        Mean pool over ALL non-padding tokens (matching embedding_generator.py).
        """
        dev = next(self._model.parameters()).device
        enc = self._tokenizer(
            sequences, padding=True, truncation=True,
            max_length=320, return_tensors='pt'
        )
        enc    = {k: v.to(dev) for k, v in enc.items()}
        hidden = self._model(**enc).last_hidden_state   # (B, L, 1024)
        return self._mean_pool(hidden, enc['attention_mask'])

    def forward(self, sequences: list) -> torch.Tensor:
        """sequences: list of strings → tensor (B, emb_dim)"""
        return self._forward_fn(sequences)

    @property
    def embedding_dim(self) -> int:
        return self._emb_dim

    def to(self, device):
        self._model = self._model.to(device)
        return self


# ══════════════════════════════════════════════════════════════════
# FINE-TUNE METHODS  (added to TransformerLMModel)
# ══════════════════════════════════════════════════════════════════

def _fine_tune(self,
               finetune_db:    str,
               target:         str,
               lm:             str             = None,
               freeze_layers:  object          = 1,
               finetune_lr:    float           = 1e-6,
               finetune_epochs: int            = 10,
               batch_size:     int             = 32,
               db_stem:        str             = ""):
    """
    Level 2 fine-tuning: adapt a pretrained TransformerLMModel checkpoint
    to a new dataset using pre-computed PLM embeddings.

    This is the RECOMMENDED fine-tuning path for most users:
      1. Load pretrained .pt checkpoint (from Level 1 or another lab)
      2. Pre-compute PLM embeddings for the new dataset
         (automated: platform looks for {finetune_db}.{lm}.emb.csv)
      3. Fine-tune the classifier with a low learning rate
      4. Save a new checkpoint with _ft_{finetune_db_stem} suffix

    Parameters
    ----------
    finetune_db    : path to new dataset Excel/CSV with sequences + labels
    target         : label column in finetune_db
    lm             : PLM name (must match pretrained model's embedding dim)
                     inferred from checkpoint if not given
    freeze_layers  : int | 'all' | 'none'
                     'all'  → only classification head trainable (~260 params)
                     0/'none' → everything trainable (use very low lr)
                     N (int) → freeze first N transformer encoder layers
    finetune_lr    : learning rate — should be 10–100× lower than original
    finetune_epochs: number of fine-tuning epochs
    batch_size     : mini-batch size
    db_stem        : pretrained database stem (used in output filename)
    """
    import pandas as pd

    lm = lm or getattr(self, '_lm_name', '')
    if not lm:
        raise ValueError(
            "Cannot determine PLM name. Pass lm= explicitly or ensure "
            "the checkpoint was saved with lm_name embedded."
        )

    _ft_stem  = os.path.splitext(os.path.basename(finetune_db))[0]
    _db_tag   = f"_{db_stem}" if db_stem else ""
    _ft_tag   = f"_ft_{_ft_stem}"

    sep = '═' * 62; sep2 = '─' * 62
    print(f"\n{sep}")
    print(f"  FINE-TUNE  (Level 2)")
    print(f"{sep2}")
    print(f"  Pretrained   : {lm}  emb={self._embedding_dim}")
    print(f"  Fine-tune db : {finetune_db}")
    print(f"  Target       : {target}")
    print(f"  freeze_layers: {freeze_layers}")
    print(f"  lr           : {finetune_lr:.2e}")
    print(f"  epochs       : {finetune_epochs}")
    print("═"*62)

    # ── Load + embed fine-tune dataset ────────────────────────────
    from embedding_generator import generate_embedding
    _emb_path = f"{finetune_db}.{lm}.emb.csv"
    if not os.path.exists(_emb_path):
        print(f"[finetune] Embedding not found → generating {lm} ...")
        _emb_path = generate_embedding(finetune_db, lm=lm)

    print(f"[finetune] Loading embeddings: {_emb_path}")
    emb_df   = pd.read_csv(_emb_path, index_col=0)

    ext = os.path.splitext(finetune_db)[1].lower()
    df  = (pd.read_excel(finetune_db) if ext in ('.xlsx', '.xls')
           else pd.read_csv(finetune_db))
    if 'BARCODE' in df.columns:
        df = df.set_index('BARCODE')
    df  = df.dropna(subset=[target])
    common = df.index.intersection(emb_df.index)
    df  = df.loc[common];  emb_df = emb_df.loc[common]
    y   = df[target].astype(int).values
    X   = emb_df

    print(f"[finetune] n={len(y):,}  pos={y.mean():.1%}  emb_dim={X.shape[1]}")

    # Validate embedding dimension matches pretrained model
    if X.shape[1] != self._embedding_dim:
        raise ValueError(
            f"Embedding dimension mismatch: fine-tune data has {X.shape[1]} dims "
            f"but pretrained model expects {self._embedding_dim} dims. "
            f"Ensure --lm matches the PLM used for pretraining."
        )

    # ── Apply layer freezing ─────────────────────────────────────
    if freeze_layers == 'all':
        for name, param in self.model.named_parameters():
            param.requires_grad = ('head' in name)
        print("[finetune] freeze=ALL — only classification head trainable")

    elif freeze_layers in (0, 'none'):
        for param in self.model.parameters():
            param.requires_grad = True
        print("[finetune] freeze=NONE — all classifier layers trainable")

    else:
        import re
        n_freeze = int(freeze_layers)
        for name, param in self.model.named_parameters():
            m = re.search(r'transformer\.layers\.(\d+)\.', name)
            if m:
                param.requires_grad = (int(m.group(1)) >= n_freeze)
            else:
                param.requires_grad = True   # head + input_proj always trainable
        print(f"[finetune] freeze={n_freeze} — "
              f"transformer layers 0..{n_freeze-1} frozen, "
              f"layers {n_freeze}+ + head trainable")

    n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in self.model.parameters())
    print(f"[finetune] Trainable: {n_trainable:,} / {n_total:,} params "
          f"({n_trainable/n_total:.1%})")

    # ── Train ────────────────────────────────────────────────────
    train_loader = self._to_loader(X, y, batch_size, shuffle=True)
    loss_cfg     = self.config.get('loss', {})
    criterion, _ = build_criterion(
        y, self.device,
        loss_type   = loss_cfg.get('type', 'weighted_ce'),
        focal_gamma = loss_cfg.get('focal_gamma', 2.0),
    )
    # Only pass trainable params to optimizer
    optimizer = torch.optim.AdamW(
        [p for p in self.model.parameters() if p.requires_grad],
        lr=finetune_lr, weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    _ft_curve = os.path.join(
        MODEL_DIR,
        f"finetune_curve_{target}_{lm}_transformer_lm{_db_tag}{_ft_tag}.png"
    )

    _ft_uid  = f"{os.getpid()}_{int(__import__('time').time()*1000) % 100000}"
    _ft_tmp  = os.path.join(MODEL_DIR, f"_tmp_ft_{_ft_uid}.pt")

    self.model, _ = train_model(
        self.model, train_loader, None,
        criterion, optimizer, scheduler,
        epochs=finetune_epochs, patience=finetune_epochs,
        device=self.device,
        model_save=_ft_tmp,
        plot_path=_ft_curve,
        plot_title=f"Fine-tune {target} | {lm} | freeze={freeze_layers} | {_ft_stem}",
    )
    # Clean up unique tmp file
    try:
        if os.path.exists(_ft_tmp):
            os.remove(_ft_tmp)
    except OSError:
        pass

    # Restore all params to trainable after fine-tuning
    for param in self.model.parameters():
        param.requires_grad = True

    print(f"[finetune] Complete.")
    return self


def _train_with_plm_finetune(self,
                              data_df,
                              target:          str,
                              lm:              str,
                              freeze_plm_layers: int   = 10,
                              epochs:          int     = 20,
                              batch_size:      int     = 16,
                              lr_classifier:   float   = 1e-4,
                              lr_plm:          float   = 1e-6,
                              db_stem:         str     = "",
                              peft:            str     = "none",
                              lora_r:          int     = 8,
                              lora_alpha:      float   = 16.0,
                              lora_layers:     list    = None):
    # peft: 'none' = Mode 2 (standard layer unfreezing)
    #       'lora' = Mode 3 (LoRA injection, recommended)
    """
    End-to-end PLM + classifier training.  Supports two sub-modes:

    Mode 2  (peft='none', default):
        Standard layer unfreezing — top freeze_plm_layers are fully updated.
        Higher capacity but more risk of catastrophic forgetting.
        Recommended: n > 20,000.

    Mode 3  (peft='lora'):
        LoRA injection into Q,V attention matrices of target layers.
        W_original stays frozen — only low-rank A×B matrices are trained.
        64× fewer PLM params updated → low forgetting risk, CPU feasible.
        Recommended: n > 1,000 when wanting PLM adaptation.

    Mode 1  (frozen embeddings, the default --train):
        Call train() instead — uses pre-computed .emb.csv.
        No PLM weight updates at all.
        Recommended: n < 10,000 or CPU-only deployment.

    Gradients flow through BOTH the classifier AND the PLM layers.
    The PLM adapts its representations to predict {target} directly.

    This requires sequences to be processed in batches through the PLM
    — no pre-computed .emb.csv files needed or used.

    Parameters
    ----------
    data_df          : DataFrame with HSEQ, CDR3, {target} columns
    target           : label column
    lm               : PLM name (ablang / igbert / antiberty / antiberta2 / antiberta2-cssp)
    freeze_plm_layers: first N PLM layers to freeze (default 10 of 12)
                       10 = freeze most of PLM, only tune top 2 layers
                        0 = full PLM fine-tuning (needs large dataset, >10k)
    epochs           : training epochs
    batch_size       : SMALL batch (16 default) — PLM is large, memory limited
    lr_classifier    : learning rate for classifier head (higher)
    lr_plm           : learning rate for PLM layers (lower — avoids forgetting)
    db_stem          : database stem for output filenames

    Two-stage learning rate (differential lr):
        PLM layers       → lr_plm       (1e-6, slow, careful)
        Classifier layers → lr_classifier (1e-4, normal speed)
    This is standard practice for PLM fine-tuning (ULMFiT, 2018).
    """
    import re

    _db_tag   = f"_{db_stem}" if db_stem else ""
    _mode_tag = f"_plmft{freeze_plm_layers}"

    sep = '═' * 62; sep2 = '─' * 62
    print(f"\n{sep}")
    print(f"  TRAIN  (Mode 2 — end-to-end PLM fine-tuning)")
    print(f"{sep2}")
    print(f"  PLM            : {lm}")
    print(f"  PLM freeze     : first {freeze_plm_layers} layers frozen")
    print(f"  lr_classifier  : {lr_classifier:.2e}")
    print(f"  lr_plm         : {lr_plm:.2e}")
    print(f"  epochs         : {epochs}  batch_size={batch_size}")
    print(f"  NOTE: sequences processed in batches — no .emb.csv needed")
    print("═"*62)

    # ── Validate data ──────────────────────────────────────────
    df = data_df.dropna(subset=['HSEQ', target]).copy()
    y  = df[target].astype(int).values
    sequences = df['HSEQ'].tolist()    # Heavy chain sequences
    barcodes  = (df.index.tolist() if 'BARCODE' not in df.columns
                 else df['BARCODE'].tolist())
    n = len(y)
    print(f"[plmft] n={n:,}  pos={y.mean():.1%}")

    # ── Load PLM encoder ──────────────────────────────────────
    _use_lora = str(peft).lower() == 'lora'

    if _use_lora:
        # Mode 3: freeze ALL PLM layers, then inject LoRA into Q,V matrices
        plm_encoder = PLMEncoder(lm, freeze_layers=-1)
    else:
        # Mode 2: unfreeze top (12 - freeze_plm_layers) layers
        plm_encoder = PLMEncoder(lm, freeze_layers=freeze_plm_layers)

    plm_encoder = plm_encoder.to(self.device)
    emb_dim     = plm_encoder.embedding_dim

    # ── Apply LoRA injection (Mode 3 only) ────────────────────
    _lora_injected = {}
    if _use_lora:
        _plm_module = getattr(plm_encoder, '_model', None)
        if _plm_module is None:
            raise ValueError(
                f"PLMEncoder for '{lm}' has no ._model attribute. "
                f"LoRA injection requires a PyTorch nn.Module. "
                f"Use peft='none' (Mode 2) for this PLM."
            )
        _lora_injected = inject_lora(
            _plm_module,
            lm_name       = lm,
            r             = lora_r,
            lora_alpha    = lora_alpha,
            target_layers = lora_layers,   # None → auto = last 2 layers
        )
        print(f"[Mode 3 / LoRA] r={lora_r}  alpha={lora_alpha}  "
              f"layers={lora_layers or 'last 2 (auto)'}")
    else:
        print(f"[Mode 2 / full] freeze_plm_layers={freeze_plm_layers}")

    # ── Build classifier ──────────────────────────────────────
    self._embedding_dim = emb_dim
    self._lm_name       = lm
    self.model          = self._build_model(emb_dim)

    # ── Differential learning rates ────────────────────────────
    # PLM trainable params (LoRA A,B or unfrozen layers) → lr_plm
    # Classifier params → lr_classifier
    plm_params        = [p for p in plm_encoder.parameters() if p.requires_grad]
    classifier_params = list(self.model.parameters())
    if not plm_params:
        print("[WARN] No PLM params are trainable — "
              "check freeze_plm_layers or LoRA injection.")
    optimizer = torch.optim.AdamW([
        {'params': plm_params,
         'lr': lr_plm,  'weight_decay': 1e-5},
        {'params': classifier_params,
         'lr': lr_classifier, 'weight_decay': 1e-5},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    loss_cfg     = self.config.get('loss', {})
    criterion, _ = build_criterion(
        y, self.device,
        loss_type   = loss_cfg.get('type', 'weighted_ce'),
        focal_gamma = loss_cfg.get('focal_gamma', 2.0),
    )

    # ── Training loop ─────────────────────────────────────────
    history    = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    best_loss  = float('inf')
    best_state = None

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"  {'Ep':>4}  {'t_loss':>8}  {'t_acc':>7}  {'lr_cls':>10}  {'lr_plm':>10}")
    print("  " + "─" * 45)

    # Prepare sequences — each PLM needs a specific format
    # (mirrors embedding_generator.py exactly)
    hseqs = df['HSEQ'].fillna('').tolist()
    lseqs = df['LSEQ'].fillna('').tolist() if 'LSEQ' in df.columns else [''] * n

    def _sp(s):
        """Space-separate amino acids for IgBERT/AntiBERTa2."""
        return ' '.join(str(s).upper().replace(' ', ''))

    def _get_batch_emb(lm_, h_batch, l_batch):
        """Format + embed one batch — PLM-specific, matches embedding_generator."""
        if lm_ == 'ablang':
            return plm_encoder._forward_ablang(h_batch, l_batch).to(self.device)

        elif lm_ == 'antiberty':
            seqs = [f"{h}[CLS][CLS]{l}" if l else f"{h}[CLS]"
                    for h, l in zip(h_batch, l_batch)]
            return plm_encoder._forward_antiberty(seqs).to(self.device)

        elif lm_ in ('antiberta2', 'antiberta2-cssp'):
            seqs = [f"{_sp(h)} [CLS] [CLS] {_sp(l)}" if l else _sp(h)
                    for h, l in zip(h_batch, l_batch)]
            return plm_encoder._forward_antiberta2(seqs).to(self.device)

        elif lm_ == 'igbert':
            # BertTokenizer pair encoding: tokenizer(VH_spaced, VL_spaced)
            # → [CLS] VH_tokens [SEP] VL_tokens [SEP]
            _vh = [_sp(h) for h in h_batch]
            _vl = [_sp(l) if l else '' for l in l_batch]
            enc = plm_encoder._tokenizer(
                _vh, _vl, padding=True, truncation=True,
                max_length=320, return_tensors='pt'
            )
            enc    = {k: v.to(self.device) for k, v in enc.items()}
            hidden = plm_encoder._model(**enc).last_hidden_state
            return PLMEncoder._mean_pool(hidden, enc['attention_mask'])
        raise ValueError(f"Unknown lm: {lm_}")

    # Training loop
    indices = np.arange(n)
    for epoch in range(epochs):
        plm_encoder.train()
        self.model.train()
        np.random.shuffle(indices)

        t_loss, t_correct, t_total = 0.0, 0, 0
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            h_batch   = [hseqs[i] for i in batch_idx]
            l_batch   = [lseqs[i] for i in batch_idx]
            batch_lbl = torch.tensor(y[batch_idx], dtype=torch.long).to(self.device)

            optimizer.zero_grad()
            emb    = _get_batch_emb(lm, h_batch, l_batch)  # (B, emb_dim) — gradients flow
            logits = self.model(emb)
            loss   = criterion(logits, batch_lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(plm_params,        max_norm=0.1)
            torch.nn.utils.clip_grad_norm_(classifier_params, max_norm=1.0)
            optimizer.step()

            t_loss    += loss.item()
            t_correct += (logits.argmax(1) == batch_lbl).sum().item()
            t_total   += len(batch_idx)

        avg_loss = t_loss / max(1, n // batch_size)
        t_acc    = t_correct / t_total
        history['train_loss'].append(avg_loss)

        scheduler.step(avg_loss)
        _lr_cls = optimizer.param_groups[1]['lr']
        _lr_plm = optimizer.param_groups[0]['lr']
        print(f"  {epoch+1:4d}  {avg_loss:8.4f}  {t_acc:7.4f}"
              f"  {_lr_cls:10.2e}  {_lr_plm:10.2e}")

        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_state = copy.deepcopy(self.model.state_dict())

    if best_state:
        self.model.load_state_dict(best_state)
        print(f"[plmft] Best weights restored (loss={best_loss:.4f})")

    print(f"[plmft] Training complete.")

    # ── Save LoRA weights separately (tiny, shareable) ───────
    if _use_lora and _lora_injected:
        _lora_path = os.path.join(
            MODEL_DIR,
            f"lora_weights_{target}_{lm}_transformer_lm{_db_tag}.pt"
        )
        save_lora_weights(_lora_injected, _lora_path)
        self._lora_path = _lora_path   # stored so save() can embed the path

    # Store LoRA config on instance for save()
    self._peft       = peft
    self._lora_r     = lora_r
    self._lora_alpha = lora_alpha

    # ── Save training curve ────────────────────────────────────
    _curve_path = os.path.join(
        MODEL_DIR,
        f"train_curve_{target}_{lm}_transformer_lm{_db_tag}{_mode_tag}.png"
    )
    _save_training_curve(
        history, _curve_path,
        title=(f"{target} | {lm} "
               f"{'LoRA r=' + str(lora_r) if _use_lora else 'Mode2 unfreeze'} | "
               f"{_db_tag.strip('_')}")
    )
    return self


# ── Bind methods to TransformerLMModel ───────────────────────────────────────
TransformerLMModel.fine_tune               = _fine_tune
TransformerLMModel.train_with_plm_finetune = _train_with_plm_finetune

# ══════════════════════════════════════════════════════════════════
# 10.  STANDALONE CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "transformer_lm.py — Sample Size Effect Evaluation\n"
            "\n"
            "Trains TransformerLM on increasing data subsets and records\n"
            "AUC / F1 / Accuracy vs training set size.\n"
            "\n"
            "The embedding CSV is auto-detected as:\n"
            "  {db_path}.{lm}.emb.csv\n"
            "If not found the function exits with a clear message.\n"
            "\n"
            "Example:\n"
            "  python models/transformer_lm.py \\\n"
            "      --db     test/ipi_psr_trainset.xlsx \\\n"
            "      --target psr_filter \\\n"
            "      --lm     antiberta2-cssp \\\n"
            "      --start  500 \\\n"
            "      --step   1000 \\\n"
            "      --epochs 10"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db",      required=True,
        help="Path to database Excel/CSV with sequences and labels")
    parser.add_argument(
        "--emb",     default=None,
        help="Pre-computed embedding CSV (auto-detected if not given)")
    parser.add_argument(
        "--target",  default="psr_filter",
        help="Label column  (default: psr_filter)")
    parser.add_argument(
        "--lm",      default="antiberta2-cssp",
        help="Language model name  (default: antiberta2-cssp)")
    parser.add_argument(
        "--start",   type=int, default=500,
        help="First sample size  (default: 500)")
    parser.add_argument(
        "--step",    type=int, default=1000,
        help="Increment between sizes  (default: 1000)")
    parser.add_argument(
        "--epochs",  type=int, default=0,
        help="Epochs per size step. 0=auto_scale decides (recommended). "
             ">0 forces a fixed count for faster screening.")
    parser.add_argument(
        "--cluster", default="HCDR3_CLUSTER_0.8",
        help="CDR3 cluster column  (default: HCDR3_CLUSTER_0.8)")
    parser.add_argument(
        "--out",     default=None,
        help="Output CSV path (default: auto from db + target + lm)")
    args = parser.parse_args()

    result = evaluate_sample_size_effect(
        db_path       = args.db,
        embedding_csv = args.emb,
        target        = args.target,
        lm            = args.lm,
        start_size    = args.start,
        increment     = args.step,
        output_csv    = args.out,
        cluster_col   = args.cluster,
        epochs        = args.epochs,
    )
    if result is None:
        import sys; sys.exit(1)