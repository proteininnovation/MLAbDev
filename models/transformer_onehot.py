# models/transformer_onehot.py
# Transformer with One-Hot Encoding + HCDR3 Attention
# IPI Antibody Developability Prediction Platform
# Created by Hoan Nguyen | Final Production Version — DEC-2025
# Updated: All fixes applied — see [FIX] and [MOD] comments

import os
import sys
import copy

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config import MODEL_DIR

# ── Soft import: threshold_optimizer ─────────────────────────────────────────
# Runs automatically at the end of kfold_validation().
# If utils/threshold_optimizer.py is absent nothing breaks —
# kfold still completes and threshold defaults to 0.5 at predict time.
try:
    from utils.threshold_optimizer import run_full_threshold_pipeline
    _THRESHOLD_OPT_AVAILABLE = True
except ImportError:
    _THRESHOLD_OPT_AVAILABLE = False

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, Subset
from captum.attr import IntegratedGradients
from matplotlib.patches import Patch
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, roc_curve, auc
)

# ── Amino acid alphabet ────────────────────────────────────────────────────────
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_INDEX  = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}

MAX_VH_HARD_LIMIT = 150
MAX_VL_HARD_LIMIT = 150

# [FIX-5] Pre-built ASCII lookup table for fast one-hot encoding.
#   Converts ASCII byte values directly to amino-acid indices.
#   Unknown characters map to 20 (out of range → ignored when indexing).
#   Built once at module load — shared across all encoding calls.
_AA_LOOKUP = np.full(128, 20, dtype=np.int32)
for _aa, _idx in AA_TO_INDEX.items():
    _AA_LOOKUP[ord(_aa)] = _idx


# ══════════════════════════════════════════════════════════════════════════════
# POSITIONAL ENCODING (Sinusoidal - standard & effective for sequences)
# ══════════════════════════════════════════════════════════════════════════════
class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Works well for protein/antibody sequences (lengths < 300).
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.transpose(0, 1))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

# ══════════════════════════════════════════════════════════════════════════════
# 1.  ENCODING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def one_hot_encode_sequence_2d(sequence: str, max_length: int) -> np.ndarray:
    """
    Returns 2D one-hot matrix of shape (max_length, 20).

    [FIX-5] Uses ASCII lookup + np.frombuffer instead of a Python character loop.
    Avoids per-character dictionary lookups — ~1.7× faster for long sequences.
    """
    sequence = sequence.replace('-', '').upper()[:max_length]
    out      = np.zeros((max_length, len(AMINO_ACIDS)), dtype=np.float32)
    if not sequence:
        return out
    chars  = np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)
    aa_idx = _AA_LOOKUP[chars]          # shape (L,) — unknown AAs map to 20
    pos    = np.arange(len(chars))
    valid  = aa_idx < 20
    out[pos[valid], aa_idx[valid]] = 1
    return out


def one_hot_encode_sequence_1d(sequence: str, max_length: int) -> np.ndarray:
    """Returns flattened 1D vector (max_length*20,). Kept for backward compat."""
    return one_hot_encode_sequence_2d(sequence, max_length).flatten()


# Alias so external callers are not broken
def one_hot_encode_sequence(sequence: str, max_length: int) -> np.ndarray:
    return one_hot_encode_sequence_1d(sequence, max_length)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATASET
#
# Previous bottlenecks (all fixed):
#   • one_hot_encode called inside __getitem__ → re-encoded every epoch (fixed in last revision)
#   • Python char-loop inside encoder              → replaced with ASCII lookup [FIX-5]
#   • torch.tensor() in __getitem__               → replaced with torch.from_numpy [MOD-SPEED-2]
# ══════════════════════════════════════════════════════════════════════════════

class AntibodyDataset(Dataset):
    def __init__(self, heavy_seqs, light_seqs, hcdr3_seqs, labels, barcodes,
                 max_heavy_len=135, max_light_len=135, max_hcdr3_len=25,
                 vh_only=False):
        """
        vh_only=False (default) : branch-1 = VH+VL concatenated  → (270, 20)
        vh_only=True            : branch-1 = VH only              → (135, 20)
                                  light_seqs is ignored.
        """
        n      = len(labels)
        b1_len = max_heavy_len if vh_only else (max_heavy_len + max_light_len)

        # Pre-allocate and encode ALL sequences once at construction
        self._encoding = np.zeros((n, b1_len,        len(AMINO_ACIDS)), dtype=np.float32)
        self._cdr3     = np.zeros((n, max_hcdr3_len, len(AMINO_ACIDS)), dtype=np.float32)

        for i in range(n):
            h = one_hot_encode_sequence_2d(heavy_seqs[i], max_heavy_len)   # (135, 20)
            if vh_only:
                self._encoding[i] = h                                       # (135, 20)
            else:
                l = one_hot_encode_sequence_2d(
                    light_seqs[i] if (light_seqs is not None and light_seqs[i]) else '',
                    max_light_len)
                self._encoding[i] = np.concatenate([h, l], axis=0)         # (270, 20)
            self._cdr3[i] = one_hot_encode_sequence_2d(hcdr3_seqs[i], max_hcdr3_len)

        self._labels   = np.asarray(labels, dtype=np.int64)
        self._barcodes = list(barcodes)
        self._h_seqs   = list(heavy_seqs)
        self._l_seqs   = list(light_seqs) if (light_seqs is not None) else [''] * n
        self._hcdr3    = list(hcdr3_seqs)

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        # torch.from_numpy wraps existing memory — no copy
        return (
            torch.from_numpy(self._encoding[idx]),   # (270, 20)
            torch.from_numpy(self._cdr3[idx]),        # (25,  20)
            torch.tensor(self._labels[idx]),
            self._barcodes[idx],
            self._h_seqs[idx],
            self._l_seqs[idx],
            self._hcdr3[idx],
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3.  LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss — down-weights easy negatives, focuses on hard examples.
    Recommended for large, highly imbalanced datasets (>50k, minority < 15%).
    gamma=2.0 is a standard starting point.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce   = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt   = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MODEL — DevelopabilityClassifier
# ══════════════════════════════════════════════════════════════════════════════

class DevelopabilityClassifier(nn.Module):
    """
    Dual-branch Transformer with CDR3-guided cross-attention + fixes:
      • Sinusoidal positional encoding (order awareness)
      • Proper padding masks (ignores padded positions)
      • Independent encoders per branch
      • Optional CLS support prepared (commented)
    """
    def __init__(self, config, lm_mode='onehot'):
        super().__init__()
        cfg = config['model']
        hidden_dim = cfg['hidden_dim']
        num_layers = cfg['num_layers']
        nhead = cfg['num_heads']
        dim_feedforward = cfg['dim_feedforward']
        dropout = cfg['dropout']

        sl = config.get('sequence_lengths', {})
        vh_len = sl.get('max_vh_len', 135)
        vl_len = sl.get('max_vl_len', 135)
        cdr3_len = sl.get('max_hcdr3_len', 25)

        self.branch1_len = vh_len if lm_mode == 'onehot_vh' else (vh_len + vl_len)
        self.lm_mode = lm_mode
        self.hidden_dim = hidden_dim

        # Input projections
        self.input_fc = nn.Linear(len(AMINO_ACIDS), hidden_dim)
        self.cdr3_fc = nn.Linear(len(AMINO_ACIDS), hidden_dim)

        # Positional encodings
        self.pos_enc = SinusoidalPositionalEncoding(hidden_dim, max_len=512)
        self.cdr3_pos_enc = SinusoidalPositionalEncoding(hidden_dim, max_len=512)

        def _make_encoder():
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            return nn.TransformerEncoder(
                layer, num_layers=num_layers, enable_nested_tensor=False
            )

        self.transformer = _make_encoder()
        self.cdr3_transformer = _make_encoder()

        self.attention = nn.MultiheadAttention(hidden_dim, nhead, batch_first=True, dropout=dropout)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 2),
        )

    def _create_padding_mask(self, encoding: torch.Tensor) -> torch.Tensor:
        """Create padding mask from one-hot: True where token is all-zero (padding)."""
        # encoding: (B, L, 20) → sum over AA dim == 0 means padding
        return (encoding.sum(dim=-1) == 0)  # (B, L)

    def forward(self, encoding, encoding_cdr3):
        # encoding: (B, branch1_len, 20)
        # encoding_cdr3: (B, 25, 20)

        # --- Branch 1: VH ± VL ---
        x = self.input_fc(encoding)                    # (B, L1, H)
        x = self.pos_enc(x)                            # Add positional info

        pad_mask1 = self._create_padding_mask(encoding)  # (B, L1)

        x = self.transformer(x, src_key_padding_mask=pad_mask1)  # (B, L1, H)
        x_pooled = x.mean(dim=1)                       # mean pool (robust with mask)

        # --- Branch 2: HCDR3 ---
        cdr3_x = self.cdr3_fc(encoding_cdr3)
        cdr3_x = self.cdr3_pos_enc(cdr3_x)

        pad_mask_cdr3 = self._create_padding_mask(encoding_cdr3)

        cdr3_x = self.cdr3_transformer(cdr3_x, src_key_padding_mask=pad_mask_cdr3)
        cdr3_pooled = cdr3_x.mean(dim=1)

        # --- CDR3-guided Cross Attention ---
        # query from CDR3 summary, keys/values from full chain (with mask)
        attn_out, _ = self.attention(
            cdr3_pooled.unsqueeze(1),      # (B, 1, H)
            x,                             # (B, L1, H)
            x,
            key_padding_mask=pad_mask1     # ignore padding in full chain
        )

        combined = torch.cat((x_pooled, attn_out.squeeze(1)), dim=1)  # (B, 2H)
        return self.head(combined)
    
# ══════════════════════════════════════════════════════════════════════════════
# 5.  WRAPPER — TransformerOneHotModel
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLD OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════════

# _find_best_threshold removed — replaced by utils/threshold_optimizer.py
# which provides a richer set of methods (youden, f1, f2, fbeta, cost,
# precision_at, recall_at) plus 4-panel plots, JSON export, and fold stability.
# Called automatically at the end of kfold_validation() via run_full_threshold_pipeline().


class TransformerOneHotModel:

    _DEFAULT_CONFIG = {
        'mode': 'manual',   # 'auto' | 'manual'  — can be set in YAML
        'model': {
            'hidden_dim':       64,   # 4.9x faster than 128; head_dim=64/4=16 = same as 128/8
            'num_heads':         4,   # must match: hidden_dim / num_heads = 16
            'num_layers':        4,   # 4 layers sufficient for sequence classification at 12k
            'dim_feedforward': 256,   # 4× hidden_dim; FFN is 78% of compute — halving saves most
            'dropout':         0.3,
        },
        'sequence_lengths': {
            'max_vh_len':    135,
            'max_vl_len':    135,
            'max_hcdr3_len':  25,
        },
        'training': {
            'epochs':       50,
            'batch_size':   64,    # raised from 32 — halves steps/epoch; model converges fast
            'lr':           1e-4,
            'weight_decay': 1e-5,
            'loss_type':    'weighted_ce',   # 'label_smooth'          balanced ≥40%  (best F1/Acc)
                                            # 'weighted_label_smooth'  imbalanced, best F1/Acc
                                            # 'ce'                     balanced, no calibration
                                            # 'weighted_ce'            imbalanced, safe default
                                            # 'focal'                  severe imbalance <10%
            'label_smoothing': 0.1,         # ε for label_smooth / weighted_label_smooth (0.05–0.15)
            'focal_gamma':  2.0,
            'focal_alpha':  0.25,
            'patience':     5,     # lowered from 7/10 — convergence typically by ep 5-8
        },
        'scheduler': {
            'factor':   0.5,   # LR multiplier when plateau detected  (1e-4 → 5e-5)
            'patience': 5,     # epochs without improvement before LR halves
                               # 5 for label_smooth (needs time for probs to spread from 0.5)
                               # 3 for ce/weighted_ce (faster convergence signal)
        },
        'mutagenesis': {'amino_acids': "ACDEFGHIKLMNPQRSTVWY"},
        # [FIX-YAML] interpretability block now read from config.
        #   Previously ig_n_steps and top_features were hardcoded (50 and 60).
        #   val_split is noted but unused — kfold_validation handles all splitting.
        'interpretability': {
            'ig_n_steps':   50,    # steps for Integrated Gradients (higher = more accurate, slower)
            'top_features': 60,    # number of top positions shown in global IG bar chart
        },
    }

    # ── auto-detect ───────────────────────────────────────────────────────────

    @staticmethod
    def auto_detect_config(n: int, pos_rate: float) -> dict:
        """
        Derive a fully-reasoned config from dataset size and class balance alone.
        Returns a config dict in the same shape as _DEFAULT_CONFIG — can be passed
        directly to train() / kfold_validation() or merged with a YAML override.

        Size tiers
        ──────────
          xs  n < 5k      hidden=64,  layers=4, ffn=256,  dropout=0.40, batch=16,  patience=12
          sm  5k–20k      hidden=128, layers=6, ffn=512,  dropout=0.30, batch=32,  patience=10
          md  20k–80k     hidden=192, layers=6, ffn=768,  dropout=0.20, batch=64,  patience=8
          lg  80k–200k    hidden=256, layers=8, ffn=1024, dropout=0.15, batch=128, patience=7
          xl  >200k       hidden=256, layers=8, ffn=1024, dropout=0.10, batch=256, patience=7

        Balance + loss rule  (stricter of size and balance wins)
        ──────────────────────────────────────────────────────────
          balanced   40–60%         → ce
          mild       20–40%         → weighted_ce
          moderate   10–20%         → weighted_ce
          severe     5–10%          → focal
          extreme    <5%            → focal  (+ warning)
          large+imb  n>80k AND
                     pos<40%        → focal  (overrides weighted_ce)
        """
        # ── size tier ─────────────────────────────────────────────────────────
        if n < 5_000:
            tier    = 'xs'
            hidden, layers, ffn = 64,  4, 256
            dropout, batch, patience = 0.40, 16,  12
        elif n < 20_000:
            tier    = 'sm'
            hidden, layers, ffn = 64, 4, 256
            dropout, batch, patience = 0.30, 64,  5
        elif n < 80_000:
            tier    = 'md'
            hidden, layers, ffn = 192, 6, 768
            dropout, batch, patience = 0.20, 64,   8
        elif n < 200_000:
            tier    = 'lg'
            hidden, layers, ffn = 256, 8, 1024
            dropout, batch, patience = 0.15, 128,  7
        else:
            tier    = 'xl'
            hidden, layers, ffn = 256, 8, 1024
            dropout, batch, patience = 0.10, 256,  7

        # num_heads: chosen so head_dim = hidden/heads = 16 across all tiers
        # 64→4 (64/4=16), 128→8 (128/8=16), 192→8 (192/8=24), 256→8 (256/8=32)
        nhead = 8 if hidden >= 128 else 4

        # ── balance + loss rule (stricter wins) ───────────────────────────────
        balanced   = 0.40 <= pos_rate <= 0.60
        severe     = pos_rate < 0.10 or pos_rate > 0.90
        large_imb  = (n > 80_000) and (pos_rate < 0.40 or pos_rate > 0.60)

        if balanced:
            loss_type     = 'ce'
            balance_label = 'balanced'
        elif severe or large_imb:
            loss_type     = 'focal'
            balance_label = 'severe' if severe else 'large+imbalanced'
        else:
            loss_type     = 'weighted_ce'
            balance_label = 'mild-moderate imbalanced'

        # warn on extreme imbalance
        if pos_rate < 0.05 or pos_rate > 0.95:
            print(f"  [WARN] Extreme class imbalance (pos_rate={pos_rate:.1%}). "
                  f"Consider upsampling the minority class before training.")

        # ── LR: scale linearly with batch size relative to base (batch=32, lr=1e-4)
        lr = round(1e-4 * (batch / 32), 6)

        # ── IG steps: 50 for tiny datasets, 200 for everything else
        ig_steps = 50 if n < 5_000 else 200

        # ── assemble final config (deep copy of defaults, then override) ──────
        cfg = copy.deepcopy(TransformerOneHotModel._DEFAULT_CONFIG)
        cfg['model'].update({
            'hidden_dim':      hidden,
            'num_heads':       nhead,
            'num_layers':      layers,
            'dim_feedforward': ffn,
            'dropout':         dropout,
        })
        cfg['training'].update({
            'batch_size': batch,
            'lr':         lr,
            'patience':   patience,
            'loss_type':  loss_type,
        })
        cfg['interpretability']['ig_n_steps'] = ig_steps

        # metadata stored under _auto — used by print_config_report and YAML export
        cfg['_auto'] = {
            'n':         n,
            'pos_rate':  round(pos_rate, 4),
            'size_tier': tier,
            'balance':   balance_label,
        }
        return cfg

    @staticmethod
    def export_config_yaml(cfg: dict, path: str) -> None:
        """
        Save a config dict (auto or manual) to a YAML file.
        Strips the internal _auto metadata block before writing so the file
        is clean and can be loaded back directly by __init__.
        """
        import yaml
        clean = {k: v for k, v in cfg.items() if k != '_auto'}
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(clean, f, default_flow_style=False, sort_keys=False)
        print(f"[export_config_yaml] → {path}")

    @staticmethod
    def print_config_report(cfg: dict, lm_mode: str = 'onehot') -> None:
        """
        Print every parameter loaded from the YAML file, plus one-hot encoding dims.
        Called automatically from __init__() and _resolve_config() (auto mode).
        """
        a  = cfg.get('_auto', {})
        m  = cfg['model']
        t  = cfg['training']
        sl = cfg.get('sequence_lengths', {})
        ig = cfg.get('interpretability', {})
        mu = cfg.get('mutagenesis', {})

        W    = 62
        sep  = '═' * W
        sep2 = '─' * W

        print(f"\n{sep}")
        if a:
            print(f"  TransformerOneHot  ·  AUTO CONFIG")
            print(f"  Dataset : n={a['n']:,}  pos_rate={a['pos_rate']:.1%}  "
                  f"tier={a['size_tier']}  balance={a['balance']}")
        else:
            print(f"  TransformerOneHot  ·  MANUAL CONFIG  "
                  f"(mode: {cfg.get('mode','manual')})")
        print(sep2)

        # ── Model architecture ────────────────────────────────────
        head_dim = m['hidden_dim'] // max(m['num_heads'], 1)
        n_params_branch = 2 * m['num_layers'] * (
            4 * m['hidden_dim']**2
            + 2 * m['hidden_dim'] * m['dim_feedforward']
            + m['hidden_dim']
        )
        n_params_head = 2 * m['hidden_dim'] * 2
        n_params_total = n_params_branch + n_params_head
        print(f"  MODEL ARCHITECTURE")
        print(f"    hidden_dim      : {m['hidden_dim']}")
        print(f"    num_heads       : {m['num_heads']}  →  head_dim = {head_dim}")
        print(f"    num_layers      : {m['num_layers']}  (per branch × 2 branches)")
        print(f"    dim_feedforward : {m['dim_feedforward']}")
        print(f"    dropout         : {m['dropout']}")
        print(f"    trainable params: ~{n_params_total/1e6:.2f}M")
        print(sep2)

        # ── One-hot encoding dims ─────────────────────────────────
        aa_dim    = len(AMINO_ACIDS)
        vh_len    = sl.get('max_vh_len',    135)
        vl_len    = sl.get('max_vl_len',    135)
        cdr3_len  = sl.get('max_hcdr3_len',  25)
        hl_tokens = vh_len + vl_len
        hl_dim    = hl_tokens * aa_dim
        cdr3_dim  = cdr3_len  * aa_dim
        print(f"  ONE-HOT ENCODING  (alphabet: {aa_dim} AAs)")
        if lm_mode == 'onehot_vh':
            b1_dim = vh_len * aa_dim
            print(f"    VH   max_len={vh_len:3d}  → ({vh_len:3d} × {aa_dim}) = {b1_dim:,} dims  ← branch-1 input")
            print(f"    VL   not used  (onehot_vh mode)")
            print(f"    HCDR3 max_len={cdr3_len:2d}  → ({cdr3_len:2d}  × {aa_dim}) = {cdr3_dim:,} dims  ← branch-2 input")
            print(f"    Total per sample: {b1_dim + cdr3_dim:,} dims  ({b1_dim:,} + {cdr3_dim:,})")
        else:
            print(f"    VH   max_len={vh_len:3d}  → ({vh_len:3d} × {aa_dim}) = {vh_len*aa_dim:,} dims")
            print(f"    VL   max_len={vl_len:3d}  → ({vl_len:3d} × {aa_dim}) = {vl_len*aa_dim:,} dims")
            print(f"    VH+VL concat    → ({hl_tokens:3d} × {aa_dim}) = {hl_dim:,} dims  ← branch-1 input")
            print(f"    HCDR3 max_len={cdr3_len:2d}  → ({cdr3_len:2d}  × {aa_dim}) = {cdr3_dim:,} dims  ← branch-2 input")
            print(f"    Total per sample: {hl_dim + cdr3_dim:,} dims  ({hl_dim:,} + {cdr3_dim:,})")
        print(sep2)

        # ── Training ──────────────────────────────────────────────
        loss_type  = t.get('loss_type', 'weighted_ce')
        loss_extra = ""
        if loss_type == 'focal':
            loss_extra = (f"  (gamma={t.get('focal_gamma',2.0)}"
                          f"  alpha={t.get('focal_alpha',0.25)})")
        elif loss_type in ('label_smooth', 'weighted_label_smooth'):
            loss_extra = f"  (ε={t.get('label_smoothing',0.1)})"
            if loss_type == 'weighted_label_smooth':
                loss_extra += "  +class_weights"
        print(f"  TRAINING")
        print(f"    epochs          : {t.get('epochs',50)}")
        print(f"    batch_size      : {t.get('batch_size',64)}")
        print(f"    lr              : {t.get('lr',1e-4):.2e}")
        print(f"    weight_decay    : {t.get('weight_decay',1e-5):.2e}")
        print(f"    patience        : {t.get('patience',5)}  (early stopping on val_AUC)")
        print(f"    loss_type       : {loss_type}{loss_extra}")
        print(sep2)

        # ── Sequence lengths ──────────────────────────────────────
        print(f"  SEQUENCE LENGTHS  (must match your dataset)")
        print(f"    max_vh_len      : {vh_len}")
        print(f"    max_vl_len      : {vl_len}")
        print(f"    max_hcdr3_len   : {cdr3_len}")
        print(sep2)

        # ── Interpretability ──────────────────────────────────────
        print(f"  INTERPRETABILITY")
        print(f"    ig_n_steps      : {ig.get('ig_n_steps',50)}")
        print(f"    top_features    : {ig.get('top_features',60)}")
        print(sep2)

        # ── Scheduler ────────────────────────────────────────────
        sch = cfg.get('scheduler', {})
        print(f"  SCHEDULER  (ReduceLROnPlateau)")
        print(f"    factor          : {sch.get('factor', 0.5)}  (LR multiplied on plateau)")
        print(f"    patience        : {sch.get('patience', 5)}  (epochs without AUC gain before LR drops)")
        print(sep2)

        # ── Mutagenesis ───────────────────────────────────────────
        print(f"  MUTAGENESIS")
        print(f"    amino_acids     : {mu.get('amino_acids', AMINO_ACIDS)}")
        print(f"{sep}\n")


    def __init__(self, config_path="config/transformer_onehot.yaml", config: dict = None):
        """
        config_path : str  — path to YAML (merged over _DEFAULT_CONFIG)
        config      : dict — from auto_detect_config() or built manually;
                             takes precedence over config_path when provided.
        Auto mode example:
            cfg   = TransformerOneHotModel.auto_detect_config(n=12000, pos_rate=0.5)
            model = TransformerOneHotModel(config=cfg)
        """
        import yaml
        if config is not None:
            self.config = _deep_merge(copy.deepcopy(self._DEFAULT_CONFIG), config)
            tier = config.get("_auto", {}).get("size_tier", "?")
            mode = f"auto (tier={tier})" if "_auto" in config else "dict"
            print(f"[TransformerOneHotModel] config ← {mode}")
        elif os.path.exists(config_path):
            with open(config_path) as f:
                user_cfg = yaml.safe_load(f) or {}
            self.config = _deep_merge(copy.deepcopy(self._DEFAULT_CONFIG), user_cfg)
            print(f"[TransformerOneHotModel] config ← {config_path}")
        else:
            self.config = copy.deepcopy(self._DEFAULT_CONFIG)
            print(f"[TransformerOneHotModel] {config_path} not found — using built-in defaults")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = None

        cfg = self.config['sequence_lengths']
        self.max_heavy_len = cfg['max_vh_len']
        self.max_light_len = cfg['max_vl_len']
        self.max_hcdr3_len = cfg['max_hcdr3_len']

        # [FIX-YAML] Warn if YAML sets a small batch_size — 16 was the old default
        #   and is the most common source of slow training on ≥8k datasets.
        bs = self.config['training'].get('batch_size', 32)
        if bs < 32:
            print(f"  [WARN] config batch_size={bs} — consider ≥32 for faster training "
                  f"(set training.batch_size: 32 in your YAML)")

        # [FIX-YAML] val_split is present in the YAML but is NOT used by this model.
        #   All splitting is done via kfold_validation() or train_test_split_group_stratified().
        #   Keeping it here so the YAML is self-documenting, but issuing a notice.
        if self.config['training'].get('val_split', 0.0) > 0:
            print("  [NOTE] training.val_split is set but unused — "
                  "splits are controlled by kfold_validation() / train_test_split_group_stratified()")
        # lm_mode controls branch-1 input: 'onehot' = VH+VL, 'onehot_vh' = VH only
        # Set to default here; overridden by set_lm_mode() in train() / kfold_validation()
        self.lm_mode      = 'onehot'
        self.encoding_dim = (self.max_heavy_len + self.max_light_len) * len(AMINO_ACIDS)
        self.cdr3_dim     = self.max_hcdr3_len * len(AMINO_ACIDS)

        # Print full config + encoding dims on every init (manual mode only;
        # auto mode prints after _resolve_config() when the final values are known)
        if self.config.get('mode', 'manual') == 'manual':
            TransformerOneHotModel.print_config_report(self.config, lm_mode=self.lm_mode)

    # ── helpers ───────────────────────────────────────────────────────────────

    def set_lm_mode(self, embedding_lm: str) -> None:
        """
        Configure the model for VH+VL (onehot) or VH-only (onehot_vh) input.
        Must be called before _build_model() so DevelopabilityClassifier
        receives the correct branch1_len.

          embedding_lm    branch-1        encoding_dim
          ─────────────────────────────────────────────
          'onehot'        VH+VL (270×20)  5,400
          'onehot_vh'     VH    (135×20)  2,700
        """
        self.lm_mode = embedding_lm
        aa = len(AMINO_ACIDS)
        if embedding_lm == 'onehot_vh':
            self.encoding_dim = self.max_heavy_len * aa
        else:
            self.encoding_dim = (self.max_heavy_len + self.max_light_len) * aa

    def _vh_only(self) -> bool:
        """True when operating in VH-only mode (lm_mode='onehot_vh')."""
        return self.lm_mode == 'onehot_vh'

    def _get_light_seqs(self, X, n: int):
        """
        Return light sequences from X, or empty strings for VH-only mode.
        Always returns an array/list of length n.
        """
        if self._vh_only():
            return [''] * n
        if hasattr(X, 'columns') and 'LSEQ' in X.columns:
            return X['LSEQ'].values
        return [''] * n

    def _resolve_config(self, y: 'np.ndarray') -> None:
        """
        Called at the start of train() and kfold_validation().

        mode: manual  — use config exactly as loaded from YAML. Nothing changes.
        mode: auto    — derive model + training settings entirely from data.
                        All model/training YAML values are IGNORED in auto mode.
                        Only these YAML sections are always respected:
                          • sequence_lengths   (dataset-specific, never auto-derived)
                          • interpretability   (speed/accuracy tradeoff, not data-dependent)
                          • mutagenesis        (alphabet — never changes)
                          • training.epochs    (kept as a hard cap on max epochs)
        """
        mode = self.config.get('mode', 'manual')
        if mode not in ('auto', 'manual'):
            raise ValueError(f"config.mode must be 'auto' or 'manual', got '{mode!r}'")
        if mode == 'manual':
            return   # use YAML as-is — nothing to do

        # ── auto mode: derive everything from data ────────────────────────────
        n        = len(y)
        pos_rate = float(np.mean(np.asarray(y, dtype=int)))
        resolved = TransformerOneHotModel.auto_detect_config(n, pos_rate)

        # Always carry over YAML-controlled sections (not data-dependent)
        for section in ('sequence_lengths', 'mutagenesis', 'interpretability'):
            if section in self.config:
                resolved[section] = copy.deepcopy(self.config[section])

        # epochs: treat YAML value as a hard cap (auto never exceeds it)
        yaml_epochs = self.config['training'].get('epochs', 30)
        resolved['training']['epochs'] = min(resolved['training']['epochs'], yaml_epochs)

        self.config = resolved
        TransformerOneHotModel.print_config_report(self.config, lm_mode=self.lm_mode)

    def _build_model(self):
        return DevelopabilityClassifier(
            config=self.config, lm_mode=self.lm_mode
        ).to(self.device)

    def _build_criterion(self, y):
        """
        Centralised loss builder — shared by train() and kfold_validation().

        Decision table (use this to pick loss_type in YAML):
        ┌──────────────────────────┬───────────────────────┬──────────────────────────────┐
        │ Dataset                  │ min_class_rate        │ Recommended loss_type        │
        ├──────────────────────────┼───────────────────────┼──────────────────────────────┤
        │ Balanced, any size       │ ≥ 40%                 │ label_smooth  (best F1/Acc)  │
        │ Mildly imbalanced        │ 20–40%  (e.g. SEC)    │ weighted_label_smooth        │
        │ Moderately imbalanced    │ 10–20%                │ weighted_label_smooth        │
        │ Severely imbalanced      │ < 10%   (e.g. rare)   │ focal                        │
        └──────────────────────────┴───────────────────────┴──────────────────────────────┘

        WHY label_smooth alone is insufficient for imbalanced data:
          Label smoothing softens targets (0→0.05, 1→0.95 at ε=0.1).
          It does NOT change gradient weight between classes.
          The minority class remains under-represented in the loss signal.
          → Must combine with class weights to fix both problems at once:
              weight=  fixes minority class gradient imbalance
              smoothing= prevents overconfident majority class predictions

        loss_type options
        ─────────────────────────────────────────────────────────────────────
        'label_smooth'          CE + label smoothing, NO class weights.
                                For balanced data only (min_rate ≥ 40%).
                                ε = label_smoothing (default 0.1).

        'weighted_label_smooth' CE + class weights + label smoothing.
                                Best for ANY imbalanced dataset that also
                                needs high accuracy and F1.
                                Combines both fixes in one loss.

        'ce'                    Plain CE. Balanced data, no calibration needed.

        'weighted_ce'           CE + class weights. Imbalanced, no smoothing.

        'focal'                 FocalLoss + class weights. Severe imbalance
                                (minority < 10%) or very large datasets.
        """
        loss_type   = self.config['training'].get('loss_type', 'weighted_ce')
        smoothing   = self.config['training'].get('label_smoothing', 0.1)
        pos_rate    = float(np.mean(y))
        min_rate    = min(pos_rate, 1 - pos_rate)
        pos_w       = (1 - pos_rate) / (pos_rate + 1e-8)
        weight      = torch.tensor([1.0, pos_w], dtype=torch.float32).to(self.device)

        extra_note = ""
        if loss_type in ('label_smooth', 'weighted_label_smooth'):
            extra_note = f"  ε={smoothing}"
        if loss_type in ('weighted_ce', 'weighted_label_smooth', 'focal'):
            extra_note += f"  pos_weight={pos_w:.2f}"

        print(f"  [loss] pos_rate={pos_rate:.1%}  min_rate={min_rate:.1%}"
              f"  type={loss_type}{extra_note}")

        if loss_type == 'focal':
            # Severe imbalance (minority < 10%) or large imbalanced datasets.
            # Focuses training on hard/misclassified examples.
            gamma     = self.config['training'].get('focal_gamma', 2.0)
            alpha_val = self.config['training'].get('focal_alpha', 0.25)
            alpha     = torch.tensor([1 - alpha_val, alpha_val],
                                      dtype=torch.float32).to(self.device)
            return FocalLoss(gamma=gamma, alpha=alpha)

        elif loss_type == 'weighted_label_smooth':
            # Best for imbalanced data when accuracy + F1 matter.
            # Class weights fix minority gradient; smoothing fixes calibration.
            # Works for ANY imbalance level — safe universal choice.
            return nn.CrossEntropyLoss(weight=weight, label_smoothing=smoothing)

        elif loss_type == 'label_smooth':
            # Balanced data only (min_rate ≥ 40%).
            # Smoothing alone; no class weights needed when data is balanced.
            if min_rate < 0.30:
                print(f"  [loss] WARNING: label_smooth without weights"
                      f" on imbalanced data (min_rate={min_rate:.1%}).")
                print(f"  [loss] Consider switching to: weighted_label_smooth")
            return nn.CrossEntropyLoss(label_smoothing=smoothing)

        elif loss_type == 'ce':
            # Plain CE — correct for balanced datasets.
            return nn.CrossEntropyLoss()

        else:
            # weighted_ce — reliable default for imbalanced datasets.
            return nn.CrossEntropyLoss(weight=weight)

    # ── train ─────────────────────────────────────────────────────────────────

    def train(self, X, y, val_X=None, val_y=None, epochs=None, batch_size=None,
              target: str = "model", save_plot: bool = True):
        """
        Full training loop with per-epoch logging and training curve plot.

        Parameters
        ----------
        X, y          : training data (DataFrame with HSEQ/LSEQ/CDR3, int labels)
        val_X, val_y  : optional validation split — enables val metrics + early stopping
        epochs        : override YAML value
        batch_size    : override YAML value
        target        : used in output filenames (plot, checkpoint)
        save_plot     : save training curve PNG to MODEL_DIR (default True)

        Per-epoch output (no validation):
          Ep  1/50  t_loss=0.6821  t_acc=0.5312  lr=1.00e-04

        Per-epoch output (with validation):
          Ep  1/50  t_loss=0.6821  t_acc=0.5312  |  v_loss=0.6544  v_acc=0.5780  v_auc=0.6120  lr=1.00e-04

        Mode is read from self.config['mode'] (set via YAML or config dict):
          mode: auto   — auto-detect config from data before training.
          mode: manual — use config exactly as loaded (default).
        """
        self._resolve_config(y)   # no-op in manual mode; rebuilds config in auto mode

        # Ensure encoding_dim and _vh_only() match the caller's embedding_lm.
        # train() doesn't receive embedding_lm directly, so the caller must call
        # set_lm_mode(embedding_lm) before train(), OR kfold_validation() does it.
        # Default lm_mode='onehot' is set in __init__ so standalone train() works.

        tcfg       = self.config['training']
        epochs     = epochs     or tcfg.get('epochs',     50)
        batch_size = batch_size or tcfg.get('batch_size', 64)
        lr         = tcfg.get('lr',           1e-4)
        wd         = tcfg.get('weight_decay', 1e-5)
        patience   = tcfg.get('patience',      5)

        heavy    = X['HSEQ'].values
        light    = self._get_light_seqs(X, len(y))
        hcdr3    = X['CDR3'].values
        barcodes = X['BARCODE'].values if 'BARCODE' in X else [f"ab_{i}" for i in range(len(y))]

        dataset = AntibodyDataset(heavy, light, hcdr3, y, barcodes,
                                  self.max_heavy_len, self.max_light_len, self.max_hcdr3_len,
                                  vh_only=self._vh_only())
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=(str(self.device) != 'cpu'))

        # ── Optional validation loader ────────────────────────────
        has_val     = (val_X is not None and val_y is not None)
        val_loader  = None
        if has_val:
            val_y    = np.asarray(val_y, dtype=int)
            val_bcs  = (val_X['BARCODE'].values if 'BARCODE' in val_X
                        else [f"val_{i}" for i in range(len(val_y))])
            val_ds   = AntibodyDataset(
                val_X['HSEQ'].values, self._get_light_seqs(val_X, len(val_y)),
                val_X['CDR3'].values, val_y, val_bcs,
                self.max_heavy_len, self.max_light_len, self.max_hcdr3_len,
                vh_only=self._vh_only()
            )
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                    num_workers=0, pin_memory=(str(self.device) != 'cpu'))

        self.model = self._build_model()
        criterion  = self._build_criterion(y)
        optimizer  = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        sch_cfg      = self.config.get('scheduler', {})
        sch_factor   = sch_cfg.get('factor',   0.5)
        sch_patience = sch_cfg.get('patience', 3)
        sch_mode     = sch_cfg.get('mode',     'min')  # min = step on v_loss (smooth)
        scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=sch_mode,
            factor=sch_factor, patience=sch_patience
        )

        # ── Print training summary ────────────────────────────────
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[train] n={len(y):,}  batch={batch_size}  epochs={epochs}  "
              f"lr={lr:.2e}  wd={wd:.2e}  patience={patience}")
        print(f"[train] device={self.device}  params={n_params:,}")
        print(f"[train] val={'yes  (early stopping on val_AUC)' if has_val else 'no  (no early stopping)'}")
        if has_val:
            print(f"[train] val_n={len(val_y):,}  pos_rate={val_y.mean():.1%}")
        header = (
            f"  {'Ep':>4}  {'t_loss':>8}  {'t_acc':>7}"
            + (f"  {'v_loss':>8}  {'v_acc':>7}  {'v_auc':>7}" if has_val else "")
            + f"  {'lr':>10}"
        )
        print(header)
        print("  " + "─" * (len(header) - 2))

        # ── History for plot ──────────────────────────────────────
        hist = {'t_loss': [], 't_acc': [],
                'v_loss': [], 'v_acc': [], 'v_auc': []}

        best_val_auc = -1.0
        best_state   = None
        patience_ctr = 0

        for epoch in range(epochs):
            # ── train pass ───────────────────────────────────────
            self.model.train()
            t_loss_sum, t_correct, t_total = 0.0, 0, 0
            for enc, cdr3_enc, lbl, *_ in loader:
                enc, cdr3_enc, lbl = (enc.to(self.device),
                                      cdr3_enc.to(self.device),
                                      lbl.to(self.device))
                optimizer.zero_grad()
                logits = self.model(enc, cdr3_enc)
                loss   = criterion(logits, lbl)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                t_loss_sum += loss.item()
                t_correct  += (logits.argmax(1) == lbl).sum().item()
                t_total    += lbl.size(0)

            t_loss = t_loss_sum / len(loader)
            t_acc  = t_correct  / t_total
            hist['t_loss'].append(t_loss)
            hist['t_acc'].append(t_acc)

            # ── validation pass ───────────────────────────────────
            if has_val:
                self.model.eval()
                v_loss_sum, v_correct, v_total = 0.0, 0, 0
                v_probs_all, v_true_all = [], []
                with torch.no_grad():
                    for enc, cdr3_enc, lbl, *_ in val_loader:
                        enc, cdr3_enc, lbl = (enc.to(self.device),
                                              cdr3_enc.to(self.device),
                                              lbl.to(self.device))
                        logits    = self.model(enc, cdr3_enc)
                        v_loss_sum += criterion(logits, lbl).item()
                        probs      = torch.softmax(logits, 1)[:, 1]
                        v_correct  += (logits.argmax(1) == lbl).sum().item()
                        v_total    += lbl.size(0)
                        v_probs_all.extend(probs.cpu().numpy())
                        v_true_all.extend(lbl.cpu().numpy())

                v_loss = v_loss_sum / len(val_loader)
                v_acc  = v_correct  / v_total
                try:
                    v_auc = roc_auc_score(v_true_all, v_probs_all)                             if len(set(v_true_all)) > 1 else 0.5
                except Exception:
                    v_auc = 0.5

                hist['v_loss'].append(v_loss)
                hist['v_acc'].append(v_acc)
                hist['v_auc'].append(v_auc)

                scheduler.step(v_loss)  # step on v_loss (smooth) — matches transformer_lm
                                        # early stopping still uses v_auc (correct)

                # ── early stopping on val_AUC ─────────────────────
                if v_auc > best_val_auc:
                    best_val_auc = v_auc
                    best_state   = copy.deepcopy(self.model.state_dict())
                    patience_ctr = 0
                else:
                    patience_ctr += 1

                log = (f"  {epoch+1:4d}  {t_loss:8.4f}  {t_acc:7.4f}"
                       f"  {v_loss:8.4f}  {v_acc:7.4f}  {v_auc:7.4f}"
                       f"  {optimizer.param_groups[0]['lr']:10.2e}")
            else:
                scheduler.step(t_loss)
                log = (f"  {epoch+1:4d}  {t_loss:8.4f}  {t_acc:7.4f}"
                       f"  {optimizer.param_groups[0]['lr']:10.2e}")

            print(log)

            if has_val and patience_ctr >= patience:
                print(f"  → Early stopping at epoch {epoch+1}  "
                      f"(best val_AUC={best_val_auc:.4f})")
                break

        # Restore best weights when val available
        if has_val and best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"  → Best weights restored  (val_AUC={best_val_auc:.4f})")

        print("[train] completed.")

        # ── Training curve plot ───────────────────────────────────
        if save_plot:
            n_ep   = len(hist['t_loss'])
            ep_ax  = range(1, n_ep + 1)
            n_panels = 3 if has_val else 2
            fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))

            # Panel 1: loss
            axes[0].plot(ep_ax, hist['t_loss'], 'b-o', ms=3, label='train')
            if has_val:
                axes[0].plot(ep_ax, hist['v_loss'], 'r-o', ms=3, label='val')
                axes[0].legend(fontsize=8)
            axes[0].set_title('Loss', fontsize=10)
            axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
            axes[0].grid(alpha=0.3)

            # Panel 2: accuracy
            axes[1].plot(ep_ax, hist['t_acc'], 'b-o', ms=3, label='train')
            if has_val:
                axes[1].plot(ep_ax, hist['v_acc'], 'r-o', ms=3, label='val')
                axes[1].legend(fontsize=8)
            axes[1].set_title('Accuracy', fontsize=10)
            axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
            axes[1].set_ylim(0, 1); axes[1].grid(alpha=0.3)

            # Panel 3: val AUC (only when val available)
            if has_val:
                best_ep  = int(np.argmax(hist['v_auc'])) + 1
                best_auc = max(hist['v_auc'])
                axes[2].plot(ep_ax, hist['v_auc'], 'g-o', ms=3, label='val AUC')
                axes[2].axvline(best_ep, color='gray', lw=1, linestyle='--',
                                label=f'best ep={best_ep}  AUC={best_auc:.4f}')
                axes[2].legend(fontsize=8)
                axes[2].set_title('Val AUC', fontsize=10)
                axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('AUC')
                axes[2].set_ylim(0, 1); axes[2].grid(alpha=0.3)

            m    = self.config['model']
            t    = self.config['training']
            info = (f"hidden={m['hidden_dim']}  layers={m['num_layers']}  "
                    f"ffn={m['dim_feedforward']}  heads={m['num_heads']}  "
                    f"dropout={m['dropout']}  "
                    f"batch={batch_size}  lr={lr:.2e}  "
                    f"loss={t.get('loss_type','weighted_ce')}")
            fig.suptitle(f"Training curve — {target}\n{info}", fontsize=8)
            plt.tight_layout()

            os.makedirs(MODEL_DIR, exist_ok=True)
            plot_path = os.path.join(MODEL_DIR, f"train_curve_{target}_onehot.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"[train] curve → {plot_path}")

        return self

    # ── kfold_validation ──────────────────────────────────────────────────────

    @classmethod
    def kfold_validation(
        cls,
        data,
        X,
        y,
        embedding_lm: str = "onehot",
        title:        str = "TransformerOneHot",
        kfold:        int = 10,
        target:       str = "sec_filter",
        cluster_col:  str = "HCDR3_CLUSTER_0.8",
        db_stem:      str = "",
    ):
        """
        Mode is read from config['mode'] (set in YAML or config dict):
          mode: auto   — auto-detect settings from data before running folds.
          mode: manual — use YAML config exactly (default).
        A temporary instance is used to resolve the config, then each fold
        constructs its own model_inst with the resolved settings.
        """
        # Resolve config once before the fold loop — no temp instance needed.
        # _resolve_config only mutates self.config, so we read the resolved dict
        # and pass it directly to each fold's model_inst via cls(config=...).
        # This replaces the old pattern of storing state on the class (cls._kfold_resolved_cfg).
        # db_tag is inserted into all output filenames when db_stem is provided
        # e.g. db_stem="psr_trainset_elisa_ngs" → "_psr_trainset_elisa_ngs"
        _db_tag  = f"_{db_stem}" if db_stem else ""

        y_arr    = np.asarray(y, dtype=int)
        _cfg_holder = cls.__new__(cls)
        _cfg_holder.config = copy.deepcopy(cls._DEFAULT_CONFIG)
        import yaml as _yaml
        _cfg_path = "config/transformer_onehot.yaml"
        if os.path.exists(_cfg_path):
            with open(_cfg_path) as _f:
                _user = _yaml.safe_load(_f) or {}
            _deep_merge(_cfg_holder.config, _user)
        _cfg_holder._resolve_config(y_arr)
        resolved_cfg = _cfg_holder.config
        del _cfg_holder

        data = data.copy()

        # [FIX-1] BARCODE always comes from data.index — do not rely on X having a BARCODE column
        data['BARCODE'] = data.index.astype(str).tolist()

        y       = np.asarray(y, dtype=int)
        n       = len(y)
        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mean_fpr= np.linspace(0, 1, 100)

        # Build full dataset ONCE — reused per fold via Subset
        print(f"[kfold] Pre-encoding {n:,} sequences (once for all folds)...")
        _sl        = resolved_cfg.get('sequence_lengths', {})
        _max_vh    = _sl.get('max_vh_len',    135)
        _max_vl    = _sl.get('max_vl_len',    135)
        _max_hcdr3 = _sl.get('max_hcdr3_len',  25)
        _lseqs = [''] * n if embedding_lm == 'onehot_vh' else X['LSEQ'].values
        full_ds = AntibodyDataset(
            X['HSEQ'].values, _lseqs, X['CDR3'].values,
            y, data['BARCODE'].values,
            _max_vh, _max_vl, _max_hcdr3,
            vh_only=(embedding_lm == 'onehot_vh')
        )
        print("[kfold] Encoding complete.")

        # ── Splitter — identical fallback chain to transformer_lm.py ──────────
        # [FIX-7] Use kfold_actual so that the printed fold count always matches
        #         the iterator even when kfold is reduced due to too-few clusters.
        kfold_actual = kfold

        if cluster_col in data.columns:
            groups          = data[cluster_col].values
            n_unique_groups = len(np.unique(groups))

            if n_unique_groups == n:
                print(f"[kfold] WARNING: all-singleton clusters → StratifiedKFold")
                splitter   = StratifiedKFold(n_splits=kfold_actual, shuffle=True, random_state=42)
                split_iter = splitter.split(np.arange(n), y)
            elif n_unique_groups < kfold_actual:
                # [FIX-7] Mutate kfold_actual, not kfold, so the variable name is unambiguous
                kfold_actual = n_unique_groups
                print(f"[kfold] WARNING: only {n_unique_groups} clusters → "
                      f"reducing folds to {kfold_actual}")
                splitter   = StratifiedGroupKFold(n_splits=kfold_actual, shuffle=True, random_state=42)
                split_iter = splitter.split(np.arange(n), y, groups)
            else:
                splitter   = StratifiedGroupKFold(n_splits=kfold_actual, shuffle=True, random_state=42)
                split_iter = splitter.split(np.arange(n), y, groups)
                print(f"[kfold] StratifiedGroupKFold on '{cluster_col}' "
                      f"({n_unique_groups} clusters, {kfold_actual} folds)")
        else:
            print(f"[kfold] WARNING: '{cluster_col}' not found → StratifiedKFold")
            splitter   = StratifiedKFold(n_splits=kfold_actual, shuffle=True, random_state=42)
            split_iter = splitter.split(np.arange(n), y)

        # ── State tracking ────────────────────────────────────────────────────
        tprs, aucs_list, fold_metrics = [], [], []
        best_fold_auc   = -1.0
        best_fold_num   = -1
        best_fold_state = None
        best_fold_cfg   = None
        all_records     = []

        os.makedirs(MODEL_DIR, exist_ok=True)
        plt.figure(figsize=(8, 7))
        print(f"\n[kfold] {kfold_actual}-fold CV | {title} | target={target.upper()}")

        for fold, (tr_idx, va_idx) in enumerate(split_iter, 1):
            print(f"\n── Fold {fold}/{kfold_actual} ──")

            # ── Leakage check ─────────────────────────────────────────────────
            if cluster_col in data.columns:
                groups_arr   = data[cluster_col].values
                train_groups = set(groups_arr[tr_idx])
                val_groups   = set(groups_arr[va_idx])
                leaked       = train_groups & val_groups
                if leaked:
                    print(f"  [WARN] {len(leaked)} cluster(s) leaked — "
                          f"check clustering threshold")
                else:
                    print(f"  [OK]  No CDR3 leakage | "
                          f"train_clusters={len(train_groups)}  "
                          f"val_clusters={len(val_groups)}")

            y_train = y[tr_idx]
            y_val   = y[va_idx]
            print(f"  Train={len(tr_idx):,} pos={y_train.mean():.1%}  "
                  f"Val={len(va_idx):,} pos={y_val.mean():.1%}")

            # [FIX-2] Construct via cls() so __init__ always runs correctly with
            #         its default config_path argument.
            #         Previously used cls.__new__(cls) + manual __init__() call
            #         which breaks if __init__ signature changes.
            # Pass auto-detected config when mode=="auto"
            # Use the resolved config (auto or manual) for every fold
            model_inst = cls(config=resolved_cfg)
            model_inst.device = device
            model_inst.set_lm_mode(embedding_lm)   # VH+VL or VH-only — must be before _build_model()

            tcfg       = model_inst.config['training']
            bs         = tcfg.get('batch_size', 32)
            lr         = tcfg.get('lr',          1e-4)
            wd         = tcfg.get('weight_decay', 1e-5)
            max_epochs = tcfg.get('epochs',       20)
            patience   = tcfg.get('patience',      7)

            pin = (str(device) != 'cpu')
            tr_loader = DataLoader(Subset(full_ds, tr_idx), batch_size=bs,
                                   shuffle=True,  num_workers=0, pin_memory=pin)
            va_loader = DataLoader(Subset(full_ds, va_idx), batch_size=bs,
                                   shuffle=False, num_workers=0, pin_memory=pin)

            model_inst.model = model_inst._build_model()
            criterion        = model_inst._build_criterion(y_train)
            optimizer        = torch.optim.AdamW(
                model_inst.model.parameters(), lr=lr, weight_decay=wd
            )

            # [FIX-3] ReduceLROnPlateau — adapts to early stopping correctly.
            #   CosineAnnealingLR misbehaves when training stops early.
            #   Scheduler patience read from config (YAML: scheduler.patience).
            #   Use patience=5 for label_smooth (probs take several epochs to
            #   spread from 0.5 — patience=3 halved lr before convergence).
            #   Use patience=3 for ce/weighted_ce (faster convergence signal).
            sch_cfg_k      = model_inst.config.get('scheduler', {})
            sch_factor_k   = sch_cfg_k.get('factor',   0.5)
            sch_patience_k = sch_cfg_k.get('patience', 3)
            sch_mode_k     = sch_cfg_k.get('mode',     'min')  # min = step on v_loss
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=sch_mode_k,
                factor=sch_factor_k, patience=sch_patience_k
            )

            # Early stopping on val_AUC — not val_loss.
            # val_loss is misleading for imbalanced data: predicting all-majority
            # gives low loss but AUC=0.5.  AUC is threshold-independent.
            best_val_auc = -1.0
            best_epoch   = 1     # 1-indexed epoch with highest val_AUC this fold
            # [FIX-4] Initialise best_state to None; guard load_state_dict call below.
            best_state   = None
            patience_ctr = 0

            history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

            for epoch in range(max_epochs):
                # ── train ─────────────────────────────────────────────────────
                model_inst.model.train()
                t_loss, t_correct, t_total = 0.0, 0, 0
                for enc, cdr3_enc, lbl, *_ in tr_loader:
                    enc, cdr3_enc, lbl = (enc.to(device), cdr3_enc.to(device),
                                           lbl.to(device))
                    optimizer.zero_grad()
                    out  = model_inst.model(enc, cdr3_enc)
                    loss = criterion(out, lbl)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model_inst.model.parameters(), max_norm=1.0
                    )
                    optimizer.step()
                    t_loss    += loss.item()
                    t_correct += (out.argmax(1) == lbl).sum().item()
                    t_total   += lbl.size(0)

                avg_t_loss = t_loss / len(tr_loader)
                t_acc      = t_correct / t_total

                # ── validate ──────────────────────────────────────────────────
                model_inst.model.eval()
                v_loss, v_probs, v_preds, v_true = 0.0, [], [], []
                with torch.no_grad():
                    for enc, cdr3_enc, lbl, *_ in va_loader:
                        enc, cdr3_enc, lbl = (enc.to(device), cdr3_enc.to(device),
                                               lbl.to(device))
                        out    = model_inst.model(enc, cdr3_enc)
                        v_loss += criterion(out, lbl).item()
                        prob   = torch.softmax(out, 1)[:, 1].cpu().numpy()
                        v_probs.extend(prob)
                        v_preds.extend(out.argmax(1).cpu().numpy())
                        v_true.extend(lbl.cpu().numpy())

                avg_v_loss = v_loss / len(va_loader)
                v_acc      = accuracy_score(v_true, v_preds)
                try:
                    v_auc = (roc_auc_score(v_true, v_probs)
                             if len(set(v_true)) > 1 else 0.5)
                except Exception:
                    v_auc = 0.5

                history['train_loss'].append(avg_t_loss)
                history['val_loss'].append(avg_v_loss)
                history['val_auc'].append(v_auc)

                # Step scheduler on val_LOSS (smooth signal) — matches transformer_lm
                # Early stopping still uses val_AUC (correct for model selection)
                scheduler.step(v_loss)

                print(f"  Ep {epoch+1:3d}/{max_epochs}"
                      f"  t_loss={avg_t_loss:.4f}  t_acc={t_acc:.4f}"
                      f"  v_loss={avg_v_loss:.4f}  v_acc={v_acc:.4f}"
                      f"  v_auc={v_auc:.4f}"
                      f"  lr={optimizer.param_groups[0]['lr']:.2e}")

                if v_auc > best_val_auc:
                    best_val_auc  = v_auc
                    best_epoch    = epoch + 1   # 1-indexed — used for mean_best_epoch report
                    best_state    = copy.deepcopy(model_inst.model.state_dict())
                    patience_ctr  = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        print(f"  → Early stop ep {epoch+1}  "
                              f"best_epoch={best_epoch}  best_auc={best_val_auc:.4f}")
                        break

            # Save per-fold loss + AUC curve
            _save_fold_loss_plot(history, fold, kfold_actual, target, embedding_lm, db_stem=db_stem)

            # [FIX-4] Guard: only restore if best_state was actually captured.
            #   If every epoch had v_auc=0.5 (degenerate fold), best_state stays
            #   None and we fall through to evaluation with the last trained weights.
            if best_state is not None:
                model_inst.model.load_state_dict(best_state)
            else:
                print(f"  [WARN] Fold {fold}: best_state is None — "
                      f"using final epoch weights (check class distribution).")

            # ── Final evaluation ──────────────────────────────────────────────
            model_inst.model.eval()
            probs, preds, trues, bcs = [], [], [], []
            with torch.no_grad():
                for enc, cdr3_enc, lbl, bc, *_ in va_loader:
                    enc, cdr3_enc = enc.to(device), cdr3_enc.to(device)
                    out   = model_inst.model(enc, cdr3_enc)
                    prob  = torch.softmax(out, 1)[:, 1].cpu().numpy()
                    probs.extend(prob)
                    preds.extend(out.argmax(1).cpu().numpy())
                    trues.extend(lbl.numpy())
                    bcs.extend(bc)

            if len(set(trues)) < 2:
                print(f"  Skipping fold {fold} — only one class in val.")
                continue

            fold_auc  = roc_auc_score(trues, probs)
            fold_acc  = accuracy_score(trues, preds)
            fold_f1   = f1_score(trues,       preds, zero_division=0)
            fold_prec = precision_score(trues, preds, zero_division=0)
            fold_rec  = recall_score(trues,   preds, zero_division=0)

            print(f"  Fold {fold} → AUC={fold_auc:.4f}  Acc={fold_acc:.4f}"
                  f"  F1={fold_f1:.4f}  Prec={fold_prec:.4f}  Rec={fold_rec:.4f}")

            fold_metrics.append({
                'fold':       fold,
                'auc':        fold_auc,
                'acc':        fold_acc,
                'f1':         fold_f1,
                'precision':  fold_prec,
                'recall':     fold_rec,
                'best_epoch': best_epoch,   # epoch with highest val_AUC this fold
            })
            aucs_list.append(fold_auc)

            fpr, tpr, _ = roc_curve(trues, probs)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            plt.plot(fpr, tpr, alpha=0.3, lw=1,
                     label=f'Fold {fold} (AUC={fold_auc:.3f})')

            for bc, true, pred, prob in zip(bcs, trues, preds, probs):
                all_records.append({'BARCODE': bc, 'fold': fold,
                                    'true': true, 'pred': pred, 'prob': prob})

            # Save individual fold checkpoint
            torch.save({
                'state_dict': model_inst.model.state_dict(),
                'config':     model_inst.config,
                'fold':       fold,
                'fold_auc':   fold_auc,
            }, os.path.join(MODEL_DIR,
                f"TransformerOneHot_{target}_{embedding_lm}"
                f"{_db_tag}_fold{fold}_k{kfold_actual}.pt"))

            # Track best fold
            if fold_auc > best_fold_auc:
                best_fold_auc   = fold_auc
                best_fold_num   = fold
                best_fold_state = copy.deepcopy(model_inst.model.state_dict())
                best_fold_cfg   = copy.deepcopy(model_inst.config)

        if not aucs_list:
            print("[kfold] No valid folds — check class distribution.")
            return None, None, None, None, None, None

        # ── Save best-fold model ───────────────────────────────────────────────
        if best_fold_state is not None:
            best_path = os.path.join(
                MODEL_DIR,
                f"BEST_{target}_{embedding_lm}_transformer_onehot"
                f"{_db_tag}_k{kfold_actual}_fold{best_fold_num}.pt"
            )
            torch.save({
                'state_dict': best_fold_state,
                'config':     best_fold_cfg,
                'fold':       best_fold_num,
                'fold_auc':   best_fold_auc,
                'kfold':      kfold_actual,
                'target':     target,
            }, best_path)
            print(f"\n[kfold] Best fold → {best_path}"
                  f"  (fold={best_fold_num}, AUC={best_fold_auc:.4f})")

        # Save fold predictions CSV
        if all_records:
            pred_path = os.path.join(
                MODEL_DIR,
                f"fold_preds_{target}_{embedding_lm}"
                f"_transformer_onehot{_db_tag}_k{kfold_actual}.csv"
            )
            df_preds = pd.DataFrame(all_records)
            df_preds['best_fold'] = (df_preds['fold'] == best_fold_num).astype(int)
            df_preds.to_csv(pred_path, index=False)
            print(f"[kfold] Fold predictions → {pred_path}")

        # Threshold optimisation — handled below by run_full_threshold_pipeline()

        # ── Aggregate ROC plot ────────────────────────────────────────────────
        mean_auc  = float(np.mean(aucs_list))
        std_auc   = float(np.std(aucs_list))
        mean_acc  = float(np.mean([m['acc']       for m in fold_metrics]))
        mean_f1   = float(np.mean([m['f1']        for m in fold_metrics]))
        mean_prec = float(np.mean([m['precision'] for m in fold_metrics]))
        mean_rec  = float(np.mean([m['recall']    for m in fold_metrics]))

        mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
        std_tpr  = np.std(tprs,  axis=0)

        plt.plot(mean_fpr, mean_tpr, 'b', lw=3,
                 label=f'Mean ROC (AUC={mean_auc:.3f}±{std_auc:.3f})')
        plt.fill_between(
            mean_fpr,
            np.maximum(mean_tpr - std_tpr, 0),
            np.minimum(mean_tpr + std_tpr, 1),
            color='lightblue', alpha=0.3, label='±1 std',
        )
        plt.plot([0, 1], [0, 1], '--', color='gray', lw=0.8)
        plt.xlim([0, 1]); plt.ylim([0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(
            f'{title} — {target.upper()}\n{kfold_actual}-Fold SGKF ROC\n'
            f'Acc={mean_acc:.3f}  F1={mean_f1:.3f}  '
            f'Prec={mean_prec:.3f}  Rec={mean_rec:.3f}',
            fontsize=9,
        )
        plt.legend(loc='lower right', fontsize=7)
        plt.grid(alpha=0.3); plt.tight_layout()

        plot_path = os.path.join(
            MODEL_DIR,
            f"CV_ROC_{target}_{embedding_lm}"
            f"_transformer_onehot{_db_tag}_k{kfold_actual}.png"
        )
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        # ── Mean best epoch — the key number for full-data train() ─────────
        best_epochs    = [m['best_epoch'] for m in fold_metrics]
        mean_best_ep   = int(round(float(np.mean(best_epochs))))
        std_best_ep    = float(np.std(best_epochs))

        sep = '═' * 62
        print(f"\n{sep}")
        print(f"  {kfold_actual}-FOLD CV RESULTS — {target.upper()}")
        print(f"{'─'*62}")
        print(f"  {'Fold':>5}  {'AUC':>7}  {'Acc':>7}  {'F1':>7}  "
              f"{'Prec':>7}  {'Rec':>7}  {'BestEp':>7}")
        print(f"  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
        for m in fold_metrics:
            marker = " ←" if m['fold'] == best_fold_num else ""
            print(f"  {m['fold']:5d}  {m['auc']:7.4f}  {m['acc']:7.4f}  "
                  f"{m['f1']:7.4f}  {m['precision']:7.4f}  {m['recall']:7.4f}  "
                  f"{m['best_epoch']:7d}{marker}")
        print(f"{'─'*62}")
        print(f"  {'Mean':>5}  {mean_auc:7.4f}  {mean_acc:7.4f}  "
              f"{mean_f1:7.4f}  {mean_prec:7.4f}  {mean_rec:7.4f}  "
              f"{mean_best_ep:7d}")
        print(f"  {'±Std':>5}  {std_auc:7.4f}  {'':7}  {'':7}  {'':7}  {'':7}  "
              f"{std_best_ep:7.1f}")
        print(f"{'─'*62}")
        print(f"  Best fold : {best_fold_num}  (AUC={best_fold_auc:.4f})")
        print(f"  ROC plot  : {plot_path}")
        print(f"{sep}")
        print(f"")
        print(f"  ── FOR FINAL train() ON FULL DATASET ─────────────────────")
        print(f"  Mean best epoch across folds : {mean_best_ep}  (±{std_best_ep:.1f})")
        print(f"  → Set in your YAML before running train():")
        print(f"      training:")
        print(f"        epochs: {mean_best_ep}")
        print(f"  This gives the full-data model the same number of gradient")
        print(f"  updates as the average fold — no early stopping needed.")
        print(f"{sep}\n")

        # ── Threshold summary ─────────────────────────────────────────────────
        thresh_result = None   # populated below by run_full_threshold_pipeline if available
        if thresh_result is not None:
            t_rec = thresh_result['recommended']
            m_rec = thresh_result['metrics_at_recommended']
            print(f"{'═'*62}")
            print(f"  THRESHOLD OPTIMISATION  (pooled OOF, n={len(oof_true):,})")
            print(f"{'─'*62}")
            print(f"  Method           Threshold")
            print(f"  {'─'*16}  {'─'*9}")
            print(f"  Youden J         {thresh_result['youden']:.4f}  ← recommended (equal FP/FN cost)")
            print(f"  F1 optimum       {thresh_result['f1']:.4f}")
            print(f"  Cost-sensitive   {thresh_result['cost']:.4f}  (cost_fp=1.0  cost_fn=1.0)")
            print(f"  PR balanced      {thresh_result['pr_balanced']:.4f}")
            print(f"{'─'*62}")
            print(f"  At recommended threshold = {t_rec:.4f}:")
            print(f"    Accuracy   : {m_rec['accuracy']:.4f}")
            print(f"    F1         : {m_rec['f1']:.4f}")
            print(f"    Precision  : {m_rec['precision']:.4f}")
            print(f"    Recall     : {m_rec['recall']:.4f}")
            print(f"    TPR        : {m_rec['tpr']:.4f}  (sensitivity)")
            print(f"    FPR        : {m_rec['fpr']:.4f}  (1 - specificity)")
            print(f"{'─'*62}")
            print(f"  To use in predict():")
            print(f"    model.predict(X, threshold={t_rec:.4f})")
            print(f"  If missing bad antibodies is costly (e.g. cost_fn=3):") 
            print(f"    re-run with cost_fn=3.0 in run_full_threshold_pipeline()")
            print(f"{'═'*62}\n")

        # ── Auto threshold optimisation ──────────────────────────────────────
        # Mirrors the identical block in transformer_lm.py.
        # Runs automatically — no changes needed in predict_developability.py.
        # Requires:
        #   • save_fold_preds=True  (default — all_records must be non-empty)
        #   • utils/threshold_optimizer.py  (soft dependency — skipped if absent)
        #
        # Produces (in MODEL_DIR):
        #   fold_preds_{target}_{lm}_transformer_onehot_k{N}.csv  ← already written above
        #   thresh_report_{target}_{lm}.png        4-panel diagnostic plot
        #   thresh_report_{target}_{lm}.json       all methods + threshold values
        #   thresh_stability_{target}_{lm}_auto.png per-fold consistency
        #   BEST_*.pt updated in-place              recommended_threshold embedded
        if _THRESHOLD_OPT_AVAILABLE and all_records:
            fold_preds_csv = os.path.join(
                MODEL_DIR,
                f"fold_preds_{target}_{embedding_lm}"
                f"_transformer_onehot{_db_tag}_k{kfold_actual}.csv",
            )
            # best_fold_state path written above — reconstruct to pass to embedder
            best_ckpt_path = os.path.join(
                MODEL_DIR,
                f"BEST_{target}_{embedding_lm}_transformer_onehot"
                f"{_db_tag}_k{kfold_actual}_fold{best_fold_num}.pt",
            ) if best_fold_state is not None else None

            print(f"\n[threshold] Starting optimisation  ···")
            try:
                stability = run_full_threshold_pipeline(
                    fold_preds_csv = fold_preds_csv,
                    target         = target,
                    lm             = embedding_lm,
                    best_ckpt_path = best_ckpt_path,
                    output_dir     = MODEL_DIR,
                    cost_fp        = 1.0,
                    cost_fn        = 3.0,   # FN costs 3× FP — missing a bad mAb is expensive
                )
                rec_thresh = stability['pooled_threshold']

                # ── Store on class instance so save() can embed it ────────────
                # Mirrors transformer_lm.py: the threshold is embedded into
                # BEST_*.pt by run_full_threshold_pipeline() itself (in-place),
                # AND stored here so that calling code can do:
                #
                #   model = TransformerOneHotModel()
                #   model.kfold_validation(...)       # sets model.recommended_threshold
                #   model.train(X_full, y_full, epochs=mean_best_ep)
                #   model.save("FINAL_*.pt")          # save() embeds the threshold
                #
                # predict_developability.py then reads it via:
                #   payload.get('recommended_threshold', 0.5)
                # with zero changes to the calling script.
                cls.recommended_threshold = float(rec_thresh)   # class-level so kfold (classmethod) can set it

                print(f"\n  ── RECOMMENDED THRESHOLD ─────────────────────────────")
                print(f"  Pooled OOF threshold : {rec_thresh:.4f}"
                      f"  (method=auto, stability±{stability['std_threshold']:.3f})")
                print(f"  Embedded into BEST_* : {best_ckpt_path or 'N/A'}")
                print(f"  Stored on class      : model.recommended_threshold = {rec_thresh:.4f}")
                print(f"  After full-data train(), call model.save() to carry threshold into FINAL_*.pt")
                print(f"  Use in predict_developability.py:")
                print(f"    scores = model.predict_proba(X)")
                print(f"    labels = (scores >= {rec_thresh:.4f}).astype(int)")
                print(f"  {'═'*62}")
            except Exception as _e:
                print(f"[threshold] WARNING: optimisation failed — {_e}")
                print(f"[threshold] kfold results are unaffected. "
                      f"Threshold defaults to 0.5.")
        elif not _THRESHOLD_OPT_AVAILABLE:
            print(
                "\n[threshold] utils/threshold_optimizer.py not found — "
                "skipping.\n"
                "  Predictions will use threshold=0.5 until you run "
                "threshold_optimizer manually."
            )

    # ── predict ───────────────────────────────────────────────────────────────

    def predict_proba(self, X):
        """
        Predict positive-class probabilities.

        [FIX-6] Accepts either:
          • pd.DataFrame with HSEQ/LSEQ/CDR3 columns (encodes on the fly)
          • np.ndarray of shape (n, encoding_dim) — pre-encoded, skips re-encoding
            Use this path when calling repeatedly on the same dataset.
        """
        self.model.eval()

        if isinstance(X, np.ndarray):
            # Fast path: already encoded — wrap directly in a minimal Dataset
            class _ArrayDS(Dataset):
                def __init__(self, enc, cdr3):
                    self.enc  = torch.from_numpy(enc)
                    self.cdr3 = torch.from_numpy(cdr3)
                def __len__(self): return len(self.enc)
                def __getitem__(self, i): return self.enc[i], self.cdr3[i]

            # Split at self.encoding_dim — correct for both onehot (5400) and onehot_vh (2700)
            hl_dim   = self.encoding_dim                                       # 5400 or 2700
            b1_len   = self.max_heavy_len if self._vh_only() else (self.max_heavy_len + self.max_light_len)
            c_dim    = self.max_hcdr3_len * len(AMINO_ACIDS)
            enc_2d   = X[:, :hl_dim].reshape(-1, b1_len, len(AMINO_ACIDS))
            cdr3_2d  = X[:, hl_dim:hl_dim + c_dim].reshape(-1, self.max_hcdr3_len, len(AMINO_ACIDS))
            ds      = _ArrayDS(enc_2d.astype(np.float32), cdr3_2d.astype(np.float32))
            loader  = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
            probs   = []
            with torch.no_grad():
                for enc, cdr3_enc in loader:
                    out = self.model(enc.to(self.device), cdr3_enc.to(self.device))
                    probs.extend(torch.softmax(out, 1)[:, 1].cpu().numpy())
            return np.array(probs)

        # Standard path: sequence DataFrame
        heavy   = X['HSEQ'].values if hasattr(X, 'columns') and 'HSEQ' in X.columns else X
        light   = self._get_light_seqs(X, len(heavy))
        hcdr3   = X['CDR3'].values  if hasattr(X, 'columns') and 'CDR3'  in X.columns else [''] * len(heavy)
        dataset = AntibodyDataset(heavy, light, hcdr3, [0]*len(heavy), ['tmp']*len(heavy),
                                  self.max_heavy_len, self.max_light_len, self.max_hcdr3_len,
                                  vh_only=self._vh_only())
        loader  = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
        probs   = []
        with torch.no_grad():
            for enc, cdr3_enc, *_ in loader:
                out = self.model(enc.to(self.device), cdr3_enc.to(self.device))
                probs.extend(torch.softmax(out, 1)[:, 1].cpu().numpy())
        return np.array(probs)

    def predict(self, X, threshold: float = None) -> np.ndarray:
        """
        Classify antibodies using a probability threshold.

        threshold : float or None
            If None, uses self.recommended_threshold (set by load() from checkpoint).
            Falls back to 0.5 if no threshold was embedded in the checkpoint.
            Pass an explicit float to override.

        To find the optimal threshold: run kfold_validation() first.
        The recommended threshold is printed at the end and embedded
        into the BEST_*.pt checkpoint automatically.
        Then load() reads it and sets self.recommended_threshold.

        Example:
            model = TransformerOneHotModel.load('BEST_psr_onehot_k9_fold3.pt')
            preds = model.predict(X)                # uses optimal threshold
            preds = model.predict(X, threshold=0.4) # override
        """
        t = threshold if threshold is not None else getattr(self, 'recommended_threshold', 0.5)
        if threshold is None:
            print(f"[predict] using threshold={t:.4f}"
                  + (" (from checkpoint)" if hasattr(self, 'recommended_threshold') else " (default 0.5)"))
        return (self.predict_proba(X) >= t).astype(int)

    # ── save / load ───────────────────────────────────────────────────────────

    def save(self, path):
        """
        Save model weights, config, and recommended_threshold into a single .pt file.

        recommended_threshold is embedded automatically if it was set by either:
          • kfold_validation()  — set on the class instance after threshold pipeline
          • manual assignment   — model.recommended_threshold = 0.42

        At load() time, predict_developability.py reads it via:
            payload.get('recommended_threshold', 0.5)
        so no changes are needed in the calling script.

        If recommended_threshold is not set (e.g. train() called without prior kfold),
        it is saved as None and load() will fall back to 0.5.
        """
        save_dir = os.path.dirname(path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        rec_thresh = getattr(self, 'recommended_threshold', None)
        torch.save({
            'state_dict':            self.model.state_dict(),
            'config':                self.config,
            'recommended_threshold': rec_thresh,   # None → load() falls back to 0.5
            'lm_mode':               getattr(self, 'lm_mode', 'onehot'),
        }, path)
        thresh_note = f"  threshold={rec_thresh:.4f}" if rec_thresh is not None else "  threshold=None (will use 0.5 at predict time)"
        print(f"[save] → {path}{thresh_note}")

    @classmethod
    def load(cls, path, config_path="config/transformer_onehot.yaml"):
        instance = cls(config_path)
        payload  = torch.load(path, map_location=instance.device, weights_only=False)
        if isinstance(payload, dict) and 'state_dict' in payload:
            state_dict = payload['state_dict']
            instance.config = _deep_merge(instance.config, payload.get('config', {}))
        else:
            state_dict = payload   # raw state_dict (old format)
        # Restore lm_mode BEFORE _build_model so DevelopabilityClassifier
        # receives the correct branch1_len (135 for onehot_vh, 270 for onehot)
        lm_mode = payload.get('lm_mode', 'onehot') if isinstance(payload, dict) else 'onehot'
        instance.set_lm_mode(lm_mode)
        instance.model = instance._build_model()
        instance.model.load_state_dict(state_dict)
        instance.model.eval()
        # Read recommended_threshold — matches transformer_lm.py pattern exactly.
        # Pipeline embeds it into BEST_*.pt; save() embeds it into FINAL_*.pt.
        # Falls back to 0.5 if absent (old checkpoints or train()-only models).
        if isinstance(payload, dict):
            rt = payload.get('recommended_threshold', None)
            instance.recommended_threshold = float(rt) if rt is not None else 0.5
            if rt is not None and rt != 0.5:
                print(f"[load] recommended_threshold={rt:.4f}  (embedded by kfold threshold pipeline)")
            else:
                print(f"[load] recommended_threshold=0.5  (default — no threshold in checkpoint)")
        print(f"[load] ← {path}")
        return instance

    # ── single-antibody inference ─────────────────────────────────────────────

    def predict_single(self, barcode, VH, VL, HCDR3):
        """Predict probability for a single antibody. VL is ignored in onehot_vh mode."""
        enc_h = torch.from_numpy(one_hot_encode_sequence_2d(VH,    self.max_heavy_len)).float()
        enc_c = torch.from_numpy(one_hot_encode_sequence_2d(HCDR3, self.max_hcdr3_len)).float()
        if self._vh_only():
            enc_hl = enc_h.unsqueeze(0).to(self.device)          # (1, 135, 20)
        else:
            _VL   = VL or ''
            enc_l = torch.from_numpy(one_hot_encode_sequence_2d(_VL, self.max_light_len)).float()
            enc_hl = torch.cat([enc_h, enc_l], dim=0).unsqueeze(0).to(self.device)  # (1, 270, 20)
        enc_c = enc_c.unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.softmax(self.model(enc_hl, enc_c), 1)[0, 1].item()

    # ── CDR3 mutation heatmap ─────────────────────────────────────────────────

    def plot_cdr3_mutation_heatmap(self, vh_seq, vl_seq, hcdr3, barcode, label,
                                   target_analysis="PSR",
                                   output_path="SEC/image/sec_cdr3_mutation_heatmap.png"):
        prob       = self.predict_single(barcode, vh_seq, vl_seq, hcdr3)
        cdr3_start = vh_seq.find(hcdr3)
        mut_dict   = self._generate_cdr3_mutations(hcdr3)

        for pos in mut_dict:
            for aa in mut_dict[pos]:
                m_cdr3 = hcdr3[:pos - 1] + aa + hcdr3[pos:]
                VH_mut = (vh_seq[:cdr3_start] + m_cdr3 + vh_seq[cdr3_start + len(hcdr3):]
                          if cdr3_start >= 0 else vh_seq)
                mut_dict[pos][aa] = self.predict_single(barcode, VH_mut, vl_seq, m_cdr3)

        hmap, aas, positions = self._create_heatmap_data(mut_dict, hcdr3)
        plt.figure(figsize=(10, 6))
        sns.heatmap(hmap, annot=True, fmt=".3f", cmap="YlOrRd",
                    xticklabels=positions, yticklabels=list(aas))
        plt.xlabel("Original CDR3 AA"); plt.ylabel("Mutant AA")
        plt.title(f"{target_analysis}: CDR3 mutation probs\n"
                  f"ID={barcode}  label={label}  prob={prob:.3f}")
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path); plt.show(); plt.close()

    def _generate_cdr3_mutations(self, cdr3):
        return {i + 1: {aa: None for aa in AMINO_ACIDS} for i, _ in enumerate(cdr3)}

    def _create_heatmap_data(self, mutation_dict, original_cdr3):
        data = np.zeros((len(AMINO_ACIDS), len(original_cdr3)))
        for pos in range(1, len(original_cdr3) + 1):
            for i, aa in enumerate(AMINO_ACIDS):
                if pos in mutation_dict and aa in mutation_dict[pos]:
                    data[i, pos - 1] = mutation_dict[pos][aa]
        return data, AMINO_ACIDS, list(original_cdr3)

    # ── Integrated Gradients ──────────────────────────────────────────────────

    def integrated_gradients_single(self, vh_seq, vl_seq, hcdr3_seq, n_steps=None):
        """
        Compute IG attributions for one antibody.
        vl_seq is ignored when lm_mode='onehot_vh'.
        Returns 1D array of length (branch1_len + cdr3_len).
        """
        if n_steps is None:
            n_steps = self.config.get('interpretability', {}).get('ig_n_steps', 50)
        self.model.eval()
        enc_h = one_hot_encode_sequence_2d(vh_seq, self.max_heavy_len)
        if self._vh_only():
            enc_hl = torch.from_numpy(enc_h).float().unsqueeze(0).to(self.device)
        else:
            enc_l  = one_hot_encode_sequence_2d(vl_seq or '', self.max_light_len)
            enc_hl = torch.from_numpy(
                np.concatenate([enc_h, enc_l], axis=0)
            ).float().unsqueeze(0).to(self.device)
        enc_c = torch.from_numpy(
            one_hot_encode_sequence_2d(hcdr3_seq, self.max_hcdr3_len)
        ).float().unsqueeze(0).to(self.device)
        attr = IntegratedGradients(self.model).attribute(
            (enc_hl, enc_c),
            baselines=(torch.zeros_like(enc_hl), torch.zeros_like(enc_c)),
            target=1, n_steps=n_steps
        )
        return np.concatenate([
            attr[0].squeeze(0).cpu().numpy().sum(axis=1),
            attr[1].squeeze(0).cpu().numpy().sum(axis=1),
        ])

    def global_ig_analysis(self, dataset, output_prefix="global_ig"):
        print("\n[IG] Running global analysis (target=class 1 = developable)...")
        self.model.eval()

        ig_cfg = self.config.get('interpretability', {})
        n_steps = ig_cfg.get('ig_n_steps', 50)
        top_feat = ig_cfg.get('top_features', 60)

        ig = IntegratedGradients(self.model)

        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        all_enc, all_cdr3 = [], []

        for enc, cdr3_enc, *_ in loader:
            enc = enc.to(self.device)
            cdr3_enc = cdr3_enc.to(self.device)

            # Attribute to positive class (class=1 = pass/developable)
            attr = ig.attribute(
                (enc, cdr3_enc),
                baselines=(torch.zeros_like(enc), torch.zeros_like(cdr3_enc)),
                target=1,                    # ← Fixed: target=1 for positive class
                n_steps=n_steps
            )

            all_enc.append(attr[0].sum(dim=-1).detach().cpu().numpy())   # sum over AA dim
            all_cdr3.append(attr[1].sum(dim=-1).detach().cpu().numpy())

        attr_enc = np.concatenate(all_enc, axis=0)
        attr_cdr3 = np.concatenate(all_cdr3, axis=0)

        hcdr3_seqs = [dataset[i][6] for i in range(len(dataset))]
        self._plot_global_importance_2d(attr_enc, attr_cdr3, output_prefix, top_feat)
        self._plot_hcdr3_residue_heatmap_2d(attr_cdr3, hcdr3_seqs, output_prefix)
        print(f"[IG] Complete. Prefix: {output_prefix}")

    def _plot_global_importance_2d(self, attr_encoding, attr_cdr3, prefix, top_n=60):
        # [FIX-YAML] top_n passed from global_ig_analysis (read from config.interpretability.top_features)
        mean_enc  = np.mean(np.abs(attr_encoding), axis=0)
        mean_cdr3 = np.mean(np.abs(attr_cdr3),     axis=0)
        all_imp   = np.concatenate([mean_enc, mean_cdr3])
        top_idx   = np.argsort(all_imp)[::-1][:top_n]
        if self._vh_only():
            all_labels = (
                [f"H_{i+1}" for i in range(self.max_heavy_len)] +
                [f"CDR3_{i+1}" for i in range(self.max_hcdr3_len)]
            )
            legend_label = 'VH'
        else:
            all_labels = (
                [f"H_{i+1}" for i in range(self.max_heavy_len)] +
                [f"L_{i+1}" for i in range(self.max_light_len)] +
                [f"CDR3_{i+1}" for i in range(self.max_hcdr3_len)]
            )
            legend_label = 'VH/VL'
        colors = ['steelblue' if i < len(mean_enc) else 'darkorange' for i in top_idx]
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_idx)), all_imp[top_idx], color=colors)
        plt.yticks(range(len(top_idx)), [all_labels[i] for i in top_idx], fontsize=8)
        plt.xlabel("Mean |IG| Attribution")
        plt.title(f"Top {top_n} Position Importance — {prefix}")
        plt.legend(handles=[Patch(facecolor='steelblue', label=legend_label),
                             Patch(facecolor='darkorange', label='CDR3')])
        plt.tight_layout()
        out = os.path.join(MODEL_DIR, f"{prefix}_position_importance.png")
        plt.savefig(out, dpi=300); plt.close()
        print(f"[IG] → {out}")

    def _plot_hcdr3_residue_heatmap_2d(self, attr_cdr3, hcdr3_seqs, prefix):
        max_len    = max((len(s) for s in hcdr3_seqs if s), default=1)
        hcdr3_data = np.zeros((1, max_len))
        n_valid    = 0
        for i, seq in enumerate(hcdr3_seqs):
            if seq and i < len(attr_cdr3):
                sl = min(len(seq), max_len, attr_cdr3.shape[1])
                hcdr3_data[0, :sl] += np.abs(attr_cdr3[i, :sl])
                n_valid += 1
        hcdr3_data /= max(1, n_valid)
        plt.figure(figsize=(10, 2))
        sns.heatmap(hcdr3_data, cmap='viridis',
                    xticklabels=range(1, max_len + 1), yticklabels=['CDR3'])
        plt.title("HCDR3 Residue Importance (IG)"); plt.xlabel("HCDR3 Position")
        plt.tight_layout()
        out = os.path.join(MODEL_DIR, f"{prefix}_hcdr3_residue_importance.png")
        plt.savefig(out, dpi=300); plt.close()
        print(f"[IG] → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MODULE-LEVEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (in-place)."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _save_fold_loss_plot(history: dict, fold: int, kfold: int,
                          target: str, lm: str, db_stem: str = ""):
    """
    Save a two-panel figure per fold:
      Panel 1: train_loss and val_loss vs epoch
      Panel 2: val_auc vs epoch with vertical line at best epoch
    File: loss_curve_{target}_{lm}_fold{N}_k{K}.png → MODEL_DIR
    """
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    ax1.plot(epochs, history['train_loss'], 'b-o', ms=3, label='Train loss')
    ax1.plot(epochs, history['val_loss'],   'r-o', ms=3, label='Val loss')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{target.upper()} | Fold {fold}/{kfold} | {lm}')
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    best_ep  = int(np.argmax(history['val_auc'])) + 1
    best_auc = max(history['val_auc'])
    ax2.plot(epochs, history['val_auc'], 'g-o', ms=3, label='Val AUC')
    ax2.axvline(x=best_ep, color='gray', linestyle='--', lw=1,
                label=f'Best ep={best_ep}  AUC={best_auc:.4f}')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Val AUC')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.tight_layout()
    _db_tag = f"_{db_stem}" if db_stem else ""
    out = os.path.join(MODEL_DIR, f"loss_curve_{target}_{lm}{_db_tag}_fold{fold}_k{kfold}.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [plot] → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  SPLIT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def train_test_split_group_stratified(
    data,
    X,
    y,
    test_size=0.2,
    random_state=42,
    group_col='HCDR3_CLUSTER_0.8',
    verbose=True,
):
    """
    Group-stratified train/test split — no CDR3 cluster leakage.
    Picks the fold whose test-set class balance is closest to the overall rate.
    """
    if group_col not in data.columns:
        raise ValueError(f"'{group_col}' not found. Available: {list(data.columns)}")
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be in (0,1), got {test_size}")

    groups   = data[group_col].values
    n_splits = max(2, round(1.0 / test_size))
    actual   = 1.0 / n_splits
    if abs(actual - test_size) > 0.05 and verbose:
        print(f"  [WARN] test_size={test_size:.2f} → n_splits={n_splits} → actual≈{actual:.2f}")

    sgkf        = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    overall_pos = np.mean(y)
    best_split, best_diff = None, float('inf')

    for tr, te in sgkf.split(np.arange(len(y)), y, groups):
        diff = abs(np.mean(y[te]) - overall_pos)
        if diff < best_diff:
            best_diff, best_split = diff, (tr, te)

    tr_idx, te_idx        = best_split
    X_train, X_test       = X.iloc[tr_idx], X.iloc[te_idx]
    y_train, y_test       = y[tr_idx],      y[te_idx]
    train_data, test_data = data.iloc[tr_idx], data.iloc[te_idx]

    if verbose:
        leaked = set(groups[tr_idx]) & set(groups[te_idx])
        status = (f"[WARN] {len(leaked)} cluster(s) leaked"
                  if leaked else "[OK]  No CDR3 leakage")
        print(f"  {status}")
        print(f"  Train={len(train_data):,} pos={y_train.mean():.1%}  "
              f"Test={len(test_data):,} pos={y_test.mean():.1%}  diff={best_diff:.3f}")

    return train_data, test_data, X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════════
# 8.  SAMPLE SIZE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_sample_size_effect(
    db_path:      str,
    target:       str = "sec_filter",
    start_size:   int = 1000,
    increment:    int = 5000,
    output_csv:   str = "sample_size_performance.csv",
    random_state: int = 42,
):
    print(f"[eval] Loading: {db_path}")
    df = pd.read_excel(db_path)
    df = df.dropna(subset=['BARCODE', 'HSEQ', 'LSEQ', 'CDR3', target])
    if 'antigen' in df.columns:
        df = df[~df['antigen'].str.contains('test', na=False, case=False)]
    df = df[pd.notna(df[target])]

    from utils.clustering import greedy_clustering_by_levenshtein
    df['HCDR3_CLUSTER_0.8'] = greedy_clustering_by_levenshtein(df['CDR3'].tolist(), 0.8)
    print(f"[eval] n={len(df):,}  pos={df[target].mean():.1%}")

    sample_sizes = list(range(start_size, len(df) + 1, increment))
    if sample_sizes and sample_sizes[-1] > len(df):
        sample_sizes[-1] = len(df)

    results = []
    for size in sample_sizes:
        print(f"\n=== n={size:,} ===")
        sampled = df.groupby(target, group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), size // 2 + size % 2), random_state=random_state)
        ).sample(frac=1, random_state=random_state).head(size)

        _, _, X_tr, X_te, y_tr, y_te = train_test_split_group_stratified(
            sampled, sampled[['HSEQ', 'LSEQ', 'CDR3']],
            sampled[target].values, random_state=random_state
        )
        model = TransformerOneHotModel()
        model.train(X_tr, y_tr, epochs=10)
        probs = model.predict_proba(X_te)
        preds = model.predict(X_te)
        r = {
            'sample_size': size, 'train_size': len(X_tr), 'test_size': len(X_te),
            'auc':       roc_auc_score(y_te, probs),
            'accuracy':  accuracy_score(y_te, preds),
            'f1_score':  f1_score(y_te, preds),
            'precision': precision_score(y_te, preds),
            'recall':    recall_score(y_te, preds),
        }
        results.append(r)
        print(f"  AUC={r['auc']:.4f}  Acc={r['accuracy']:.4f}  F1={r['f1_score']:.4f}")
        pd.DataFrame(results).to_csv(output_csv, index=False)

    rdf = pd.DataFrame(results)
    rdf.to_csv(output_csv, index=False)
    print(f"\n[eval] → {output_csv}")

    plt.figure(figsize=(10, 6))
    for col, mk, lb in [
        ('auc', 'o', 'AUC'), ('accuracy', 's', 'Accuracy'),
        ('f1_score', '^', 'F1'), ('precision', 'd', 'Precision'),
        ('recall', 'v', 'Recall'),
    ]:
        plt.plot(rdf['sample_size'], rdf[col], marker=mk, label=lb, linewidth=2)
    plt.xlabel('Sample Size'); plt.ylabel('Metric')
    plt.title(f'Sample Size Effect ({target})')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plot_path = output_csv.replace('.csv', '_plot.png')
    plt.savefig(plot_path, dpi=300); plt.close()
    print(f"[eval] plot → {plot_path}")


if __name__ == "__main__":
    evaluate_sample_size_effect(
        db_path="data/dataset1.xlsx",
        target="psr_filter",
        start_size=100,
        increment=200,
        output_csv="transform_onehot_dataset1_sample_size_performance_psr_filter_100_200.csv"
    )