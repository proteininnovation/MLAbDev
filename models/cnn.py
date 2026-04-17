# models/cnn.py
# 1D-CNN for PLM Embeddings (ablang, antiberty, antiberta2, antiberta2-cssp)
# IPI Antibody Developability Prediction Platform — Production Version DEC-2025
#
# ── Compatibility contract with predict_developability.py ──────────────────────
#
#  TRAIN path:
#    model = CNNModel()
#    model.train(X, y)                         ← X: pd.DataFrame, y: np.ndarray
#    model.save(path)                           ← FINAL_{target}_{lm}_cnn_{db_stem}.pt
#
#  PREDICT path:
#    model = CNNModel.load(model_path, embedding_dim=X.shape[1])
#    scores = model.predict_proba(X_input)      ← X_input: pd.DataFrame
#    labels = (scores >= threshold).astype(int)
#
#  KFOLD path:
#    CNNModel().kfold_validation(data, X, y,
#        embedding_lm=args.lm, title=title, kfold=args.kfold,
#        target=args.target)
#
#  MODEL FILE NAMING (handled by main script, not here):
#    FINAL_{target}_{lm}_cnn_{db_stem}.pt
#    BEST_{target}_{lm}_cnn_k{N}_fold{F}.pt
#
# ── Architecture ──────────────────────────────────────────────────────────────
#
#  Three residual blocks with dilated convolutions (dilation 1, 2, 4) operating
#  on the PLM embedding treated as a 1D sequence of features.
#  Adaptive max-pooling collapses the feature dimension to a fixed-length vector,
#  followed by a two-layer FC classification head.
#
#  auto_scale_config() adjusts hidden_dim, dropout, batch_size, epochs, and lr
#  from dataset size × embedding dimension using the same two-axis system as
#  transformer_lm.py, ensuring consistent behaviour across all four PLMs.
#
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
    roc_curve, auc,
)
import matplotlib.pyplot as plt

try:
    from config import MODEL_DIR
except ImportError:
    MODEL_DIR = "models/saved"

try:
    from utils.threshold_optimizer import run_full_threshold_pipeline
    _THRESHOLD_OPT_AVAILABLE = True
except ImportError:
    _THRESHOLD_OPT_AVAILABLE = False


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
# 2.  LOSS FUNCTIONS  (identical to transformer_lm.py)
# ══════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def auto_select_loss(y: np.ndarray) -> tuple:
    """Same decision rules as transformer_lm.py."""
    y_np         = np.asarray(y, dtype=int)
    class_counts = np.bincount(y_np)
    total        = len(y_np)
    min_rate     = class_counts.min() / total

    if min_rate < 0.15:
        loss_type, gamma, smoothing = 'focal', 2.0, 0.0
    elif min_rate < 0.30:
        loss_type, gamma, smoothing = 'focal', 1.5, 0.0
    elif total > 50_000:
        loss_type, gamma, smoothing = 'label_smooth', 2.0, 0.10
    else:
        loss_type, gamma, smoothing = 'weighted_ce', 2.0, 0.0

    print(
        f"[auto_loss] n={total:,}  min_class_rate={min_rate:.1%}"
        f"  → loss={loss_type}"
        + (f"  γ={gamma}"     if loss_type == 'focal'        else "")
        + (f"  ε={smoothing}" if loss_type == 'label_smooth' else "")
    )
    return loss_type, gamma, smoothing


def build_criterion(y_train, device, loss_type='auto',
                    focal_gamma=2.0, label_smoothing=0.05):
    y_np         = np.asarray(y_train, dtype=int)
    class_counts = np.bincount(y_np)
    n_classes    = len(class_counts)
    total        = len(y_np)
    weights      = torch.tensor(
        total / (n_classes * class_counts), dtype=torch.float
    ).to(device)

    if loss_type == 'auto':
        loss_type, focal_gamma, label_smoothing = auto_select_loss(y_np)

    if loss_type == 'focal':
        criterion = FocalLoss(weight=weights, gamma=focal_gamma)
    elif loss_type == 'label_smooth':
        criterion = nn.CrossEntropyLoss(weight=weights,
                                        label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)

    return criterion, weights


# ══════════════════════════════════════════════════════════════════
# 3.  MODEL — ResidualBlock + BinderClassifier
# ══════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """
    Dilated residual block: two Conv1d layers with the same dilation,
    BatchNorm + GELU activation, and a projection shortcut if channel
    dimensions differ.
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1):
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.conv1    = nn.Conv1d(in_channels,  out_channels, kernel_size,
                                   padding=pad, dilation=dilation)
        self.bn1      = nn.BatchNorm1d(out_channels)
        self.conv2    = nn.Conv1d(out_channels, out_channels, kernel_size,
                                   padding=pad, dilation=dilation)
        self.bn2      = nn.BatchNorm1d(out_channels)
        self.act      = nn.GELU()
        self.shortcut = (nn.Conv1d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class CNNClassifier(nn.Module):
    """
    1D-CNN classifier for fixed-length PLM embeddings.

    Forward pass:
        embedding [B, emb_dim]
          → unsqueeze(1)              [B, 1, emb_dim]
          → ResidualBlock(1→128, d=1) [B, 128, emb_dim]
          → ResidualBlock(128→256, d=2)
          → ResidualBlock(256→256, d=4)
          → AdaptiveMaxPool1d(1)      [B, 256]
          → Dropout
          → Linear(256→hidden_dim) + BN + GELU
          → Dropout
          → Linear(hidden_dim→2)      [B, 2]  logits

    The three residual blocks with increasing dilation capture local,
    medium-range, and longer-range patterns across the embedding vector,
    analogous to multi-scale feature extraction.
    """
    def __init__(self, embedding_dim: int = 1024,
                 hidden_dim: int = 128,
                 dropout: float = 0.4,
                 num_classes: int = 2):
        super().__init__()
        self.res1    = ResidualBlock(1,   128, kernel_size=5, dilation=1)
        self.res2    = ResidualBlock(128, 256, kernel_size=5, dilation=2)
        self.res3    = ResidualBlock(256, 256, kernel_size=5, dilation=4)
        self.pool    = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(256, hidden_dim)
        self.bn_fc   = nn.BatchNorm1d(hidden_dim)
        self.act     = nn.GELU()
        self.fc2     = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)           # [B, 1, emb_dim]
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x).squeeze(-1) # [B, 256]
        x = self.dropout(x)
        x = self.act(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)           # [B, 2]

    def forward_hidden(self, x) -> torch.Tensor:
        """
        Returns the task-specific feature vector BEFORE the final linear layer.
        Shape: [B, hidden_dim]

        Extraction point:
            res1 → res2 → res3 → pool → dropout → fc1 → BN → GELU  ← here
            → dropout → fc2 (classification head, NOT included)
        """
        x = x.unsqueeze(1)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.act(self.bn_fc(self.fc1(x)))    # [B, hidden_dim]


# ══════════════════════════════════════════════════════════════════
# 4.  AUTO-SCALE CONFIG  (same two-axis system as transformer_lm.py)
# ══════════════════════════════════════════════════════════════════

def auto_scale_config(n_samples: int, embedding_dim: int, base_cfg: dict) -> dict:
    """
    Two-axis continuous scaling identical to transformer_lm.py.

    Axis 1 — n_norm: absolute dataset size   → batch_size, epochs, lr
    Axis 2 — d_norm: samples/embedding dim   → hidden_dim, dropout

    CNN-specific differences from transformer_lm:
      • hidden_dim controls the FC head (256→hidden→2), not attention
      • No num_layers / num_heads parameters
      • Slightly lower dropout ceiling (0.45 vs 0.50) — BatchNorm
        already provides regularisation inside residual blocks

    Verified outputs (IPI PSR ~12k, AntiBERTa2-CSSP 1024-dim):
      n_norm=0.44  d_norm=0.17  → hidden=128  dr=0.36  bs=32  ep=37
    """
    N_MIN, N_MAX = 500, 500_000
    n_norm = float(np.clip(
        np.log10(max(n_samples, 1) / N_MIN) / np.log10(N_MAX / N_MIN),
        0.0, 1.0
    ))

    ratio  = n_samples / max(embedding_dim, 1)
    D_MIN, D_MAX = 5.0, 500.0
    d_norm = float(np.clip(
        np.log10(max(ratio, D_MIN) / D_MIN) / np.log10(D_MAX / D_MIN),
        0.0, 1.0
    ))

    # hidden_dim: 64→256, rounded to 64; floor = max(64, min(128, emb/16))
    # CNN head needs less capacity than Transformer hidden — emb/16 floor
    hd_formula = int(round((64 + (256 - 64) * d_norm) / 64)) * 64
    hd_floor   = max(64, min(128, int(round(embedding_dim / 16 / 64)) * 64))
    hd         = max(hd_formula, hd_floor)
    hd         = max(64, min(256, hd))

    # dropout: 0.10 → 0.45 (lower ceiling than Transformer — BN helps)
    overfit_risk = 1.0 - 0.5 * (n_norm + d_norm)
    dr = round(max(0.10, min(0.45, 0.10 + 0.35 * overfit_risk)), 2)

    # batch_size: 8→128, power-of-2 (CNN handles larger batches better)
    bs = min(128, max(8, 8 * (2 ** round(4 * n_norm))))

    # epochs: 50→10 (CNN converges faster than Transformer)
    ep_base = max(10, round(50 - 40 * n_norm))
    if d_norm < 0.5:
        extra = min(15, round(ep_base * 0.4 * (0.5 - d_norm) / 0.5))
        ep    = ep_base + extra
    else:
        ep    = ep_base

    # lr: linear scaling rule
    lr = round(1e-4 * float(np.sqrt(bs / 16)), 8)

    # Apply YAML ceiling overrides (same mechanism as transformer_lm.py)
    cfg = copy.deepcopy(base_cfg)
    cfg['model']['hidden_dim']    = hd
    cfg['model']['dropout']       = dr
    cfg['training']['batch_size'] = bs
    cfg['training']['epochs']     = ep
    cfg['training']['lr']         = lr

    # Hard ceiling from YAML (user can cap epochs without disabling auto_scale)
    for key in ['epochs', 'batch_size']:
        yaml_val = base_cfg.get('training', {}).get(key)
        auto_val = cfg['training'].get(key)
        if yaml_val is not None and yaml_val < auto_val:
            cfg['training'][key] = yaml_val
            print(f"[auto_scale] {key} capped by YAML: {auto_val} → {yaml_val}")

    print(
        f"[auto_scale] n={n_samples:,}  emb={embedding_dim}"
        f"  ratio={ratio:.1f}  n_norm={n_norm:.2f}  d_norm={d_norm:.2f}"
        f"\n             → hidden={hd}  dropout={dr}"
        f"  batch={bs}  epochs={ep}  lr={lr:.2e}"
    )
    return cfg


def auto_kfold(n_samples: int, min_val_size: int = 50) -> int:
    """Identical to transformer_lm.py."""
    k_raw = n_samples // min_val_size
    k     = max(3, min(10, k_raw))
    if   k >= 8: k = 10
    elif k >= 4: k = 5
    else:        k = 3
    print(f"[auto_kfold] n={n_samples:,}  → {k}-fold CV"
          f"  (~{n_samples // k:,} samples/val fold)")
    return k


def validate_dataset(y, embedding_dim, context='train', n_samples=None) -> dict:
    """Identical guard logic as transformer_lm.py."""
    y   = np.asarray(y, dtype=int)
    n   = len(y) if n_samples is None else n_samples
    if n < 2:
        raise ValueError(f"[validate] n={n}: cannot train on fewer than 2 samples.")
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(
            f"[validate] Only class {classes[0]} found. Binary classification "
            f"requires both class 0 and class 1."
        )
    class_counts = np.bincount(y)
    min_class    = int(class_counts.min())
    min_rate     = min_class / n
    warnings_list = []

    N_MIN, N_MAX = 500, 500_000
    n_norm  = float(np.clip(
        np.log10(max(n, 1) / N_MIN) / np.log10(N_MAX / N_MIN), 0.0, 1.0
    ))
    bs_auto = min(128, max(8, 8 * (2 ** round(4 * n_norm))))
    safe_bs = min(bs_auto, max(2, n - 1))   # -1: drop_last=True needs n>bs
    if safe_bs < bs_auto:
        msg = f"batch_size clamped {bs_auto} → {safe_bs}  (n={n})"
        warnings_list.append(msg)
        print(f"[validate] ⚠  {msg}")

    safe_kfold = None
    if context == 'kfold':
        if   n < 500:  k = 3
        elif n < 1000: k = 5
        else:          k = 10
        k = min(k, min_class, n // 2)
        k = max(2, k)
        safe_kfold = k

    if not warnings_list:
        print(
            f"[validate] ✓  n={n:,}  emb={embedding_dim}  "
            f"min_class={min_class} ({min_rate:.1%})"
            f"  → safe_batch={safe_bs}"
            + (f"  safe_kfold={safe_kfold}" if safe_kfold else "")
        )
    return {'n': n, 'min_class': min_class, 'min_rate': min_rate,
            'safe_batch': safe_bs, 'safe_kfold': safe_kfold,
            'warnings': warnings_list}


# ══════════════════════════════════════════════════════════════════
# 5.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════

def train_model(model, train_loader, val_loader,
                criterion, optimizer, scheduler,
                epochs=20, patience=10, device='cpu',
                model_save="models/saved/CNN.pt",
                use_amp=False):
    """
    Training loop matching transformer_lm.py:
      ✓ gradient clipping (max_norm=1.0)
      ✓ early stopping on val_AUC  (robust to class imbalance)
      ✓ best state_dict saved + reloaded before return
      ✓ optional AMP for GPU mixed-precision
      ✓ drop_last handled at DataLoader level (BatchNorm safety)

    Note: early stopping monitors val_AUC not val_loss, for the same
    reason as transformer_lm.py — val_loss is misleading under
    class imbalance while AUC directly measures discrimination.
    """
    save_dir = os.path.dirname(model_save)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    model       = model.to(device)
    _amp_device = 'cuda' if str(device) != 'cpu' else 'cpu'
    scaler      = torch.amp.GradScaler(
        device=_amp_device,
        enabled=(use_amp and str(device) != 'cpu')
    )
    best_val_auc = -1.0
    best_state   = None
    patience_ctr = 0
    use_val      = val_loader is not None

    for epoch in range(epochs):
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

        t_acc = accuracy_score(t_true, t_preds)
        log   = (f"Epoch {epoch+1:03d}/{epochs}"
                 f"  train_loss={t_loss/len(train_loader):.4f}"
                 f"  train_acc={t_acc:.4f}")

        if use_val:
            model.eval()
            v_probs, v_preds, v_true = [], [], []
            v_loss = 0.0

            with torch.no_grad():
                for emb, lbl, _ in val_loader:
                    emb, lbl = emb.to(device), lbl.to(device)
                    out       = model(emb)
                    v_loss   += criterion(out, lbl).item()
                    v_probs.extend(torch.softmax(out, 1)[:, 1].cpu().numpy())
                    v_preds.extend(torch.argmax(out, 1).cpu().numpy())
                    v_true.extend(lbl.cpu().numpy())

            v_loss /= len(val_loader)
            try:
                from sklearn.metrics import roc_auc_score as _roc_auc
                v_auc = _roc_auc(v_true, v_probs) if len(set(v_true)) > 1 else 0.5
            except Exception:
                v_auc = 0.5

            scheduler.step(v_loss)
            log += f"  |  val_loss={v_loss:.4f}  val_auc={v_auc:.4f}"

            if v_auc > best_val_auc:
                best_val_auc = v_auc
                best_state   = copy.deepcopy(model.state_dict())
                torch.save(best_state, model_save)
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(log)
                    print(f"  → Early stopping at epoch {epoch+1}")
                    break
        else:
            torch.save(model.state_dict(), model_save)

        print(log)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  → Best weights restored  (val_auc={best_val_auc:.4f})")

    return model


# ══════════════════════════════════════════════════════════════════
# 6.  DEFAULT CONFIG
# ══════════════════════════════════════════════════════════════════

_DEFAULT_CONFIG = {
    'model': {
        'hidden_dim': 128,
        'dropout':    0.4,
    },
    'training': {
        'epochs':       20,
        'batch_size':   32,
        'lr':           1e-4,
        'weight_decay': 0.001,
    },
    'scheduler': {
        'mode':    'min',
        'factor':   0.5,
        'patience': 6,
    },
    'loss': {
        'type':             'auto',
        'focal_gamma':       2.0,
        'label_smoothing':   0.05,
    },
    'auto_scale': True,
    'amp':        False,
}


# ══════════════════════════════════════════════════════════════════
# 7.  PUBLIC WRAPPER  —  CNNModel
# ══════════════════════════════════════════════════════════════════

class CNNModel:
    """
    Framework wrapper matching TransformerLMModel's public API exactly.

    Typical usage:
        # train
        model = CNNModel()
        model.train(X, y)
        model.save("FINAL_psr_filter_ablang_cnn_ipi_antibodydb.pt")

        # predict
        model = CNNModel.load(path, embedding_dim=X.shape[1])
        scores = model.predict_proba(X)

        # k-fold
        CNNModel().kfold_validation(
            data, X, y, embedding_lm='ablang',
            title='PSR_cnn', kfold=10, target='psr_filter'
        )
    """

    def __init__(self, config_path="config/cnn.yaml"):
        if os.path.exists(config_path):
            with open(config_path) as f:
                user_cfg = yaml.safe_load(f)
            self.config = _deep_merge(copy.deepcopy(_DEFAULT_CONFIG), user_cfg or {})
            print(f"[CNNModel] config ← {config_path}")
        else:
            self.config = copy.deepcopy(_DEFAULT_CONFIG)
            print(f"[CNNModel] {config_path} not found — using built-in defaults")

        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model          = None
        self._embedding_dim = None

    # ── internal helpers ─────────────────────────────────────────

    def _build_model(self, embedding_dim: int) -> CNNClassifier:
        self._embedding_dim = embedding_dim
        c = self.config['model']
        return CNNClassifier(
            embedding_dim=embedding_dim,
            hidden_dim=c['hidden_dim'],
            dropout=c['dropout'],
        ).to(self.device)

    def _to_loader(self, X, y, batch_size: int, shuffle: bool,
                   drop_last: bool = False) -> DataLoader:
        X_np     = X.values if hasattr(X, 'values') else np.asarray(X, dtype=np.float32)
        barcodes = (X.index.tolist() if hasattr(X, 'index')
                    else [f"ab_{i}" for i in range(len(X_np))])
        ds = AntibodyDataset(X_np, y, barcodes)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            drop_last=drop_last,          # required for BatchNorm with small batches
            num_workers=0,
            pin_memory=(str(self.device) != 'cpu'),
        )

    # ── train ─────────────────────────────────────────────────────

    def train(self, X, y, val_X=None, val_y=None,
              epochs=None, batch_size=None,
              target: str = "model", db_stem: str = "",
              embedding_lm: str = "", cluster_col: str = "HCDR3_CLUSTER_0.8"):
        y       = np.asarray(y, dtype=int)
        n       = len(y)
        emb_dim = X.shape[1]

        vd = validate_dataset(y, emb_dim, context='train')

        if self.config.get('auto_scale', True):
            self.config = auto_scale_config(n, emb_dim, self.config)

        cfg        = self.config['training']
        epochs     = epochs     or cfg['epochs']
        batch_size = batch_size or vd['safe_batch']

        # drop_last=True for training — BatchNorm breaks on batch_size=1
        train_loader = self._to_loader(X, y, batch_size, shuffle=True, drop_last=True)
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
        tmp_save = os.path.join(MODEL_DIR, "_tmp_CNN.pt")

        self.model = train_model(
            self.model, train_loader, val_loader,
            criterion, optimizer, scheduler,
            epochs=epochs, patience=sch.get('patience', 10),
            device=self.device, model_save=tmp_save,
            use_amp=self.config.get('amp', False),
        )
        return self

    # ── predict ──────────────────────────────────────────────────

    def predict_proba(self, X) -> np.ndarray:
        assert self.model is not None, "Model not trained or loaded."
        self.model.eval()
        X_np     = X.values if hasattr(X, 'values') else np.asarray(X, dtype=np.float32)
        barcodes = (X.index.tolist() if hasattr(X, 'index')
                    else [f"ab_{i}" for i in range(len(X_np))])
        ds     = AntibodyDataset(X_np, np.zeros(len(X_np), dtype=int), barcodes)
        # drop_last=False for inference — need all samples
        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

        probs = []
        with torch.no_grad():
            for emb, _, _ in loader:
                out = self.model(emb.to(self.device))
                probs.extend(torch.softmax(out, 1)[:, 1].cpu().numpy())
        return np.array(probs)

    def extract_hidden_states(self, X, batch_size: int = 256) -> np.ndarray:
        """
        Extract task-specific features from the penultimate layer (after fc1+BN+GELU,
        before the final classification fc2).

        Shape: [B, hidden_dim]  e.g. (1477, 128)

        This is the CNN's learned representation — more task-specific than the raw
        PLM embedding. Use for t-SNE, UMAP, or interpretability analysis.

        Example
        -------
            model  = CNNModel.load('FINAL_psr_igbert_cnn_DS1.pt', embedding_dim=1024)
            X_emb  = pd.read_csv('DS1.xlsx.igbert.emb.csv', index_col=0)
            hidden = model.extract_hidden_states(X_emb)
            pd.DataFrame(hidden, index=X_emb.index).to_csv('DS1_igbert_cnn_hidden.emb.csv')
        """
        assert self.model is not None, "Model not trained or loaded."
        self.model.eval()

        X_np     = X.values if hasattr(X, 'values') else np.asarray(X, dtype=np.float32)
        barcodes = (X.index.tolist() if hasattr(X, 'index')
                    else [f"ab_{i}" for i in range(len(X_np))])
        ds     = AntibodyDataset(X_np, np.zeros(len(X_np), dtype=int), barcodes)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        hidden_states = []
        with torch.no_grad():
            for emb, _, _ in loader:
                h = self.model.forward_hidden(emb.to(self.device))
                hidden_states.append(h.cpu().numpy())

        result = np.concatenate(hidden_states, axis=0)
        print(f"[extract_hidden_states] shape={result.shape}  "
              f"hidden_dim={result.shape[1]}  n={result.shape[0]:,}")
        return result

    def save_hidden_states(self, X, out_csv: str, batch_size: int = 256) -> str:
        """
        Extract CNN hidden states and save to CSV (BARCODE as index).
        Same format as PLM embedding CSVs — compatible with
        --tsne-source embedding in developability_correlation.py.

        Example
        -------
            model.save_hidden_states(X_emb, 'test/DS1_igbert_cnn_hidden.emb.csv')
        """
        hidden = self.extract_hidden_states(X, batch_size=batch_size)
        index  = X.index if hasattr(X, 'index') else [f"ab_{i}" for i in range(len(hidden))]
        df_out = pd.DataFrame(hidden, index=index,
                              columns=[f"h{i}" for i in range(hidden.shape[1])])
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        df_out.to_csv(out_csv)
        print(f"[save_hidden_states] → {out_csv}  ({len(df_out):,} rows × {hidden.shape[1]} dims)")
        return out_csv

    def predict(self, X, threshold: float = None) -> np.ndarray:
        t = threshold if threshold is not None else getattr(self, 'recommended_threshold', 0.5)
        if threshold is None:
            print(f"[predict] threshold={t:.4f}"
                  + (" (from checkpoint)" if hasattr(self, 'recommended_threshold') else " (default 0.5)"))
        return (self.predict_proba(X) >= t).astype(int)

    # ── save / load ───────────────────────────────────────────────

    def save(self, path: str):
        save_dir = os.path.dirname(path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        rec_thresh = getattr(self, 'recommended_threshold', None)
        torch.save(
            {
                'state_dict':            self.model.state_dict(),
                'config':                self.config,
                'embedding_dim':         self._embedding_dim,
                'recommended_threshold': rec_thresh,
            },
            path,
        )
        thresh_note = (f"  threshold={rec_thresh:.4f}"
                       if rec_thresh is not None else "  threshold=None (0.5 at predict)")
        print(f"[CNNModel] saved → {path}{thresh_note}")

    @classmethod
    def load(cls, path: str, embedding_dim: int = None,
             config_path: str = "config/cnn.yaml"):
        """
        Load checkpoint. Handles two formats:
          NEW   dict with 'state_dict' + 'config' + 'embedding_dim'
          RAW   raw state_dict (old format) — requires embedding_dim argument
        """
        instance = cls(config_path)
        payload  = torch.load(path, map_location=instance.device,
                               weights_only=False)

        if isinstance(payload, dict) and 'state_dict' in payload:
            state_dict  = payload['state_dict']
            ckpt_config = payload.get('config', {})
            ckpt_emb    = payload.get('embedding_dim')
        else:
            state_dict  = payload   # raw state_dict (old format)
            ckpt_config = {}
            ckpt_emb    = None

        emb_dim = embedding_dim or ckpt_emb
        if emb_dim is None:
            raise ValueError(
                f"Cannot determine embedding_dim for {path}. "
                f"Pass embedding_dim=X.shape[1] to CNNModel.load()."
            )

        instance.config = _deep_merge(instance.config, ckpt_config)
        instance.model  = instance._build_model(emb_dim)
        instance.model.load_state_dict(state_dict)
        instance.model.eval()
        rt = payload.get('recommended_threshold', None) if isinstance(payload, dict) else None
        instance.recommended_threshold = float(rt) if rt is not None else 0.5
        if rt is not None and rt != 0.5:
            print(f"[CNNModel] recommended_threshold={rt:.4f}  (embedded by kfold)")
        else:
            print(f"[CNNModel] recommended_threshold=0.5  (default)")
        print(f"[CNNModel] loaded ← {path}  (emb_dim={emb_dim})")
        return instance

    # ── kfold_validation ──────────────────────────────────────────

    def kfold_validation(
        self,
        db_stem,
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
        dbname = db_stem  # backward compatibility alias
        """
        Stratified-Group K-Fold cross-validation matching transformer_lm.py:
          ✓ HCDR3-cluster-stratified splits (StratifiedGroupKFold)
          ✓ auto_scale per fold
          ✓ early stopping on val_AUC
          ✓ best-fold BEST_*.pt saved with recommended_threshold embedded
          ✓ auto threshold optimisation via utils/threshold_optimizer.py
          ✓ Rec(Fail) reported separately from weighted recall
          ✓ ROC plot saved to MODEL_DIR
        """
        if target is None:
            raise ValueError(
                "kfold_validation() requires target= to be passed explicitly."
            )
        if title is None:
            title = f"{target.upper()}_cnn"

        y       = np.asarray(y, dtype=int)
        X_np    = X.values if hasattr(X, 'values') else np.asarray(X, dtype=np.float32)
        emb_dim = X_np.shape[1]
        n       = len(y)

        vd    = validate_dataset(y, emb_dim, context='kfold')
        kfold = kfold or vd['safe_kfold']

        # Splitter selection — same logic as transformer_lm.py
        if cluster_col in data.columns:
            groups          = data[cluster_col].values
            n_unique_groups = len(np.unique(groups))
            if n_unique_groups == n:
                print(f"[kfold] all-singleton clusters → StratifiedKFold")
                splitter   = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
                split_iter = splitter.split(np.arange(n), y)
            elif n_unique_groups < kfold:
                kfold      = n_unique_groups
                splitter   = StratifiedGroupKFold(n_splits=kfold, shuffle=True, random_state=42)
                split_iter = splitter.split(np.arange(n), y, groups)
            else:
                splitter   = StratifiedGroupKFold(n_splits=kfold, shuffle=True, random_state=42)
                split_iter = splitter.split(np.arange(n), y, groups)
        else:
            print(f"[kfold] '{cluster_col}' not found → StratifiedKFold")
            splitter   = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
            split_iter = splitter.split(np.arange(n), y)

        barcodes = data.index.tolist()
        full_ds  = AntibodyDataset(X_np, y, barcodes)
        mean_fpr = np.linspace(0, 1, 100)

        tprs, aucs_list, accs, f1s, precs, recs, recs_fail = [], [], [], [], [], [], []
        all_records = []
        best_fold_auc   = -1.0
        best_fold_num   = -1
        best_fold_state = None
        best_fold_cfg   = None

        plt.figure(figsize=(5, 5))
        print(f"\n[kfold] {kfold}-fold SGKF | {title} | lm={embedding_lm}")

        for fold, (tr_idx, va_idx) in enumerate(split_iter, 1):
            print(f"\n── Fold {fold}/{kfold} ──")

            # Fresh sub-model per fold
            fm          = CNNModel.__new__(CNNModel)
            fm.config   = copy.deepcopy(self.config)
            fm.device   = self.device
            fm.model    = None
            fm._embedding_dim = None

            n_fold = len(tr_idx)
            if fm.config.get('auto_scale', True):
                fm.config = auto_scale_config(n_fold, emb_dim, fm.config)

            cfg = fm.config['training']
            bs  = cfg['batch_size']

            # drop_last=True for training — BatchNorm safety
            tr_loader = DataLoader(Subset(full_ds, tr_idx),
                                   batch_size=bs, shuffle=True,
                                   drop_last=True,  num_workers=0)
            va_loader = DataLoader(Subset(full_ds, va_idx),
                                   batch_size=bs, shuffle=False, num_workers=0)

            fold_y        = y[tr_idx]
            loss_cfg      = fm.config.get('loss', {})
            criterion, cw = build_criterion(
                fold_y, fm.device,
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
                f"_fold{fold}_{target}_{embedding_lm}_cnn.pt",
            )
            fm.model = train_model(
                fm.model, tr_loader, va_loader,
                criterion, optimizer, scheduler,
                epochs   = cfg['epochs'],
                patience = sch.get('patience', 10),
                device   = fm.device,
                model_save = fold_save,
                use_amp  = fm.config.get('amp', False),
            )

            # Evaluate
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
                print(f"  Skipping fold {fold} — only one class in val.")
                continue

            fpr, tpr, _ = roc_curve(trues, probs)
            f_auc       = auc(fpr, tpr)
            f_acc       = accuracy_score(trues, preds)
            f_f1        = f1_score(trues,       preds, average='weighted', zero_division=0)
            f_prec      = precision_score(trues, preds, average='weighted', zero_division=0)
            f_rec       = recall_score(trues,   preds, average='weighted', zero_division=0)
            f_rec_fail  = recall_score(trues, preds, pos_label=0, average='binary',
                                       zero_division=0)

            aucs_list.append(f_auc); accs.append(f_acc)
            f1s.append(f_f1);        precs.append(f_prec)
            recs.append(f_rec);      recs_fail.append(f_rec_fail)
            tprs.append(np.interp(mean_fpr, fpr, tpr)); tprs[-1][0] = 0.0

            if f_auc > best_fold_auc:
                best_fold_auc   = f_auc
                best_fold_num   = fold
                best_fold_state = copy.deepcopy(fm.model.state_dict())
                best_fold_cfg   = copy.deepcopy(fm.config)

            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label=f'Fold {fold} (AUC={f_auc:.3f})')
            print(f"  AUC={f_auc:.3f}  Acc={f_acc:.3f}  F1={f_f1:.3f}"
                  f"  Prec={f_prec:.3f}  Rec={f_rec:.3f}"
                  f"  Rec(Fail)={f_rec_fail:.3f}")

            if save_fold_preds:
                for bc, true, pred, prob in zip(bcs, trues, preds, probs):
                    all_records.append({
                        'BARCODE': bc, 'fold': fold,
                        'true': true, 'pred': pred, 'prob': prob,
                    })

        if not aucs_list:
            print("[kfold] No valid folds.")
            return None, None, None, None, None, None

        mean_tpr      = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
        mean_auc      = auc(mean_fpr, mean_tpr)
        std_auc       = np.std(aucs_list)
        mean_acc      = np.mean(accs)
        mean_f1       = np.mean(f1s)
        mean_prec     = np.mean(precs)
        mean_rec      = np.mean(recs)
        mean_rec_fail = np.mean(recs_fail)

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
        plt.xlim([0, 1]); plt.ylim([0, 1.05])
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

        os.makedirs(MODEL_DIR, exist_ok=True)
        plot_path = os.path.join(
            MODEL_DIR,
            f"CV_ROC_{target}_{embedding_lm}_cnn_{dbname}_k{kfold}.png",
        )
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\n[kfold] ROC plot → {plot_path}")

        if save_fold_preds and all_records:
            pred_path = os.path.join(
                MODEL_DIR,
                f"fold_preds_{target}_{embedding_lm}_cnn_{dbname}_k{kfold}.csv",
            )
            pd.DataFrame(all_records).to_csv(pred_path, index=False)
            print(f"[kfold] Fold predictions → {pred_path}")

        if best_fold_state is not None:
            best_path = os.path.join(
                MODEL_DIR,
                f"BEST_{target}_{embedding_lm}_cnn_{dbname}_k{kfold}_fold{best_fold_num}.pt",
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
                },
                best_path,
            )
            print(f"[kfold] Best fold model  → {best_path}"
                  f"  (fold={best_fold_num}, AUC={best_fold_auc:.3f})")

        _min_rate   = float(np.mean(y) if np.mean(y) < 0.5 else 1 - np.mean(y))

        print(f"\n{'─'*60}")
        print(f"  Best fold : {best_fold_num}  (AUC={best_fold_auc:.3f})")
        print(f"  Mean AUC  : {mean_auc:.3f} ± {std_auc:.3f}")
        if _min_rate < 0.30:
            print(f"  ── imbalanced ({_min_rate:.0%} minority) — metrics at t=0.5 ──")
        print(f"  Mean Acc  : {mean_acc:.3f}")
        print(f"  Mean F1   : {mean_f1:.3f}")
        print(f"  Mean Prec : {mean_prec:.3f}")
        print(f"  Mean Rec  : {mean_rec:.3f}  (Pass class)")
        print(f"  Rec(Fail) : {mean_rec_fail:.3f}  ← minority class recall (class 0)")
        print(f"{'─'*60}")

        # Auto threshold optimisation
        if _THRESHOLD_OPT_AVAILABLE and save_fold_preds and all_records:
            fold_preds_csv = os.path.join(
                MODEL_DIR,
                f"fold_preds_{target}_{embedding_lm}_cnn_{dbname}_k{kfold}.csv",
            )
            best_ckpt_path = os.path.join(
                MODEL_DIR,
                f"BEST_{target}_{embedding_lm}_cnn_{dbname}_k{kfold}_fold{best_fold_num}.pt",
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
                )
            except Exception as _e:
                print(f"[threshold] WARNING: optimisation failed — {_e}")
        elif not _THRESHOLD_OPT_AVAILABLE:
            print("\n[threshold] utils/threshold_optimizer.py not found — "
                  "skipping. Predictions will use threshold=0.5.")

        return mean_auc, std_auc, mean_acc, mean_f1, mean_prec, mean_rec

    # Alias for compatibility
    kfold_validation_sgkf = kfold_validation


# ══════════════════════════════════════════════════════════════════
# 8.  MODULE-LEVEL HELPER
# ══════════════════════════════════════════════════════════════════

def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base