#!/usr/bin/env python3
"""
Run from IPIPred root:
  python diagnose_7pct.py

"""
import sys, os
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np

# ── 1. Load model ──────────────────────────────────────────────────────────────
from IPIAbMLPred.models.transformer_onehot import TransformerOneHotModel
MODEL = "SEC/models_final/FINAL_psr_filter_onehot_transformer_onehot_ipi_psr_trainset.pt"
model = TransformerOneHotModel.load(MODEL)
thresh = getattr(model, 'recommended_threshold', 0.4388)
print(f"\n  Loaded: {MODEL}")
print(f"  Threshold: {thresh:.4f}")
print(f"  lm_mode  : {model.lm_mode}")

# ── 2. Predict on TRAINING DATA (known ground truth) ──────────────────────────
print("\n" + "═"*62)
print("  TEST 1: Score known positives from ipi_psr_trainset.xlsx")
print("─"*62)
train_df = pd.read_excel("test/ipi_psr_trainset.xlsx")
pos_train = train_df[train_df['psr_filter'] == 1].head(20)
neg_train = train_df[train_df['psr_filter'] == 0].head(20)

X_pos = pos_train[['HSEQ','LSEQ','CDR3']]
X_neg = neg_train[['HSEQ','LSEQ','CDR3']]

s_pos = model.predict_proba(X_pos)
s_neg = model.predict_proba(X_neg)
print(f"  Known positives: mean={s_pos.mean():.3f}  min={s_pos.min():.3f}  max={s_pos.max():.3f}")
print(f"  Known negatives: mean={s_neg.mean():.3f}  min={s_neg.min():.3f}  max={s_neg.max():.3f}")
print(f"  At thresh={thresh:.4f}:")
print(f"    pos predicted positive: {(s_pos >= thresh).mean():.0%}  (should be ~90%+)")
print(f"    neg predicted positive: {(s_neg >= thresh).mean():.0%}  (should be ~10%-)")

if s_pos.mean() < 0.4:
    print(f"\n  ✗ Model produces LOW scores even on TRAINING data!")
    print(f"    → Model checkpoint / loading is broken")
    print(f"    → Rerun kfold with current code and use the new BEST checkpoint")
else:
    print(f"\n  ✓ Model scores training data correctly")
    print(f"    → Bug is in how ipiab202603 sequences are read")

# ── 3. Compare CDR3 lengths between the two files ─────────────────────────────
print("\n" + "═"*62)
print("  TEST 2: CDR3 length distribution comparison")
print("─"*62)
pred_df = pd.read_excel("test/ipiab202603.xlsx")
train_cdr3_len = train_df['CDR3'].astype(str).str.len()
pred_cdr3_len  = pred_df['CDR3'].astype(str).str.len()
print(f"  ipi_psr_trainset CDR3:  mean={train_cdr3_len.mean():.1f}  "
      f"min={train_cdr3_len.min()}  max={train_cdr3_len.max()}")
print(f"  ipiab202603 CDR3:       mean={pred_cdr3_len.mean():.1f}  "
      f"min={pred_cdr3_len.min()}  max={pred_cdr3_len.max()}")
print(f"  CDR3 > 25 AA in trainset : {(train_cdr3_len > 25).mean():.1%}")
print(f"  CDR3 > 25 AA in ipiab    : {(pred_cdr3_len > 25).mean():.1%}")

# ── 4. Check LSEQ lengths ──────────────────────────────────────────────────────
print("\n" + "─"*62)
print("  TEST 3: LSEQ presence and length")
print("─"*62)
train_lseq = train_df['LSEQ'].astype(str)
pred_lseq  = pred_df['LSEQ'].astype(str)
print(f"  ipi_psr_trainset LSEQ:  mean_len={train_lseq.str.len().mean():.0f}  "
      f"empty={( train_lseq.str.len()==0).mean():.1%}")
print(f"  ipiab202603 LSEQ:       mean_len={pred_lseq.str.len().mean():.0f}  "
      f"empty={(pred_lseq.str.len()==0).mean():.1%}")
print(f"  'nan' strings in trainset LSEQ : {(train_lseq=='nan').mean():.1%}")
print(f"  'nan' strings in ipiab    LSEQ : {(pred_lseq=='nan').mean():.1%}")

# ── 5. Find training antibodies in ipiab and check their scores ────────────────
print("\n" + "═"*62)
print("  TEST 4: Predict on shared BARCODEs (training data in ipiab)")
print("─"*62)
if 'BARCODE' in train_df.columns and 'BARCODE' in pred_df.columns:
    shared = set(train_df['BARCODE'].astype(str)) & set(pred_df['BARCODE'].astype(str))
    print(f"  Shared BARCODEs: {len(shared)}")
    if shared:
        shared_pred = pred_df[pred_df['BARCODE'].astype(str).isin(shared)].head(20)
        X_shared = shared_pred[['HSEQ','LSEQ','CDR3']]
        s_shared = model.predict_proba(X_shared)
        print(f"  Scores for shared (from ipiab encoding): "
              f"mean={s_shared.mean():.3f}")

        # Same antibodies from trainset
        shared_train = train_df[train_df['BARCODE'].astype(str).isin(
            set(shared_pred['BARCODE'].astype(str)))].head(20)
        X_shared_tr = shared_train[['HSEQ','LSEQ','CDR3']]
        s_shared_tr = model.predict_proba(X_shared_tr)
        print(f"  Scores for same ABS from trainset file: "
              f"mean={s_shared_tr.mean():.3f}")
        diff = abs(s_shared.mean() - s_shared_tr.mean())
        if diff > 0.1:
            print(f"\n  ✗ SAME antibodies score DIFFERENTLY depending on source file!")
            print(f"    Δ = {diff:.3f} — column format differs between files")
        else:
            print(f"\n  ✓ Same antibodies score identically in both files (Δ={diff:.3f})")
else:
    print("  BARCODE column not found in one or both files — skipping")

print("\n" + "═"*62)