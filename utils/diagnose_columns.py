#!/usr/bin/env python3
"""
python utils/diagnose_columns.py test/ipiab202603_test.xlsx

Compares HSEQ / LSEQ / CDR3 content between ipiab and trainset
for the same BARCODEs. The Δ=0.683 score gap has exactly one cause:
the sequences in one file differ from the other for the same antibody.
"""
import sys, os, pandas as pd, numpy as np

TRAIN   = "test/ipi_psr_trainset.xlsx"
PREDICT = sys.argv[1] if len(sys.argv) > 1 else "test/ipiab202603_test.xlsx"

tr = pd.read_excel(TRAIN)
pr = pd.read_excel(PREDICT)

tr['BARCODE'] = tr['BARCODE'].astype(str).str.strip()
pr['BARCODE'] = pr['BARCODE'].astype(str).str.strip()

# All column names in both files
print("═"*62)
print("  COLUMN NAMES")
print("─"*62)
print(f"  trainset   : {list(tr.columns)}")
print(f"  ipiab      : {list(pr.columns)}")

# Check for exact column name match
for col in ['HSEQ', 'LSEQ', 'CDR3']:
    in_tr = col in tr.columns
    in_pr = col in pr.columns
    print(f"  {col:6s} in trainset: {'✓' if in_tr else '✗'}  "
          f"in ipiab: {'✓' if in_pr else '✗'}")

# Shared BARCODEs
shared = set(tr['BARCODE']) & set(pr['BARCODE'])
if not shared:
    print("\n  [ERROR] No shared BARCODEs found — check BARCODE column format")
    sys.exit(1)

tr_s = tr.set_index('BARCODE')
pr_s = pr.set_index('BARCODE')
sample = list(shared)[:10]

print("\n" + "═"*62)
print(f"  SEQUENCE CONTENT COMPARISON  (first 10 shared BARCODEs)")
print("─"*62)

hseq_match = lseq_match = cdr3_match = 0
hseq_swap  = 0   # HSEQ in ipiab matches LSEQ in trainset?

for bc in sample:
    tr_h = str(tr_s.loc[bc, 'HSEQ']).strip().upper()
    pr_h = str(pr_s.loc[bc, 'HSEQ']).strip().upper()
    tr_l = str(tr_s.loc[bc, 'LSEQ']).strip().upper()
    pr_l = str(pr_s.loc[bc, 'LSEQ']).strip().upper()
    tr_c = str(tr_s.loc[bc, 'CDR3']).strip().upper()
    pr_c = str(pr_s.loc[bc, 'CDR3']).strip().upper()

    h_ok  = (tr_h == pr_h)
    l_ok  = (tr_l == pr_l)
    c_ok  = (tr_c == pr_c)
    swapped = (tr_h == pr_l and tr_l == pr_h)

    if h_ok:  hseq_match += 1
    if l_ok:  lseq_match += 1
    if c_ok:  cdr3_match += 1
    if swapped: hseq_swap += 1

    print(f"\n  BARCODE={bc}")
    print(f"    HSEQ match : {'✓' if h_ok else '✗'}"
          f"  trainset[:15]={tr_h[:15]}  ipiab[:15]={pr_h[:15]}")
    print(f"    LSEQ match : {'✓' if l_ok else '✗'}"
          f"  trainset[:15]={tr_l[:15]}  ipiab[:15]={pr_l[:15]}")
    print(f"    CDR3 match : {'✓' if c_ok else '✗'}"
          f"  trainset={tr_c}  ipiab={pr_c}")
    if swapped:
        print(f"    *** HSEQ/LSEQ SWAPPED in ipiab for this BARCODE ***")

print("\n" + "═"*62)
print(f"  SUMMARY (out of {len(sample)} shared antibodies):")
print(f"    HSEQ exact match : {hseq_match}/{len(sample)}")
print(f"    LSEQ exact match : {lseq_match}/{len(sample)}")
print(f"    CDR3 exact match : {cdr3_match}/{len(sample)}")
print(f"    HSEQ↔LSEQ swapped: {hseq_swap}/{len(sample)}")
print("─"*62)

if hseq_match == 0 and hseq_swap == len(sample):
    print("  ✗ CONFIRMED: HSEQ and LSEQ are SWAPPED in ipiab202603!")
    print("    The model was trained with VH in HSEQ and VL in LSEQ.")
    print("    ipiab202603 has VL in HSEQ and VH in LSEQ.")
    print("\n  FIX:")
    print("    Option A: Swap columns in ipiab202603.xlsx before predicting")
    print("    Option B: Add column mapping to auto_predict() in predict_developability.py")
elif hseq_match == len(sample) and cdr3_match < len(sample):
    print("  ✗ HSEQ/LSEQ match but CDR3 differs!")
    print("    The CDR3 column in ipiab202603 contains different CDR3 sequences")
    print("    (e.g., LCDR3 instead of HCDR3, or different CDR3 annotation)")
elif hseq_match == 0 and hseq_swap == 0:
    print("  ✗ Sequences differ but are NOT a simple swap.")
    print("    Check: different sequence source (IMGT vs Kabat),")
    print("    gaps in alignment, or extra characters in sequences.")
elif hseq_match == len(sample):
    print("  ✓ All sequences match — bug is elsewhere in the pipeline")
print("═"*62)