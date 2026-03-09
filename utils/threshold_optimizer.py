# utils/threshold_optimizer.py
# Optimal Decision Threshold Selection for Antibody Biophysical Property Classifiers
# IPI Antibody Developability Prediction Platform — Production Version DEC-2025
#
# ── Why 0.5 is almost never the right threshold ───────────────────────────────
#
#   Classifiers output P(positive | x), calibrated to the training distribution.
#   The optimal *decision* threshold depends on:
#     1. Class imbalance  — imbalanced training shifts scores away from 0.5
#     2. Cost asymmetry   — missing a bad mAb (FN) costs more than a false alarm (FP)
#     3. Downstream use   — screening (high recall) vs confirmatory (high precision)
#
#   Your three trainsets:
#   ┌────────────────┬────────┬──────────────┬─────────────┬──────────────────┐
#   │ Dataset        │   n    │ positive rate│ imbalance   │ recommended      │
#   ├────────────────┼────────┼──────────────┼─────────────┼──────────────────┤
#   │ IPI PSR 12k    │ 12,019 │  ~52% Pass   │ balanced    │ Youden / F1      │
#   │ IPI SEC  8k    │  8,019 │  ~79% Pass   │ moderate    │ F2 / cost-sens.  │
#   │ Public DS1     │246,293 │  ~53% Pass   │ balanced    │ F1 / Youden      │
#   └────────────────┴────────┴──────────────┴─────────────┴──────────────────┘
#
# ── Methods implemented ───────────────────────────────────────────────────────
#   youden       : maximise Sensitivity + Specificity - 1  (balanced datasets)
#   f1           : maximise F1  (harmonic mean precision/recall)
#   f2           : maximise F2  (recall-weighted, β=2; SEC imbalanced datasets)
#   fbeta        : maximise Fβ  for any β (caller-specified)
#   precision_at : first threshold where Precision ≥ target (screening gate)
#   recall_at    : first threshold where Recall    ≥ target (safety net)
#   cost         : minimise FP·cost_fp + FN·cost_fn (explicit cost function)
#
# ── Integration ───────────────────────────────────────────────────────────────
#   Called after kfold_validation() with fold_preds CSV, or after .predict_proba().
#   Adds a 'recommended_threshold' field to the BEST_*.pt checkpoint metadata.
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, roc_auc_score,
)

try:
    from config import MODEL_DIR
except ImportError:
    MODEL_DIR = "models/saved"


# ══════════════════════════════════════════════════════════════════
# 1.  CORE THRESHOLD SWEEP
# ══════════════════════════════════════════════════════════════════

def _sweep(y_true: np.ndarray, y_proba: np.ndarray,
           n_steps: int = 500) -> pd.DataFrame:
    """
    Compute all metrics at every candidate threshold.

    Returns a DataFrame with columns:
        threshold, tp, fp, tn, fn,
        sensitivity (recall), specificity, precision,
        f1, f2, youden, cost_{fp_cost}_{fn_cost}
    """
    y_true  = np.asarray(y_true,  dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    thresholds = np.linspace(
        y_proba.min() + 1e-6,
        y_proba.max() - 1e-6,
        n_steps
    )

    rows = []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1   = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
        f2   = (5 * prec * sens) / (4 * prec + sens) if (4 * prec + sens) > 0 else 0.0

        rows.append({
            'threshold':   t,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'sensitivity': sens,
            'specificity': spec,
            'precision':   prec,
            'f1':          f1,
            'f2':          f2,
            'youden':      sens + spec - 1,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
# 2.  SINGLE-METHOD THRESHOLD FINDER
# ══════════════════════════════════════════════════════════════════

def find_optimal_threshold(
    y_true:       np.ndarray,
    y_proba:      np.ndarray,
    method:       str   = 'auto',
    beta:         float = 1.0,
    cost_fp:      float = 1.0,
    cost_fn:      float = 2.0,
    precision_target: float = 0.80,
    recall_target:    float = 0.80,
    n_steps:      int   = 500,
) -> dict:
    """
    Find the optimal decision threshold using the specified method.

    Parameters
    ----------
    y_true            : ground-truth binary labels (0/1)
    y_proba           : predicted probabilities for class 1
    method            : one of 'auto', 'youden', 'f1', 'f2', 'fbeta',
                        'precision_at', 'recall_at', 'cost'
    beta              : Fβ beta value (used when method='fbeta')
    cost_fp           : cost of a false positive (used when method='cost')
    cost_fn           : cost of a false negative (used when method='cost')
                        For SEC: missing a failing mAb → cost_fn should be
                        higher than cost_fp (e.g. cost_fn=3, cost_fp=1)
    precision_target  : minimum precision required (method='precision_at')
    recall_target     : minimum recall  required  (method='recall_at')

    method='auto' decision table
    ─────────────────────────────────────────────────────────────────────
    Minority class rate   n            Selected method
    ──────────────────    ──────────   ───────────────────────────────────
    > 35 %                any          youden  (balanced — PSR, DS1)
    15–35 %               any          f2      (moderate imbalance — SEC)
    < 15 %                any          cost    (heavy imbalance)

    Returns
    -------
    dict with keys:
        method, threshold, sensitivity, specificity, precision,
        f1, f2, youden, tp, fp, tn, fn, auc_roc, auc_pr
    """
    y_true  = np.asarray(y_true,  dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    # Auto-select method from class statistics
    if method == 'auto':
        class_counts = np.bincount(y_true)
        min_rate     = class_counts.min() / len(y_true)
        if min_rate > 0.35:
            method = 'youden'
        elif min_rate > 0.15:
            method = 'f2'
        else:
            method = 'cost'
        print(f"[threshold] auto → method='{method}'  "
              f"min_class_rate={min_rate:.1%}")

    sweep = _sweep(y_true, y_proba, n_steps=n_steps)

    # ── Method dispatch ──────────────────────────────────────────
    if method == 'youden':
        idx = sweep['youden'].idxmax()

    elif method == 'f1':
        idx = sweep['f1'].idxmax()

    elif method == 'f2':
        idx = sweep['f2'].idxmax()

    elif method == 'fbeta':
        # Fβ = (1+β²)·P·R / (β²·P + R)
        b2    = beta ** 2
        fbeta = ((1 + b2) * sweep['precision'] * sweep['sensitivity']) / \
                (b2 * sweep['precision'] + sweep['sensitivity'] + 1e-12)
        sweep['fbeta'] = fbeta
        idx = sweep['fbeta'].idxmax()

    elif method == 'precision_at':
        # Lowest threshold where precision ≥ target (maximises recall subject to precision)
        candidates = sweep[sweep['precision'] >= precision_target]
        if candidates.empty:
            print(f"  [warning] No threshold achieves precision≥{precision_target:.0%}"
                  f" — relaxing to best available")
            idx = sweep['precision'].idxmax()
        else:
            idx = candidates['threshold'].idxmin()  # lowest threshold = highest recall
            idx = candidates.index[candidates['threshold'] == sweep.loc[idx, 'threshold']][0]

    elif method == 'recall_at':
        # Highest threshold where recall ≥ target (maximises precision subject to recall)
        candidates = sweep[sweep['sensitivity'] >= recall_target]
        if candidates.empty:
            print(f"  [warning] No threshold achieves recall≥{recall_target:.0%}"
                  f" — relaxing to best available")
            idx = sweep['sensitivity'].idxmax()
        else:
            idx = candidates['threshold'].idxmax()  # highest threshold = highest precision
            idx = candidates.index[candidates['threshold'] == sweep.loc[idx, 'threshold']][0]

    elif method == 'cost':
        # Minimise total misclassification cost: FP·cost_fp + FN·cost_fn
        sweep['cost'] = sweep['fp'] * cost_fp + sweep['fn'] * cost_fn
        idx = sweep['cost'].idxmin()

    else:
        raise ValueError(f"Unknown method '{method}'. "
                         f"Choose from: auto, youden, f1, f2, fbeta, "
                         f"precision_at, recall_at, cost")

    best = sweep.loc[idx]

    # AUC metrics
    fpr, tpr, _  = roc_curve(y_true, y_proba)
    prec_c, rec_c, _ = precision_recall_curve(y_true, y_proba)
    auc_roc = auc(fpr, tpr)
    auc_pr  = auc(rec_c[::-1], prec_c[::-1])

    result = {
        'method':      method,
        'threshold':   float(best['threshold']),
        'sensitivity': float(best['sensitivity']),
        'specificity': float(best['specificity']),
        'precision':   float(best['precision']),
        'f1':          float(best['f1']),
        'f2':          float(best['f2']),
        'youden':      float(best['youden']),
        'tp':          int(best['tp']),
        'fp':          int(best['fp']),
        'tn':          int(best['tn']),
        'fn':          int(best['fn']),
        'auc_roc':     float(auc_roc),
        'auc_pr':      float(auc_pr),
        '_sweep':      sweep,    # internal — stripped before JSON export
    }
    return result


# ══════════════════════════════════════════════════════════════════
# 3.  FULL REPORT  (all methods + 4-panel plot + CSV + JSON)
# ══════════════════════════════════════════════════════════════════

def threshold_report(
    y_true:    np.ndarray,
    y_proba:   np.ndarray,
    target:    str  = 'psr_filter',
    lm:        str  = 'ablang',
    output_dir: str = None,
    cost_fp:   float = 1.0,
    cost_fn:   float = 3.0,
    show_plot: bool  = True,
) -> pd.DataFrame:
    """
    Run all threshold-selection methods, produce a comparison table,
    a 4-panel diagnostic plot, and save artefacts.

    Cost defaults  (cost_fn=3, cost_fp=1):
      Missing a failing mAb (FN) costs 3× more than a false alarm (FP).
      This reflects the reality that advancing a bad mAb wastes downstream
      resources (SPR, in-vivo, manufacturing) far more than re-screening
      a good mAb that was flagged.
      Adjust cost_fn/cost_fp to match your lab's risk tolerance.

    Outputs saved to output_dir (defaults to MODEL_DIR):
        thresh_report_{target}_{lm}.csv    — full metric sweep
        thresh_report_{target}_{lm}.json   — recommended thresholds per method
        thresh_report_{target}_{lm}.png    — 4-panel diagnostic figure

    Returns
    -------
    pd.DataFrame  — one row per method, all metrics at that threshold
    """
    output_dir = output_dir or MODEL_DIR
    os.makedirs(output_dir, exist_ok=True)

    y_true  = np.asarray(y_true,  dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    # Class statistics for context
    n         = len(y_true)
    pos_rate  = y_true.mean()
    min_rate  = min(pos_rate, 1 - pos_rate)
    print(f"\n{'═'*60}")
    print(f"  Threshold Report  |  target={target}  lm={lm}")
    print(f"  n={n:,}  pos_rate={pos_rate:.1%}  min_class_rate={min_rate:.1%}")
    print(f"{'═'*60}")

    # ── Run all methods ──────────────────────────────────────────
    methods_to_run = [
        ('auto',         dict()),
        ('youden',       dict()),
        ('f1',           dict()),
        ('f2',           dict()),
        ('fbeta',        dict(beta=0.5)),   # precision-oriented
        ('fbeta',        dict(beta=2.0)),   # recall-oriented (same as F2 but explicit)
        ('recall_at',    dict(recall_target=0.90)),
        ('precision_at', dict(precision_target=0.80)),
        ('cost',         dict(cost_fp=cost_fp, cost_fn=cost_fn)),
    ]

    records = []
    sweep_df = None

    for method, kwargs in methods_to_run:
        label = method
        if method == 'fbeta':
            label = f"f{kwargs['beta']}"
        elif method == 'recall_at':
            label = f"recall≥{kwargs['recall_target']:.0%}"
        elif method == 'precision_at':
            label = f"prec≥{kwargs['precision_target']:.0%}"
        elif method == 'cost':
            label = f"cost(fp={cost_fp},fn={cost_fn})"

        r = find_optimal_threshold(y_true, y_proba, method=method, **kwargs)
        sweep_df = r.pop('_sweep')   # keep last sweep for plot
        r['label'] = label
        records.append(r)
        print(
            f"  {label:<22}  thresh={r['threshold']:.3f}"
            f"  sens={r['sensitivity']:.3f}  spec={r['specificity']:.3f}"
            f"  prec={r['precision']:.3f}  F1={r['f1']:.3f}  F2={r['f2']:.3f}"
        )

    df = pd.DataFrame(records).set_index('label')

    # ── 4-panel diagnostic plot ───────────────────────────────────
    fig = plt.figure(figsize=(13, 10))
    fig.patch.set_facecolor('#0f1117')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    ax_roc  = fig.add_subplot(gs[0, 0])
    ax_pr   = fig.add_subplot(gs[0, 1])
    ax_f    = fig.add_subplot(gs[1, 0])
    ax_cost = fig.add_subplot(gs[1, 1])

    panel_style = dict(facecolor='#1a1d27', alpha=1.0)
    tick_kw     = dict(colors='#aab4c8', labelsize=8)
    grid_kw     = dict(color='#2e3347', lw=0.5, alpha=0.7)

    for ax in (ax_roc, ax_pr, ax_f, ax_cost):
        ax.set_facecolor('#1a1d27')
        ax.tick_params(**tick_kw)
        ax.grid(**grid_kw)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2e3347')

    label_kw  = dict(color='#aab4c8', fontsize=8)
    title_kw  = dict(color='#e2e8f0', fontsize=9, fontweight='bold', pad=6)

    # Colour palette per method (matches label names)
    METHOD_COLORS = {
        'auto':                '#f59e0b',
        'youden':              '#38bdf8',
        'f1':                  '#a78bfa',
        'f2':                  '#34d399',
        f'f{0.5}':             '#fb923c',
        f'f{2.0}':             '#86efac',
        f'recall≥90%':         '#f472b6',
        f'prec≥80%':           '#c084fc',
        f'cost(fp={cost_fp},fn={cost_fn})': '#fbbf24',
    }

    # Panel 1 — ROC curve + threshold markers
    fpr_arr, tpr_arr, thr_roc = roc_curve(y_true, y_proba)
    auc_roc = auc(fpr_arr, tpr_arr)
    ax_roc.plot(fpr_arr, tpr_arr, color='#60a5fa', lw=1.8,
                label=f'ROC  AUC={auc_roc:.3f}')
    ax_roc.plot([0, 1], [0, 1], color='#4b5563', lw=0.8, ls='--')
    for _, row in df.iterrows():
        color = METHOD_COLORS.get(row.name, '#94a3b8')
        ax_roc.scatter(
            1 - row['specificity'], row['sensitivity'],
            color=color, s=55, zorder=5,
            label=f"{row.name} t={row['threshold']:.3f}"
        )
    ax_roc.set_xlabel('False Positive Rate', **label_kw)
    ax_roc.set_ylabel('True Positive Rate',  **label_kw)
    ax_roc.set_title('ROC Curve + Threshold Markers', **title_kw)
    ax_roc.legend(fontsize=5.5, loc='lower right',
                  facecolor='#1a1d27', labelcolor='#aab4c8',
                  edgecolor='#2e3347')

    # Panel 2 — Precision-Recall curve + threshold markers
    prec_arr, rec_arr, thr_pr = precision_recall_curve(y_true, y_proba)
    auc_pr = auc(rec_arr[::-1], prec_arr[::-1])
    ax_pr.plot(rec_arr, prec_arr, color='#34d399', lw=1.8,
               label=f'PR  AUC={auc_pr:.3f}')
    ax_pr.axhline(pos_rate, color='#4b5563', lw=0.8, ls='--',
                  label=f'Baseline ({pos_rate:.1%})')
    for _, row in df.iterrows():
        color = METHOD_COLORS.get(row.name, '#94a3b8')
        ax_pr.scatter(
            row['sensitivity'], row['precision'],
            color=color, s=55, zorder=5,
            label=f"{row.name} t={row['threshold']:.3f}"
        )
    ax_pr.set_xlabel('Recall (Sensitivity)', **label_kw)
    ax_pr.set_ylabel('Precision',            **label_kw)
    ax_pr.set_title('Precision-Recall Curve + Markers', **title_kw)
    ax_pr.legend(fontsize=5.5, loc='upper right',
                 facecolor='#1a1d27', labelcolor='#aab4c8',
                 edgecolor='#2e3347')

    # Panel 3 — F1, F2, Youden vs threshold (full sweep)
    t_sweep = sweep_df['threshold'].values
    ax_f.plot(t_sweep, sweep_df['f1'].values,
              color='#a78bfa', lw=1.4, label='F1')
    ax_f.plot(t_sweep, sweep_df['f2'].values,
              color='#34d399', lw=1.4, label='F2')
    ax_f.plot(t_sweep, sweep_df['youden'].values,
              color='#38bdf8', lw=1.4, label='Youden J')
    ax_f.plot(t_sweep, sweep_df['sensitivity'].values,
              color='#f472b6', lw=0.9, ls='--', alpha=0.7, label='Sensitivity')
    ax_f.plot(t_sweep, sweep_df['specificity'].values,
              color='#fb923c', lw=0.9, ls='--', alpha=0.7, label='Specificity')
    # Mark optimal thresholds for key methods
    for lbl_key, metric, color in [
            ('youden', 'youden', '#38bdf8'),
            ('f1',     'f1',     '#a78bfa'),
            ('f2',     'f2',     '#34d399'),
    ]:
        if lbl_key in df.index:
            t_opt = df.loc[lbl_key, 'threshold']
            ax_f.axvline(t_opt, color=color, lw=0.8, ls=':', alpha=0.8)
    ax_f.set_xlabel('Threshold', **label_kw)
    ax_f.set_ylabel('Score',     **label_kw)
    ax_f.set_title('Metrics vs Threshold', **title_kw)
    ax_f.legend(fontsize=6, loc='lower left',
                facecolor='#1a1d27', labelcolor='#aab4c8',
                edgecolor='#2e3347')
    ax_f.set_xlim([t_sweep.min(), t_sweep.max()])

    # Panel 4 — Cost surface + sensitivity/specificity crossover
    total = len(y_true)
    sweep_df['cost_total'] = (
        sweep_df['fp'] * cost_fp + sweep_df['fn'] * cost_fn
    ) / total   # normalised per sample
    ax_cost.plot(t_sweep, sweep_df['cost_total'].values,
                 color='#fbbf24', lw=1.6,
                 label=f'Cost/sample  (FP×{cost_fp}, FN×{cost_fn})')
    cost_lbl = f'cost(fp={cost_fp},fn={cost_fn})'
    if cost_lbl in df.index:
        t_cost = df.loc[cost_lbl, 'threshold']
        cost_val = sweep_df.iloc[
            (sweep_df['threshold'] - t_cost).abs().argsort().iloc[0]
        ]['cost_total']
        ax_cost.scatter(t_cost, cost_val, color='#fbbf24', s=80,
                        zorder=5, label=f'Optimal t={t_cost:.3f}')
    ax_cost.set_xlabel('Threshold',      **label_kw)
    ax_cost.set_ylabel('Norm. Cost',     **label_kw)
    ax_cost.set_title(
        f'Cost Surface  (FN costs {cost_fn}× FP)', **title_kw
    )
    ax_cost.legend(fontsize=6.5, loc='upper left',
                   facecolor='#1a1d27', labelcolor='#aab4c8',
                   edgecolor='#2e3347')
    ax_cost.set_xlim([t_sweep.min(), t_sweep.max()])

    # Super-title
    fig.suptitle(
        f'Threshold Analysis  |  {target}  ·  {lm}\n'
        f'n={n:,}   pos_rate={pos_rate:.1%}   '
        f'AUC-ROC={auc_roc:.3f}   AUC-PR={auc_pr:.3f}',
        color='#e2e8f0', fontsize=10, fontweight='bold', y=0.99,
    )

    stem = f"thresh_report_{target}_{lm}"
    plt.savefig(os.path.join(output_dir, f"{stem}.png"),
                dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()

    # ── Save CSV full sweep ──────────────────────────────────────
    sweep_cols = ['threshold', 'tp', 'fp', 'tn', 'fn',
                  'sensitivity', 'specificity', 'precision',
                  'f1', 'f2', 'youden']
    if 'cost' in sweep_df.columns:
        sweep_cols.append('cost')
    if 'cost_total' in sweep_df.columns:
        sweep_cols.append('cost_total')
    sweep_df[sweep_cols].to_csv(
        os.path.join(output_dir, f"{stem}_sweep.csv"), index=False
    )

    # ── Save JSON summary (strip internal fields) ─────────────────
    json_out = {}
    for lbl, row in df.iterrows():
        json_out[str(lbl)] = {
            k: v for k, v in row.to_dict().items()
            if not k.startswith('_')
        }
    with open(os.path.join(output_dir, f"{stem}.json"), 'w') as f:
        json.dump(json_out, f, indent=2)

    # ── Console summary table ─────────────────────────────────────
    display_cols = ['threshold', 'sensitivity', 'specificity',
                    'precision', 'f1', 'f2']
    print(f"\n{'─'*72}")
    print(f"  {'Method':<24} {'Thresh':>7}  {'Sens':>6}  {'Spec':>6}"
          f"  {'Prec':>6}  {'F1':>6}  {'F2':>6}")
    print(f"{'─'*72}")
    for lbl, row in df[display_cols].iterrows():
        print(f"  {lbl:<24} {row['threshold']:>7.3f}  "
              f"{row['sensitivity']:>6.3f}  {row['specificity']:>6.3f}  "
              f"{row['precision']:>6.3f}  {row['f1']:>6.3f}  {row['f2']:>6.3f}")
    print(f"{'─'*72}")
    print(f"\n  Recommendation for your datasets:")
    print(f"    IPI PSR  (~52/48%)  → use  youden  "
          f"t={df.loc['youden','threshold']:.3f}"  if 'youden' in df.index else "")
    print(f"    IPI SEC  (~79/21%)  → use  f2      "
          f"t={df.loc['f2','threshold']:.3f}"       if 'f2'     in df.index else "")
    print(f"    DS1      (~53/47%)  → use  f1      "
          f"t={df.loc['f1','threshold']:.3f}"       if 'f1'     in df.index else "")
    print(f"\n  Saved:")
    print(f"    {stem}.png")
    print(f"    {stem}_sweep.csv")
    print(f"    {stem}.json")

    return df


# ══════════════════════════════════════════════════════════════════
# 4.  CROSS-VALIDATED THRESHOLD STABILITY
#     Use fold_preds CSV from kfold_validation() to measure how
#     stable the optimal threshold is across folds.
# ══════════════════════════════════════════════════════════════════

def threshold_stability_from_folds(
    fold_preds_csv: str,
    method:    str   = 'auto',
    target:    str   = 'psr_filter',
    lm:        str   = 'ablang',
    output_dir: str  = None,
    cost_fp:   float = 1.0,
    cost_fn:   float = 3.0,
) -> dict:
    """
    Load fold_preds CSV (written by kfold_validation) and compute
    the optimal threshold on each fold independently.

    This tells you:
      - How stable the threshold is (std across folds)
      - Whether 0.5 is in the optimal range
      - A pooled threshold from all OOF predictions combined

    Returns
    -------
    dict with keys:
        pooled_threshold   — from all OOF predictions combined
        mean_threshold     — mean of per-fold optima
        std_threshold      — std  of per-fold optima
        fold_thresholds    — list of per-fold optimal thresholds
        pooled_result      — full result dict from find_optimal_threshold
        fold_results       — list of per-fold result dicts
    """
    output_dir = output_dir or MODEL_DIR
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(fold_preds_csv)
    required = {'fold', 'true', 'prob'}
    if not required.issubset(df.columns):
        raise ValueError(
            f"fold_preds CSV must contain columns: {required}. "
            f"Found: {set(df.columns)}"
        )

    print(f"\n[threshold_stability]  {os.path.basename(fold_preds_csv)}")
    print(f"  Folds: {sorted(df['fold'].unique())}  |  method={method}")

    # Per-fold optimal threshold
    fold_results    = []
    fold_thresholds = []

    for fold_id, fold_df in df.groupby('fold'):
        y_f = fold_df['true'].values
        p_f = fold_df['prob'].values
        if len(np.unique(y_f)) < 2:
            print(f"  Fold {fold_id}: skipped (one class only)")
            continue
        r = find_optimal_threshold(
            y_f, p_f, method=method,
            cost_fp=cost_fp, cost_fn=cost_fn,
        )
        r.pop('_sweep', None)
        r['fold'] = fold_id
        fold_results.append(r)
        fold_thresholds.append(r['threshold'])
        print(
            f"  Fold {fold_id:<3}  thresh={r['threshold']:.3f}"
            f"  sens={r['sensitivity']:.3f}  spec={r['specificity']:.3f}"
            f"  F1={r['f1']:.3f}  F2={r['f2']:.3f}"
        )

    # Pooled OOF threshold
    pooled_result = find_optimal_threshold(
        df['true'].values, df['prob'].values,
        method=method, cost_fp=cost_fp, cost_fn=cost_fn,
    )
    pooled_result.pop('_sweep', None)

    mean_t = float(np.mean(fold_thresholds))
    std_t  = float(np.std(fold_thresholds))

    print(f"\n  Per-fold thresholds : {[f'{t:.3f}' for t in fold_thresholds]}")
    print(f"  Mean ± std          : {mean_t:.3f} ± {std_t:.3f}")
    print(f"  Pooled OOF thresh   : {pooled_result['threshold']:.3f}")

    if std_t < 0.05:
        stability = "✓ STABLE   — safe to use pooled threshold in production"
    elif std_t < 0.10:
        stability = "△ MODERATE — consider pooled OOF threshold"
    else:
        stability = "✗ UNSTABLE — model is sensitive to data split; inspect folds"
    print(f"  Stability           : {stability}")

    # ── Stability plot ───────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor('#0f1117')
    for ax in (ax1, ax2):
        ax.set_facecolor('#1a1d27')
        ax.tick_params(colors='#aab4c8', labelsize=8)
        ax.grid(color='#2e3347', lw=0.5, alpha=0.7)
        for sp in ax.spines.values():
            sp.set_edgecolor('#2e3347')

    fold_ids = [r['fold'] for r in fold_results]
    # Left: threshold per fold
    ax1.bar(fold_ids, fold_thresholds, color='#60a5fa', alpha=0.8, width=0.6)
    ax1.axhline(mean_t,               color='#f59e0b', lw=1.5, ls='--',
                label=f'Mean {mean_t:.3f}')
    ax1.axhline(pooled_result['threshold'], color='#34d399', lw=1.5, ls='-.',
                label=f'Pooled OOF {pooled_result["threshold"]:.3f}')
    ax1.axhline(0.5, color='#f87171', lw=1.0, ls=':',
                label='Default 0.5')
    ax1.fill_between(
        [min(fold_ids) - 0.5, max(fold_ids) + 0.5],
        mean_t - std_t, mean_t + std_t,
        color='#f59e0b', alpha=0.15, label=f'±1σ={std_t:.3f}'
    )
    ax1.set_xlabel('Fold', color='#aab4c8', fontsize=8)
    ax1.set_ylabel('Optimal Threshold', color='#aab4c8', fontsize=8)
    ax1.set_title('Per-Fold Threshold Stability', color='#e2e8f0',
                  fontsize=9, fontweight='bold')
    ax1.legend(fontsize=6.5, facecolor='#1a1d27',
               labelcolor='#aab4c8', edgecolor='#2e3347')

    # Right: sens/spec/F1/F2 at per-fold optimal threshold
    metrics = ['sensitivity', 'specificity', 'f1', 'f2']
    colors  = ['#f472b6', '#38bdf8', '#a78bfa', '#34d399']
    x       = np.arange(len(fold_ids))
    w       = 0.18
    for i, (m, c) in enumerate(zip(metrics, colors)):
        vals = [r[m] for r in fold_results]
        ax2.bar(x + i * w, vals, width=w, color=c, alpha=0.85, label=m)
    ax2.set_xticks(x + 1.5 * w)
    ax2.set_xticklabels([f'F{f}' for f in fold_ids], fontsize=7,
                        color='#aab4c8')
    ax2.set_ylim([0, 1.05])
    ax2.set_ylabel('Score', color='#aab4c8', fontsize=8)
    ax2.set_title('Metrics at Per-Fold Optimal Threshold',
                  color='#e2e8f0', fontsize=9, fontweight='bold')
    ax2.legend(fontsize=6.5, facecolor='#1a1d27',
               labelcolor='#aab4c8', edgecolor='#2e3347')

    fig.suptitle(
        f'Threshold Stability  |  {target}  ·  {lm}  ·  method={method}\n'
        f'Pooled={pooled_result["threshold"]:.3f}  '
        f'Mean={mean_t:.3f}±{std_t:.3f}',
        color='#e2e8f0', fontsize=9, fontweight='bold',
    )
    plt.tight_layout()
    stab_path = os.path.join(
        output_dir, f"thresh_stability_{target}_{lm}_{method}.png"
    )
    plt.savefig(stab_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Plot saved → {stab_path}")

    return {
        'pooled_threshold': pooled_result['threshold'],
        'mean_threshold':   mean_t,
        'std_threshold':    std_t,
        'fold_thresholds':  fold_thresholds,
        'pooled_result':    pooled_result,
        'fold_results':     fold_results,
    }


# ══════════════════════════════════════════════════════════════════
# 5.  CHECKPOINT UPDATER
#     Embed recommended threshold into BEST_*.pt metadata so
#     predict_developability.py can use it directly.
# ══════════════════════════════════════════════════════════════════

def embed_threshold_in_checkpoint(
    checkpoint_path: str,
    threshold:       float,
    method:          str = 'auto',
):
    """
    Load a BEST_*.pt or FINAL_*.pt checkpoint, add
    'recommended_threshold' and 'threshold_method' fields,
    and save back in-place.

    Usage after kfold:
        stability = threshold_stability_from_folds(fold_preds_csv, ...)
        embed_threshold_in_checkpoint(
            best_ckpt_path,
            threshold = stability['pooled_threshold'],
            method    = 'auto',
        )
    """
    import torch
    payload = torch.load(checkpoint_path, map_location='cpu',
                         weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(
            "Only new-style checkpoints (dict with 'state_dict') are supported."
        )
    payload['recommended_threshold'] = threshold
    payload['threshold_method']      = method
    torch.save(payload, checkpoint_path)
    print(f"[embed_threshold] {checkpoint_path}")
    print(f"  recommended_threshold = {threshold:.4f}  (method={method})")


# ══════════════════════════════════════════════════════════════════
# 6.  CONVENIENCE: RUN FULL PIPELINE FROM FOLD_PREDS CSV
# ══════════════════════════════════════════════════════════════════

def run_full_threshold_pipeline(
    fold_preds_csv:  str,
    target:          str   = 'psr_filter',
    lm:              str   = 'ablang',
    best_ckpt_path:  str   = None,
    output_dir:      str   = None,
    cost_fp:         float = 1.0,
    cost_fn:         float = 3.0,
):
    """
    One-call convenience wrapper:
      1. Load fold_preds CSV
      2. Run threshold_report  (all methods, 4-panel plot)
      3. Run threshold_stability_from_folds (per-fold consistency)
      4. Embed pooled OOF threshold into checkpoint (if path given)

    Typical call after kfold_validation():
        run_full_threshold_pipeline(
            fold_preds_csv = f"{MODEL_DIR}/fold_preds_psr_filter_ablang_transformer_lm_k10.csv",
            target         = 'psr_filter',
            lm             = 'ablang',
            best_ckpt_path = f"{MODEL_DIR}/BEST_psr_filter_ablang_transformer_lm_k10_fold7.pt",
            cost_fn        = 3.0,
        )
    """
    output_dir = output_dir or MODEL_DIR
    df_preds = pd.read_csv(fold_preds_csv)

    # 1 — Full threshold report (all OOF predictions pooled)
    threshold_report(
        df_preds['true'].values,
        df_preds['prob'].values,
        target=target, lm=lm,
        output_dir=output_dir,
        cost_fp=cost_fp, cost_fn=cost_fn,
    )

    # 2 — Per-fold stability analysis
    stability = threshold_stability_from_folds(
        fold_preds_csv, method='auto',
        target=target, lm=lm,
        output_dir=output_dir,
        cost_fp=cost_fp, cost_fn=cost_fn,
    )

    # 3 — Embed into checkpoint
    if best_ckpt_path and os.path.exists(best_ckpt_path):
        embed_threshold_in_checkpoint(
            best_ckpt_path,
            threshold=stability['pooled_threshold'],
            method='auto',
        )

    return stability
