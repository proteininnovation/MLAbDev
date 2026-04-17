"""
utils/developability_correlation.py
─────────────────────────────────────────────────────────────────────────────
IPI Antibody Developability Prediction Platform — Correlation Analysis Module
─────────────────────────────────────────────────────────────────────────────
Correlates IPI model prediction scores against experimental developability
assays (PSR, SEC, HIC, AC-SINS, SPR, ELISA, Tm, viscosity, titer ...).

WHAT IT PRODUCES
────────────────
  1. Scatter plots       IPI score vs each assay  (Spearman ρ + Pearson r per panel)
  2. Correlation CSV     All ρ / r values → *_results.csv
  3. Assay heatmaps      Pairwise Spearman ρ AND Pearson r between assay columns
  4. Full heatmaps       Assay + ML scores together (dashed separator)
  5. Boxplots            Assay distributions split FAIL / PASS with T-test p-value
  6. t-SNE (target)      Assay-space clustering coloured by true label
  7. t-SNE (ML labels)   Same embedding coloured by each ML model prediction
                         — includes both *_label (0.5) and *_optimallabel columns

AUTO-DETECTION
──────────────
  Model score columns — auto-detected: *_score, transformer_lm*, xgboost*, cnn* ...
  Target column       — auto-detected: psr_filter, sec_filter, label, class, target
                        Use --target to override.

USAGE
─────
    python utils/developability_correlation.py \
        --files   <pred_file.xlsx> [<pred_file2.xlsx> ...] \
        --assay   <col1> [<col2> ...] \
        [--target <label_col>] \
        [--out    <output_stem>] \
        [--title  "Custom figure title"] \
        [--xlabel "Custom x-axis label"] \
        [--logit-trans | --logit_trans] \
        [--list-scores]

    Multi-word column names: use quotes  → --assay "PSR SMP Score (0-1)"
    Multiple columns: space-separated   → --assay psr_norm_dna psr_norm_smp
    Or comma-separated                  → --assay psr_norm_dna,psr_norm_smp

EXAMPLES
────────
  ── PSR (polyreactivity) ─────────────────────────────────────────────────────

  # Validation set: IPI PSR vs normalised PSR panel
  python utils/developability_correlation.py \
      --files  test/ipi_psr_val_elisa_pred_psr_filter_all_transformer_lm_ipi_psr_trainset_train.xlsx \
      --assay  psr_norm_dna psr_norm_avidin psr_norm_insulin psr_norm_smp psr_norm_mean \
      --target psr_filter \
      --title  "IPI PSR model vs normalised PSR panel" \
      --out    results/ipi_psr_correlation

  # With logit transform (improves Pearson r; Spearman ρ unchanged)
  python utils/developability_correlation.py \
      --files  test/ipi_psr_val_elisa_pred_psr_filter_all_transformer_lm_ipi_psr_trainset_train.xlsx \
      --assay  psr_norm_dna psr_norm_avidin psr_norm_insulin psr_norm_smp \
      --target psr_filter \
      --logit-trans \
      --out    results/ipi_psr_correlation_logit

  # MiSeq antibodies vs PSR panel
  python utils/developability_correlation.py \
      --files  test/ipiab_Miseq92To98_pred_psr_filter_ablang_transformer_lm_ipi_psr_trainset.xlsx \
      --assay  psr_norm_dna psr_norm_avidin psr_norm_insulin psr_norm_smp \
      --target psr_filter \
      --title  "IPI PSR — MiSeq antibodies (n=1374)" \
      --out    results/Miseq_psr_correlation

  # External benchmark: GDPa3 (n=80)
  python utils/developability_correlation.py \
      --files  test/GDPa3_pred_psr_filter_ablang_transformer_lm_ipi_psr_trainset.xlsx \
               test/GDPa3_pred_psr_filter_antiberta2_transformer_lm_ipi_psr_trainset.xlsx \
               test/GDPa3_pred_psr_filter_igbert_transformer_lm_ipi_psr_trainset.xlsx \
      --assay  polyreactivity_prscore_ova_avg polyreactivity_prscore_cho_avg \
      --target psr_filter \
      --title  "GDPa3 (n=80) — IPI PSR vs PROPHET-Ab polyreactivity" \
      --out    results/GDPa3_psr_correlation

  # External benchmark: Jain2017 (n=137) — full developability panel
  python utils/developability_correlation.py \
      --files  test/Jain2017_pred_psr_filter_ablang_transformer_lm_ipi_psr_trainset.xlsx \
      --assay  "PSR  SMP Score (0-1)" "BVP ELISA" ELISA \
               "HIC Retention Time (Min)a" "Fab Tm by DSF (°C)" "HEK Titer (mg/L)" \
      --target psr_filter \
      --title  "Jain2017 (n=137) — IPI PSR vs full developability panel" \
      --out    results/Jain2017_psr_correlation

  ── SEC (size exclusion chromatography) ──────────────────────────────────────

  python utils/developability_correlation.py \
      --files  test/ipi_sec_val_pred_sec_filter_all_transformer_lm_ipi_sec_trainset_train.xlsx \
      --assay  sec_monomer_pct sec_aggregate_pct sec_hm_pct \
      --target sec_filter \
      --xlabel "IPI SEC model score  (P(Pass))" \
      --out    results/ipi_sec_correlation

  ── HIC (hydrophobicity interaction chromatography) ──────────────────────────

  python utils/developability_correlation.py \
      --files  test/dataset_hic_pred_all.xlsx \
      --assay  hic_retention_time "HIC Retention Time (Min)a" \
      --target hic_filter \
      --out    results/hic_correlation

  ── AC-SINS (self-association) ────────────────────────────────────────────────

  python utils/developability_correlation.py \
      --files  test/dataset_acsins_pred_all.xlsx \
      --assay  "SGAC-SINS AS100 ((NH4)2SO4 mM)" "AC-SINS delta_lambda" \
      --target acsins_filter \
      --out    results/acsins_correlation

  ── SPR / affinity ────────────────────────────────────────────────────────────

  python utils/developability_correlation.py \
      --files  test/dataset_spr_pred_all.xlsx \
      --assay  kd_nM kon_1_Ms koff_1_s \
      --title  "IPI model vs SPR binding kinetics" \
      --out    results/spr_correlation

  ── Tm / thermostability ──────────────────────────────────────────────────────

  python utils/developability_correlation.py \
      --files  test/dataset_tm_pred_all.xlsx \
      --assay  "Fab Tm by DSF (°C)" fab_tm ch2_tm \
      --target sec_filter \
      --out    results/tm_correlation

  ── Full multi-assay panel ────────────────────────────────────────────────────

  python utils/developability_correlation.py \
      --files  test/full_panel_pred_all.xlsx \
      --assay  psr_norm_mean sec_monomer_pct hic_retention_time \
               "SGAC-SINS AS100 ((NH4)2SO4 mM)" "Fab Tm by DSF (°C)" "HEK Titer (mg/L)" \
      --target psr_filter \
      --title  "IPI models vs full developability panel" \
      --out    results/full_panel_correlation

  ── Compare multiple LMs ──────────────────────────────────────────────────────

  python utils/developability_correlation.py \
      --files  test/pred_ablang.xlsx test/pred_antiberta2.xlsx \
               test/pred_igbert.xlsx test/pred_antiberta2_cssp.xlsx \
      --assay  psr_norm_dna psr_norm_smp \
      --target psr_filter \
      --title  "PSR — ablang vs antiberta2 vs igbert vs antiberta2-cssp" \
      --out    results/multi_lm_psr_comparison

  ── Utility ───────────────────────────────────────────────────────────────────

  # Dry run — list detected score columns, no analysis
  python utils/developability_correlation.py \
      --files  test/my_predictions.xlsx \
      --assay  psr_norm_dna \
      --list-scores

OUTPUTS (all saved next to --out stem)
──────────────────────────────────────
  *_results.csv                  Spearman ρ + Pearson r, all models × assays
  *.tiff + *_preview.png         Scatter plots (model score vs each assay)

  *_heatmap_assay_spearman.tiff  Assay × assay  Spearman ρ
  *_heatmap_assay_pearson.tiff   Assay × assay  Pearson r
  *_heatmap_all_spearman.tiff    Assays + ML scores  Spearman ρ
  *_heatmap_all_pearson.tiff     Assays + ML scores  Pearson r

  *_boxplot_{target}.tiff        Assay distributions  FAIL vs PASS  (T-test p-value)

  *_tsne_{target}.tiff           t-SNE on assay features, coloured by true label
  *_tsne_ml_labels.tiff          t-SNE coloured by ML prediction labels
                                   — *_label (threshold = 0.5)
                                   — *_optimallabel (optimal threshold from kfold)
                                   — *_costlabel, *_f1label if present

  All TIFF files also saved as *_preview.png (150 DPI) for quick viewing.

ARGUMENTS
─────────
  --files        Required. Prediction .xlsx or .csv file(s).
                 Multiple files merged by BARCODE column.
  --assay        Required. Assay column name(s).
  --target       Label column (e.g. psr_filter, sec_filter).
                 Auto-detected if omitted. Required for boxplots + t-SNE.
  --out          Output file stem. Auto-derived from input filename if omitted.
  --title        Custom figure title.
  --xlabel       Custom x-axis label for scatter plots.
  --logit-trans  Apply logit transform to scores before correlating.
  --logit_trans  Alias for --logit-trans (both accepted).
  --list-scores  Print detected score columns and exit (no analysis).

DEPENDENCIES
────────────
  pip install pandas openpyxl scipy matplotlib numpy scikit-learn
"""

import os
import re
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats


# ── Nature Biotechnology style ────────────────────────────────────────────────
_MM         = 1 / 25.4
DOUBLE_COL  = 183 * _MM
FONT_FAMILY = 'Arial'
SIZE_PANEL  = 8
SIZE_AXIS   = 7
SIZE_TICK   = 6
SIZE_ANNOT  = 5.5
LW          = 0.6
ALPHA       = 0.65
MARKER_SIZE = 16

COLORS = ['#4C9BE8', '#F28C38', '#2ca02c', '#9467bd', '#d62728',
          '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#aec7e8']


# ── Logit transformation ─────────────────────────────────────────────────────

def logit_transform(scores: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Transform P(Pass) in [0,1] to logit space: log(p / (1-p)).

    Binary classification training compresses scores at the tails (sigmoid
    saturation). Logit stretches them back out, improving Pearson r and scatter
    plot spread without affecting Spearman rho (rank-invariant).
    eps clamps to avoid log(0) — default keeps values in [-13.8, 13.8].
    """
    p = np.clip(scores, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


# ── Score column auto-detection ───────────────────────────────────────────────
#
# Priority-ordered patterns — first match wins.
# IPI model score columns are always probabilities in [0, 1].
#
_SCORE_PATTERNS = [
    ('transformer_lm',      r'transformer_lm.*score'),
    ('transformer_onehot',  r'transformer_onehot.*score'),
    ('transformer',         r'transformer.*score'),
    ('xgboost',             r'(xgboost|xgb).*score'),
    # RF: new convention rf_{lm}_{db}_score  OR old pred_proba
    ('random_forest',       r'^rf_.*_score$'),
    ('random_forest',       r'(rf|random_forest).*pred.*proba'),
    ('cnn',                 r'cnn.*score'),
    ('ensemble',            r'(ensemble|mean).*score'),
    ('pred_proba',          r'pred.*proba'),
    ('score',               r'(?<![a-z_])score$'),         # bare *_score fallback
]

# Always exclude binary label / flag columns
_EXCLUDE_PATTERNS = [
    r'_label$', r'_pred$', r'_filter$', r'_flag$',
    r'optimallabel', r'costlabel', r'f1label', r'f2label',
    r'f05label', r'sens90label', r'_bvp_filter', r'_elisa_filter',
]


def _is_excluded(col: str) -> bool:
    return any(re.search(p, col.lower()) for p in _EXCLUDE_PATTERNS)


def detect_all_score_columns(df: pd.DataFrame) -> list:
    """
    Find all IPI model score columns (numeric, in [0,1], not a binary label).
    Returns list of (score_col, short_label).
    """
    found = []
    seen  = set()

    for col in df.columns:
        if col in seen or _is_excluded(col):
            continue

        # Must be numeric and in [0, 1] range (probability score)
        try:
            vals = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(vals) == 0:
                continue
            if vals.min() < -0.01 or vals.max() > 1.01:
                continue
        except Exception:
            continue

        col_l = col.lower()
        for label_prefix, pattern in _SCORE_PATTERNS:
            if re.search(pattern, col_l):
                short = _make_label(col, label_prefix)
                found.append((col, short))
                seen.add(col)
                break

    return found


def _make_label(col: str, prefix: str) -> str:
    """
    Build a short readable label from a score column name.
    Supports new naming: {model}_{lm}_{db}_score
      e.g. rf_antiberta2-cssp_ipi_psr_trainset_score
           → random_forest (antiberta2-cssp)
    And also:
      rf_kmer_*, rf_biophysical_*, rf_onehot_hcdr3_*, rf_onehot_vh_*, rf_onehot_*
    """
    col_l = col.lower()

    # ── New convention: extract LM from position after model prefix ───────────
    _lm_from_col = None
    for _model_pfx in ('rf_', 'transformer_lm_', 'transformer_onehot_',
                       'xgboost_', 'cnn_'):
        if col_l.startswith(_model_pfx):
            rest = col_l[len(_model_pfx):]
            # LM ends before _ipi_ or _train or known db markers
            for sep in ('_ipi_', '_train', '_val', '_test'):
                idx = rest.find(sep)
                if idx > 0:
                    _lm_from_col = rest[:idx]
                    break
            if _lm_from_col is None and '_' in rest:
                _lm_from_col = rest.rsplit('_', 1)[0]  # fallback
            break

    if _lm_from_col:
        # Clean up: onehot_hcdr3 → onehot_hcdr3, biophysical → biophysical
        return f"{prefix}\n({_lm_from_col})"

    # ── Legacy: scan for known LM keywords ───────────────────────────────────
    lm_hits = []
    for lm in ['antiberta2-cssp', 'antiberta2', 'antiberty', 'ablang',
                'esm2', 'esm-2', 'igbert', 'igt5',
                'onehot_hcdr3', 'onehot_vh', 'onehot',
                'biophysical', 'kmer']:
        if lm in col_l:
            lm_hits.append(lm)

    lm_hits = [
        lm for lm in lm_hits
        if not any(lm != other and lm in other for other in lm_hits)
    ]

    parts = []
    if lm_hits:
        parts.append(', '.join(lm_hits))

    return f'{prefix}\n({", ".join(parts)})' if parts else prefix


# ── File loading ──────────────────────────────────────────────────────────────

def _preferred_sheet(xl: pd.ExcelFile) -> str:
    preferred = ['Predictions_with_OptimalLabels', 'Sheet1', 'Predictions',
                 'Results', 'Data']
    for p in preferred:
        if p in xl.sheet_names:
            return p
    return xl.sheet_names[0]


def load_file(path: str, assay_cols: list,
              list_scores: bool = False) -> list:
    """
    Load one file, detect IPI score columns, validate assay columns.
    Returns list of (df_sub, score_col, model_label, assay_cols_present).
    """
    # Support both Excel and CSV
    if path.lower().endswith('.csv'):
        df    = pd.read_csv(path)
        sheet = 'CSV'
    else:
        xl    = pd.ExcelFile(path)
        sheet = _preferred_sheet(xl)
        df    = xl.parse(sheet)

    fname = os.path.basename(path)
    print(f"\n  File : {fname}  ({len(df)} rows, sheet='{sheet}')")

    # ── Validate assay columns ────────────────────────────────────────────────
    assay_present = []
    for ac in assay_cols:
        if ac in df.columns:
            assay_present.append(ac)
        else:
            # Case-insensitive fallback
            match = [c for c in df.columns if c.lower() == ac.lower()]
            if match:
                assay_present.append(match[0])
                if match[0] != ac:
                    print(f"    [info] '{ac}' matched as '{match[0]}'")
            else:
                print(f"    [warn] Assay column '{ac}' not found — skipping")

    if not assay_present:
        print(f"    [error] No assay columns found — skipping file")
        return []

    # ── Detect score columns ──────────────────────────────────────────────────
    score_entries = detect_all_score_columns(df)

    if not score_entries:
        print(f"    [warn] No IPI score columns detected")
        print(f"    Available columns: {list(df.columns)}")
        return []

    print(f"    Assay cols  : {assay_present}")
    print(f"    Score cols detected ({len(score_entries)}):")
    for sc, lbl in score_entries:
        n_valid = df[[sc] + assay_present].dropna().shape[0]
        print(f"      • {sc}")
        print(f"        → label='{lbl.replace(chr(10),' ')}'  n_valid={n_valid}")

    if list_scores:
        return []

    # ── Build one entry per score column ─────────────────────────────────────
    entries = []
    for sc, model_label in score_entries:
        sub = df[[sc] + assay_present].dropna()
        if len(sub) < 5:
            print(f"    [skip] {sc} — only {len(sub)} non-null rows")
            continue
        entries.append((sub.copy(), sc, model_label, assay_present))

    return entries


# ── Correlation computation ───────────────────────────────────────────────────

def compute_correlations(df: pd.DataFrame, score_col: str,
                         model_label: str, assay_cols: list,
                         file_name: str) -> pd.DataFrame:
    rows = []
    for ac in assay_cols:
        if ac not in df.columns:
            continue
        if ac == score_col:
            continue   # skip self-correlation
        # Guard: skip if assay col is itself a ML score col (wrong usage of --assay)
        _is_score_col = (ac.endswith('_score') and
                         any(ac.startswith(m) for m in
                             ('rf_','transformer_','xgboost_','cnn_')))
        if _is_score_col:
            continue
        try:
            # Use .values to avoid duplicate-column issues
            _x_vals = pd.to_numeric(df[score_col], errors='coerce')
            _y_vals = pd.to_numeric(df[ac],         errors='coerce')
            # If duplicate column names, take first occurrence
            if isinstance(_x_vals, pd.DataFrame):
                _x_vals = _x_vals.iloc[:, 0]
            if isinstance(_y_vals, pd.DataFrame):
                _y_vals = _y_vals.iloc[:, 0]
            sub = pd.DataFrame({'x': _x_vals, 'y': _y_vals}).dropna()
        except Exception as _se:
            print(f"    [warn] Cannot extract {score_col} vs {ac}: {_se}")
            continue
        if len(sub) < 5:
            continue
        x = sub['x'].values.flatten()
        y = sub['y'].values.flatten()
        if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
            print(f"    [warn] Shape mismatch {score_col} vs {ac}: "
                  f"x={x.shape} y={y.shape} — skipping")
            continue
        sp, p_sp = stats.spearmanr(x, y)
        pe, p_pe = stats.pearsonr(x, y)
        sig = ('***' if p_sp < 0.001 else
               '**'  if p_sp < 0.01  else
               '*'   if p_sp < 0.05  else 'ns')
        rows.append({
            'file':         file_name,
            'model':        model_label.replace('\n', ' '),
            'score_col':    score_col,
            'assay_col':    ac,
            'n':            len(sub),
            'spearman_rho': round(sp, 4),
            'spearman_p':   round(p_sp, 6),
            'significance': sig,
            'pearson_r':    round(pe, 4),
            'pearson_p':    round(p_pe, 6),
        })
    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────

def set_nature_style():
    plt.rcParams.update({
        'font.family':      'sans-serif',
        'font.sans-serif':  [FONT_FAMILY, 'Helvetica', 'DejaVu Sans'],
        'font.size':         SIZE_TICK,
        'axes.labelsize':    SIZE_AXIS,
        'axes.titlesize':    SIZE_AXIS,
        'axes.linewidth':    LW,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'xtick.labelsize':   SIZE_TICK,
        'ytick.labelsize':   SIZE_TICK,
        'xtick.major.width': LW,
        'ytick.major.width': LW,
        'xtick.major.size':  2.5,
        'ytick.major.size':  2.5,
        'xtick.direction':   'out',
        'ytick.direction':   'out',
        'legend.fontsize':   SIZE_TICK,
        'legend.frameon':    False,
        'pdf.fonttype':      42,
        'ps.fonttype':       42,
    })


def _scatter_panel(ax, x, y, color, ylabel, panel_letter,
                   xlabel: str, is_bottom_row: bool):
    sp, p_sp = stats.spearmanr(x, y)

    ax.scatter(x, y, color=color, alpha=ALPHA, s=MARKER_SIZE, lw=0, zorder=3)

    m, b = np.polyfit(x, y, 1)
    xfit = np.linspace(x.min(), x.max(), 100)
    ax.plot(xfit, m * xfit + b, color='#333333', lw=0.8, ls='--', zorder=4)

    if is_bottom_row:
        ax.set_xlabel(xlabel, fontsize=SIZE_AXIS, labelpad=3)
    ax.set_ylabel(ylabel, fontsize=SIZE_AXIS, labelpad=3)
    ax.tick_params(labelsize=SIZE_TICK, length=2.5)

    sig   = ('***' if p_sp < 0.001 else '**' if p_sp < 0.01 else
             '*'   if p_sp < 0.05  else 'ns')
    p_str = 'p<0.001' if p_sp < 0.001 else f'p={p_sp:.3f}'
    ax.text(0.97, 0.97,
            f'ρ = {sp:+.3f}  {sig}\n{p_str}  n={len(x):,}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=SIZE_ANNOT, color='#333333',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', lw=0.5))

    ax.text(-0.18, 1.12, panel_letter,
            transform=ax.transAxes,
            fontsize=SIZE_PANEL, fontweight='bold', va='top', ha='left')

    # Auto x-axis limits: if scores are [0,1] → fix limits; else auto
    xmin, xmax = x.min(), x.max()
    if xmin >= -0.01 and xmax <= 1.01:
        ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.2, lw=0.4)


def build_figure(all_entries: list, assay_cols: list,
                 title: str, xlabel: str) -> plt.Figure:
    """
    Rows = one per model entry.
    Cols = one per assay column.
    all_entries: list of (df, score_col, model_label, assay_present, color)
    """
    set_nature_style()

    n_rows = len(all_entries)
    n_cols = len(assay_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(DOUBLE_COL, 0.36 * DOUBLE_COL * n_rows),
        squeeze=False,
    )

    letters = list('abcdefghijklmnopqrstuvwxyz'
                   'ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    for ri, (df, sc, model_label, assay_present, color) in enumerate(all_entries):
        is_bottom = (ri == n_rows - 1)
        for ci, ac in enumerate(assay_cols):
            ax = axes[ri, ci]

            if ac not in df.columns:
                ax.set_visible(False)
                continue

            sub = df[[sc, ac]].dropna()
            if len(sub) < 5:
                ax.set_visible(False)
                continue

            letter = letters[ri * n_cols + ci] if ri * n_cols + ci < len(letters) else '?'
            _scatter_panel(
                ax, sub[sc].values, sub[ac].values,
                color, ac, letter, xlabel, is_bottom,
            )

            if ci == 0:
                ax.set_title(
                    model_label,
                    fontsize=SIZE_AXIS - 0.5, loc='left',
                    color=color, fontweight='bold', pad=4,
                )

    fig.suptitle(title, fontsize=SIZE_AXIS, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig



# ── Assay type detection ──────────────────────────────────────────────────────

# Maps keywords in column names → short assay tag used in output filenames
_ASSAY_TAG_RULES = [
    (['psr', 'polyreactivity', 'ova', 'cho', 'smp', 'bvp', 'elisa',
      'avidin', 'insulin', 'dna', 'scp'],   'psr_elisa'),
    (['sec', 'monomer', 'aggregate', 'dls'],  'sec'),
    (['hic', 'hydrophob', 'retention'],       'hic'),
    (['acsins', 'ac_sins', 'self_assoc', 'self-assoc', 'ac-sins'], 'acsins'),
    (['spr', 'kd_', 'kon_', 'koff_', 'affinity'], 'spr'),
    (['tm', 'thermostab', 'dsf', 'melting'],  'tm'),
    (['viscos', 'viscosity'],                 'viscosity'),
    (['titer', 'expression', 'yield'],        'titer'),
]


def _detect_assay_tag(assay_cols: list) -> str:
    """
    Infer a short assay tag from the assay column names.
    Returns the first matching tag, or 'assay' as fallback.
    """
    combined = " ".join(assay_cols).lower()
    for keywords, tag in _ASSAY_TAG_RULES:
        if any(kw in combined for kw in keywords):
            return tag
    return "assay"


def _auto_out_stem(files: list, assay_cols: list) -> str:
    """
    Derive output stem from the first input file:
        {parent_dir}/{input_stem}_{assay_tag}_correlation
    e.g. test/manuscript/GDPa3_pred_psr_filter_all_transformer_lm
         → test/manuscript/GDPa3_pred_psr_filter_all_transformer_lm_psr_elisa_correlation
    """
    p   = Path(files[0])
    tag = _detect_assay_tag(assay_cols)
    return str(p.parent / f"{p.stem}_{tag}_correlation")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(files: list, assay_cols: list, out: str = None,
        title: str = None, xlabel: str = None,
        logit_trans: bool = False,
        list_scores: bool = False):

    # Auto-derive output stem if not provided
    if out is None:
        out = _auto_out_stem(files, assay_cols)
    if logit_trans:
        print(f"  [logit] Score transformation ON: logit(p) = log(p / (1-p))")

    W = 70
    print(f"\n{'═'*W}")
    print(f"  IPI Developability Correlation Analysis")
    print(f"  Assay columns : {assay_cols}")
    print(f"  Output stem   : {out}")
    print(f"{'─'*W}")

    all_entries  = []   # (df, score_col, model_label, assay_present, color)
    all_corr_dfs = []
    color_idx    = 0

    seen_score_cols = set()   # guard against duplicate score columns across files

    for fpath in files:
        entries = load_file(fpath, assay_cols, list_scores=list_scores)
        if list_scores:
            continue

        fname = os.path.basename(fpath)
        for df, sc, model_label, assay_present in entries:
            if sc in seen_score_cols:
                print(f"    [skip-dup] {sc} already loaded from a previous file — skipping")
                continue
            seen_score_cols.add(sc)
            color = COLORS[color_idx % len(COLORS)]
            color_idx += 1

            # Apply logit transform to score column if requested
            df_plot = df.copy()
            if logit_trans:
                df_plot[sc] = logit_transform(df_plot[sc].values)
            all_entries.append((df_plot, sc, model_label, assay_present, color))

            corr_df = compute_correlations(df_plot, sc, model_label,
                                           assay_present, fname)
            all_corr_dfs.append(corr_df)

            # Print score distribution diagnostics
            sc_vals = df[sc].dropna()
            sc_std  = sc_vals.std()
            sc_warn = "  ⚠ narrow spread — possible saturation" if sc_std < 0.15 else ""
            print(f"\n  Correlations — {model_label.replace(chr(10),' ')}"
                  f"  [{fname}]:")
            print(f"    score: min={sc_vals.min():.3f}  max={sc_vals.max():.3f}"
                  f"  mean={sc_vals.mean():.3f}  std={sc_std:.3f}{sc_warn}")
            for _, row in corr_df.iterrows():
                # Show assay distribution alongside correlation
                assay_vals = df[row['assay_col']].dropna() if row['assay_col'] in df.columns else pd.Series(dtype=float)
                assay_info = (f"  [assay: min={assay_vals.min():.3f}"
                              f" max={assay_vals.max():.3f}"
                              f" std={assay_vals.std():.3f}]"
                              if len(assay_vals) > 0 else "")
                print(f"    {row['assay_col']:35s}"
                      f"  ρ={row['spearman_rho']:+.4f}"
                      f" ({row['significance']}, p={row['spearman_p']:.4f})"
                      f"  r={row['pearson_r']:+.4f}"
                      f"  n={row['n']}"
                      f"{assay_info}")

    if list_scores or not all_entries:
        if not list_scores:
            print("[error] No valid model entries found.")
        return

    # ── Save CSV ──────────────────────────────────────────────────────────────
    results = pd.concat(all_corr_dfs, ignore_index=True)
    csv_path = f"{out}_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"\n  Results CSV → {csv_path}")
    print(f"  Total correlations : {len(results)}")

    # Print summary table
    print(f"\n  {'─'*W}")
    print(f"  {'Model':40s}  {'Assay':25s}  {'ρ':>7}  {'sig':>4}  {'n':>6}")
    print(f"  {'─'*W}")
    for _, row in results.iterrows():
        print(f"  {row['model'][:40]:40s}  {row['assay_col'][:25]:25s}"
              f"  {row['spearman_rho']:+.4f}  {row['significance']:>4}  {row['n']:>6}")
    print(f"  {'─'*W}")

    # ── Build figure ──────────────────────────────────────────────────────────
    assay_in_data = [ac for ac in assay_cols
                     if any(ac in e[3] for e in all_entries)]
    if not assay_in_data:
        print("[warn] No assay columns matched any file — skipping figure")
        return

    n_models = len(all_entries)
    n_assays = len(assay_in_data)

    fig_title = title if title else (
        f"IPI model score vs developability assay scores\n"
        f"{n_models} model(s)  ×  {n_assays} assay(s)"
    )
    if xlabel:
        fig_xlabel = xlabel
    elif logit_trans:
        fig_xlabel = 'IPI logit score  (log p / (1−p))'
    else:
        fig_xlabel = 'IPI model score  (P(Pass))'

    fig = build_figure(all_entries, assay_in_data, fig_title, fig_xlabel)

    tiff_path = f"{out}.tiff"
    png_path  = f"{out}_preview.png"
    fig.savefig(tiff_path, dpi=300, format='tiff',
                bbox_inches='tight', facecolor='white')
    fig.savefig(png_path, dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    plt.rcdefaults()

    print(f"  Figure TIFF → {tiff_path}")
    print(f"  Figure PNG  → {png_path}")
    print(f"{'═'*W}\n")

# ══════════════════════════════════════════════════════════════════════════════
# NEW FEATURES — added after original code
# 1. Assay-vs-assay Spearman correlation matrix
# 2. Two correlation heatmaps (assay-only + assay+ML)
# 3. Boxplots per assay coloured by target label
# 4. t-SNE on assay columns (coloured by target + by each ML label)
# ══════════════════════════════════════════════════════════════════════════════

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    _TSNE_AVAILABLE = True
except ImportError:
    _TSNE_AVAILABLE = False


# ── Palette ───────────────────────────────────────────────────────────────────
_PASS_COLOR = '#4C9BE8'   # blue  — PASS / positive class
_FAIL_COLOR = '#F28C38'   # orange — FAIL / negative class
_GRID_ALPHA  = 0.2


# ── 1.  Assay correlation matrix ──────────────────────────────────────────────

def compute_assay_correlations(df: pd.DataFrame,
                                assay_cols: list) -> pd.DataFrame:
    """
    Compute pairwise Spearman ρ between all assay columns.
    Returns a square DataFrame (assay × assay).
    """
    valid = [c for c in assay_cols if c in df.columns]
    if len(valid) < 2:
        return pd.DataFrame()

    sub = df[valid].apply(pd.to_numeric, errors='coerce').dropna()
    n   = len(sub)
    mat = pd.DataFrame(index=valid, columns=valid, dtype=float)

    for i, c1 in enumerate(valid):
        for j, c2 in enumerate(valid):
            if i == j:
                mat.loc[c1, c2] = 1.0
            elif j > i:
                rho, _ = stats.spearmanr(sub[c1], sub[c2])
                mat.loc[c1, c2] = round(rho, 4)
                mat.loc[c2, c1] = round(rho, 4)

    print(f"\n  Assay-vs-assay Spearman ρ  (n={n}):")
    print(mat.to_string())
    return mat


# ── 2.  Correlation heatmaps ──────────────────────────────────────────────────

def _heatmap_ax(ax, mat: pd.DataFrame, title: str,
                vmin: float = -1.0, vmax: float = 1.0,
                annot: bool = True, fmt: str = '.2f',
                cmap: str = 'RdBu_r'):
    """Draw a single annotated heatmap on ax."""
    import matplotlib.colors as mcolors

    data  = mat.values.astype(float)
    n     = data.shape[0]
    im    = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(mat.columns, rotation=45, ha='right',
                       fontsize=SIZE_TICK - 0.5)
    ax.set_yticklabels(mat.index, fontsize=SIZE_TICK - 0.5)
    ax.set_title(title, fontsize=SIZE_AXIS, pad=6)

    if annot:
        for i in range(n):
            for j in range(n):
                v   = data[i, j]
                txt = format(v, fmt) if not np.isnan(v) else ''
                # Use white text on dark cells, dark text on light cells
                brightness = (v - vmin) / (vmax - vmin + 1e-9)
                tc = 'white' if brightness < 0.25 or brightness > 0.75 else 'black'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=SIZE_TICK - 1, color=tc)

    return im


def _compute_corr_matrix(df: pd.DataFrame, cols: list,
                          method: str = 'spearman') -> pd.DataFrame:
    """
    Compute pairwise correlation matrix.
    method: 'spearman' | 'pearson'
    Returns square DataFrame (cols × cols).
    """
    sub = df[cols].apply(pd.to_numeric, errors='coerce').dropna()
    mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i == j:
                mat.loc[c1, c2] = 1.0
            elif j > i:
                if method == 'pearson':
                    r, _ = stats.pearsonr(sub[c1], sub[c2])
                else:
                    r, _ = stats.spearmanr(sub[c1], sub[c2])
                mat.loc[c1, c2] = round(r, 4)
                mat.loc[c2, c1] = round(r, 4)
    return mat, len(sub)


def build_heatmaps(df_full: pd.DataFrame,
                   assay_cols: list,
                   all_entries: list,
                   out: str,
                   title_prefix: str = ""):
    """
    Save four heatmap figures (Spearman + Pearson × assay-only + assay+ML):
      {out}_heatmap_assay_spearman.tiff
      {out}_heatmap_assay_pearson.tiff
      {out}_heatmap_all_spearman.tiff
      {out}_heatmap_all_pearson.tiff
    """
    set_nature_style()
    valid_assays = [c for c in assay_cols if c in df_full.columns]
    if len(valid_assays) < 2:
        print("  [heatmap] < 2 assay columns — skipping heatmaps")
        return

    score_cols = [sc for (_, sc, _, _, _) in all_entries
                  if sc in df_full.columns]
    lbl_map    = {sc: lbl.replace('\n', ' ')[:30]
                  for (_, sc, lbl, _, _) in all_entries}
    all_cols   = valid_assays + score_cols
    na         = len(valid_assays)

    for method, sym, cbar_lbl in [('spearman', 'ρ', 'Spearman ρ'),
                                   ('pearson',  'r', 'Pearson r')]:

        # ── Assay-only heatmap ────────────────────────────────────────────────
        mat_assay, n_samp = _compute_corr_matrix(df_full, valid_assays, method)
        if not mat_assay.empty:
            n  = len(valid_assays)
            sz = max(3.0, n * 0.65)
            fig, ax = plt.subplots(figsize=(sz + 1.2, sz))
            im = _heatmap_ax(
                ax, mat_assay,
                title=f"{title_prefix}Assay-vs-Assay  {cbar_lbl}  (n={n_samp:,})")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_lbl)
            plt.tight_layout()
            p1 = f"{out}_heatmap_assay_{method}.tiff"
            p2 = f"{out}_heatmap_assay_{method}_preview.png"
            fig.savefig(p1, dpi=300, format='tiff',
                        bbox_inches='tight', facecolor='white')
            fig.savefig(p2, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  Heatmap assay  {method:8s} → {p1}")

        # ── Assay + ML scores heatmap ─────────────────────────────────────────
        if len(all_cols) < 2:
            continue

        mat_all, n_samp2 = _compute_corr_matrix(df_full, all_cols, method)
        mat_disp = mat_all.rename(index=lbl_map, columns=lbl_map)

        n  = len(all_cols)
        sz = max(4.0, n * 0.7)
        fig, ax = plt.subplots(figsize=(sz + 1.5, sz))
        im = _heatmap_ax(
            ax, mat_disp,
            title=f"{title_prefix}Assays + ML Scores  {cbar_lbl}  (n={n_samp2:,})",
            annot=(n <= 20))

        # Dashed separator between assay block and ML score block
        ax.axhline(na - 0.5, color='#555555', lw=1.2, ls='--')
        ax.axvline(na - 0.5, color='#555555', lw=1.2, ls='--')

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_lbl)
        plt.tight_layout()
        p3 = f"{out}_heatmap_all_{method}.tiff"
        p4 = f"{out}_heatmap_all_{method}_preview.png"
        fig.savefig(p3, dpi=300, format='tiff',
                    bbox_inches='tight', facecolor='white')
        fig.savefig(p4, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Heatmap all    {method:8s} → {p3}")


# ── 3.  Boxplots per assay coloured by target ─────────────────────────────────

def build_boxplots(df_full: pd.DataFrame,
                   assay_cols: list,
                   target_col: str,
                   out: str,
                   title_prefix: str = ""):
    """
    One subplot per assay column. Each box shows distribution of the assay
    score split by target_col (0=FAIL / 1=PASS).

    Outputs:
      {out}_boxplot_{target_col}.tiff
      {out}_boxplot_{target_col}_preview.png
    """
    set_nature_style()
    valid = [c for c in assay_cols if c in df_full.columns]
    if not valid:
        print(f"  [boxplot] No assay columns in data — skipping")
        return
    if target_col not in df_full.columns:
        print(f"  [boxplot] Target column '{target_col}' not found — skipping")
        return

    df_bp = df_full[valid + [target_col]].copy()
    df_bp[target_col] = pd.to_numeric(df_bp[target_col], errors='coerce')
    df_bp = df_bp.dropna(subset=[target_col])
    df_bp[target_col] = df_bp[target_col].astype(int)

    labels_map = {0: 'FAIL', 1: 'PASS'}
    groups     = sorted(df_bp[target_col].unique())
    colors_map = {0: _FAIL_COLOR, 1: _PASS_COLOR}

    n_assays  = len(valid)
    n_cols_bp = min(n_assays, 4)
    n_rows_bp = int(np.ceil(n_assays / n_cols_bp))
    pw        = 1.6 * n_cols_bp
    ph        = 2.0 * n_rows_bp

    fig, axes = plt.subplots(n_rows_bp, n_cols_bp,
                             figsize=(pw, ph), squeeze=False)

    for idx, ac in enumerate(valid):
        ax  = axes[idx // n_cols_bp][idx % n_cols_bp]
        sub = df_bp[[ac, target_col]].dropna()
        sub[ac] = pd.to_numeric(sub[ac], errors='coerce')
        sub = sub.dropna()

        plot_data = [sub.loc[sub[target_col] == g, ac].values for g in groups]
        bp = ax.boxplot(plot_data,
                        patch_artist=True,
                        widths=0.55,
                        medianprops=dict(color='black', lw=1.2),
                        whiskerprops=dict(lw=0.8),
                        capprops=dict(lw=0.8),
                        flierprops=dict(marker='.',
                                        markersize=2,
                                        alpha=0.5,
                                        markeredgewidth=0))
        for patch, g in zip(bp['boxes'], groups):
            patch.set_facecolor(colors_map.get(g, '#aaaaaa'))
            patch.set_alpha(0.75)

        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels([labels_map.get(g, str(g)) for g in groups],
                           fontsize=SIZE_TICK)
        ax.set_ylabel(ac, fontsize=SIZE_TICK - 0.5, labelpad=2)
        ax.set_xlabel(target_col, fontsize=SIZE_TICK - 0.5, labelpad=2)
        ax.tick_params(labelsize=SIZE_TICK, length=2)
        ax.grid(axis='y', alpha=_GRID_ALPHA, lw=0.4)

        # T-test p-value annotation
        if len(groups) == 2 and len(plot_data[0]) > 1 and len(plot_data[1]) > 1:
            t_stat, p_val = stats.ttest_ind(plot_data[0], plot_data[1],
                                            equal_var=False)
            sig = ('***' if p_val < 0.001 else '**' if p_val < 0.01 else
                   '*'   if p_val < 0.05  else 'ns')
            p_str = 'p<0.001' if p_val < 0.001 else f'p={p_val:.3f}'
            ax.set_title(f'T-test, {p_str}', fontsize=SIZE_TICK - 0.5, pad=3)

    # Hide unused axes
    for idx in range(len(valid), n_rows_bp * n_cols_bp):
        axes[idx // n_cols_bp][idx % n_cols_bp].set_visible(False)

    sup = f"{title_prefix}Assay distributions  |  by {target_col}"
    fig.suptitle(sup, fontsize=SIZE_AXIS, fontweight='bold', y=1.01)
    plt.tight_layout()

    p1 = f"{out}_boxplot_{target_col}.tiff"
    p2 = f"{out}_boxplot_{target_col}_preview.png"
    fig.savefig(p1, dpi=300, format='tiff',
                bbox_inches='tight', facecolor='white')
    fig.savefig(p2, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Boxplot → {p1}")


# ── 4.  t-SNE ─────────────────────────────────────────────────────────────────

def _tsne_scatter(ax, xy, colors, title, labels_uniq,
                  palette, alpha=0.6, s=12):
    """Draw a single t-SNE scatter with legend inside the axes (no cutoff)."""
    for lbl in labels_uniq:
        mask = colors == lbl
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   color=palette[lbl], label=str(lbl),
                   alpha=alpha, s=s, lw=0, rasterized=True)
    ax.set_title(title, fontsize=SIZE_AXIS, pad=4)
    ax.set_xlabel('First t-SNE',  fontsize=SIZE_TICK)
    ax.set_ylabel('Second t-SNE', fontsize=SIZE_TICK)
    ax.tick_params(labelsize=SIZE_TICK, length=2)
    ax.grid(alpha=_GRID_ALPHA, lw=0.4)
    # Place legend inside upper-right with a light background box — never cut off
    ax.legend(markerscale=1.6, fontsize=SIZE_TICK,
              loc='upper right', frameon=True,
              framealpha=0.85, edgecolor='#cccccc',
              bbox_to_anchor=(0.99, 0.99),
              borderpad=0.5)


def build_tsne(df_full: pd.DataFrame,
               assay_cols: list,
               target_col: str,
               all_entries: list,
               out: str,
               title_prefix: str = "",
               perplexity: int = 30,
               n_iter: int = 1000,
               random_state: int = 42,
               feature_source: str = 'assay',
               embedding_file: str = None):
    """
    t-SNE on antibody data, coloured by labels.

    feature_source:
      'assay'  (default) — features = assay columns (biological clustering)
                           Separation in this space = assay-based FAIL/PASS signal
      'scores'           — features = all ML model scores (model decision space)
                           Predicted labels should overlap with true labels here
                           Mismatch = model errors (false positives / negatives)

    Outputs (per feature_source):
      {out}_tsne_{target_col}.tiff            — coloured by true label
      {out}_tsne_labels.tiff                  — coloured by ML standard labels
      {out}_tsne_optimallabels.tiff           — coloured by optimal threshold labels
    """
    if not _TSNE_AVAILABLE:
        print("  [tsne] scikit-learn not found — skipping t-SNE (pip install scikit-learn)")
        return

    set_nature_style()
    valid = [c for c in assay_cols if c in df_full.columns]
    if len(valid) < 2:
        print("  [tsne] < 2 assay columns — skipping t-SNE")
        return

    # ── Build feature matrix based on feature_source ────────────────────────────
    # 'assay'  → assay columns (biological clustering)
    # 'scores' → ML model score columns (model decision space)

    score_cols_all = [sc for (_, sc, _, _, _) in all_entries
                      if sc in df_full.columns]

    if feature_source == 'scores':
        if not score_cols_all:
            print("  [tsne] No score columns found — cannot run score-based t-SNE")
            return
        feat_cols = score_cols_all
        source_tag = 'scores'
        source_label = f"ML score features  ({len(feat_cols)} models)"
        print(f"\n  [tsne-scores] Features: {feat_cols}")
    elif feature_source == 'embedding':
        if not embedding_file or not os.path.exists(embedding_file):
            print(f"  [tsne-embedding] File not found: {embedding_file}")
            return
        _emb_raw = pd.read_csv(embedding_file, index_col=0)
        _emb_raw.index = _emb_raw.index.astype(str).str.strip()
        df_full = df_full.copy()
        # If BARCODE is still a column (integer index), set it as index
        if 'BARCODE' in df_full.columns:
            df_full = df_full.set_index('BARCODE')
        df_full.index = df_full.index.astype(str).str.strip()
        _common_bcs = df_full.index.intersection(_emb_raw.index)
        if len(_common_bcs) == 0:
            print(f"  [tsne-embedding] No overlapping BARCODEs")
            print(f"  [tsne-embedding]   data sample    : {df_full.index[:3].tolist()}")
            print(f"  [tsne-embedding]   emb sample     : {_emb_raw.index[:3].tolist()}")
            return
        _emb      = _emb_raw.loc[_common_bcs]
        df_full   = df_full.loc[_common_bcs]
        _lm_name  = Path(embedding_file).name.split('.emb.csv')[0].split('.')[-1]
        _emb_vals = _emb.values.astype(np.float32)
        n_pca     = min(50, _emb_vals.shape[1], len(_common_bcs) - 1)
        from sklearn.decomposition import PCA
        _emb_pca  = PCA(n_components=n_pca,
                        random_state=random_state).fit_transform(
            StandardScaler().fit_transform(_emb_vals))
        print(f"  [tsne-embedding] {_lm_name}: {_emb_vals.shape[1]:,} dims "
              f"→ PCA({n_pca}) → t-SNE  ({len(_common_bcs):,} antibodies)")
        feat_assay_emb = pd.DataFrame(_emb_pca, index=_common_bcs)
        feat_cols      = list(range(n_pca))
        source_tag     = f'embedding_{_lm_name}'
        source_label   = (f"{_lm_name} PLM embedding  "
                          f"({_emb_vals.shape[1]:,} dims → PCA{n_pca})")

    else:
        feat_cols    = valid
        source_tag   = 'assay'
        source_label = f"assay features  ({len(feat_cols)} assays)" 

    if feature_source == 'embedding':
        feat_assay = feat_assay_emb
        X_feat = feat_assay.values.astype(np.float32)
        X_sc   = X_feat   # already scaled inside PCA block
    else:
        feat_assay = df_full[feat_cols].apply(pd.to_numeric, errors='coerce').dropna()
        if len(feat_assay) < 10:
            print(f"  [tsne] Only {len(feat_assay)} rows with complete "
                  f"{source_tag} data — skipping")
            return
        X_feat = feat_assay.values
        X_sc   = StandardScaler().fit_transform(X_feat)

    # Collect ALL colour columns (target + ML labels) aligned to feat_assay index
    seen_lc = set()
    ml_label_cols = []
    colour_df = pd.DataFrame(index=feat_assay.index)

    # Target column
    if target_col in df_full.columns:
        colour_df[target_col] = pd.to_numeric(
            df_full[target_col], errors='coerce').reindex(feat_assay.index)

    # *_label and *_optimallabel from detected score columns
    for (_, sc, lbl, _, _) in all_entries:
        base = sc.replace('_score', '')
        for suffix, tag in [('_label',        ''),
                             ('_optimallabel', ' (optimal thresh)')]:
            lc = base + suffix
            if lc in df_full.columns and lc not in seen_lc:
                colour_df[lc] = pd.to_numeric(
                    df_full[lc], errors='coerce').reindex(feat_assay.index)
                ml_label_cols.append((lc, lbl.replace('\n', ' ') + tag))
                seen_lc.add(lc)

    # Auto-detect any remaining *optimallabel/*costlabel/*f1label columns
    _opt_pattern_pre = re.compile(r'optimallabel|costlabel|f1label|f2label', re.I)
    for col in df_full.columns:
        if col not in seen_lc and _opt_pattern_pre.search(col):
            colour_df[col] = pd.to_numeric(
                df_full[col], errors='coerce').reindex(feat_assay.index)
            ml_label_cols.append((col, col))
            seen_lc.add(col)

    # Print what was found
    std_found = [lc for lc, _ in ml_label_cols
                 if not _opt_pattern_pre.search(lc)]
    opt_found = [lc for lc, _ in ml_label_cols
                 if _opt_pattern_pre.search(lc)]
    print(f"  [tsne] Standard labels   : {std_found if std_found else 'none'}")
    print(f"  [tsne] Optimal labels    : {opt_found if opt_found else 'none'}")

    # Use feat_assay index as canonical — colour_df aligned to same index
    feat = feat_assay  # kept as alias for downstream colour lookups

    # Adjust perplexity to dataset size
    perp = min(perplexity, max(5, len(feat_assay) // 5))
    print(f"\n  [tsne-{source_tag}] Running t-SNE on {len(feat_assay):,} antibodies × "
          f"{len(feat_cols)} {source_tag} features  (perplexity={perp}) ...")

    # n_iter renamed to max_iter in scikit-learn >= 1.4 — support both
    import sklearn as _skl
    _skl_ver = tuple(int(x) for x in _skl.__version__.split('.')[:2])
    _iter_kw = 'max_iter' if _skl_ver >= (1, 4) else 'n_iter'
    tsne_xy = TSNE(n_components=2, perplexity=perp,
                   **{_iter_kw: n_iter},
                   random_state=random_state,
                   init='pca', learning_rate='auto').fit_transform(X_sc)

    label_palette = {0: _FAIL_COLOR, 1: _PASS_COLOR,
                     '0': _FAIL_COLOR, '1': _PASS_COLOR,
                     'FAIL': _FAIL_COLOR, 'PASS': _PASS_COLOR}

    # ── Figure 1: coloured by true target label ───────────────────────────────
    if target_col in colour_df.columns and colour_df[target_col].notna().any():
        fig_t, ax_t = plt.subplots(1, 1, figsize=(5.5, 4.5))
        tgt_vals = colour_df[target_col].fillna(-1).values.astype(int)
        # Only plot rows with valid label
        _mask    = tgt_vals >= 0
        tsne_tgt = tsne_xy[_mask]
        tgt_vals = tgt_vals[_mask]
        uniq     = sorted(np.unique(tgt_vals))
        pal      = {v: label_palette.get(v, COLORS[i % len(COLORS)])
                    for i, v in enumerate(uniq)}
        lbl_map  = {0: 'FAIL', 1: 'PASS'}
        lbl_arr  = np.array([lbl_map.get(v, str(v)) for v in tgt_vals])
        pal_str  = {lbl_map.get(k, str(k)): v for k, v in pal.items()}
        uniq_str = [lbl_map.get(v, str(v)) for v in uniq]

        _tsne_scatter(ax_t, tsne_tgt, lbl_arr,
                      title=f"{title_prefix}t-SNE ({source_label})  |  {target_col}",
                      labels_uniq=uniq_str, palette=pal_str)
        fig_t.suptitle("", fontsize=0)
        plt.tight_layout()
        p1 = f"{out}_tsne_{source_tag}_{target_col}.tiff"
        p2 = f"{out}_tsne_{source_tag}_{target_col}_preview.png"
        fig_t.savefig(p1, dpi=300, format='tiff',
                      bbox_inches='tight', facecolor='white')
        fig_t.savefig(p2, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  t-SNE ({source_tag}, target)  → {p1}")



    # ── Figure 2: one figure with ALL standard *_label columns (subplots) ──────
    # ── Figure 3: one figure with ALL *_optimallabel columns (subplots) ─────────
    #
    # Groups:
    #   standard  : *_label               (threshold = 0.5)
    #   optimal   : *_optimallabel, *_costlabel, *_f1label, *_f2label ...
    std_cols = [(lc, lbl) for lc, lbl in ml_label_cols
                if not _opt_pattern_pre.search(lc)]
    opt_cols = [(lc, lbl) for lc, lbl in ml_label_cols
                if _opt_pattern_pre.search(lc)]

    def _save_tsne_group(group, group_name, out_suffix, title_suffix,
                         src_tag=source_tag, src_lbl=source_label):
        if not group:
            return
        n        = len(group)
        n_cols_g = min(n, 3)
        n_rows_g = int(np.ceil(n / n_cols_g))
        pw       = 5.5 * n_cols_g
        ph       = 4.8 * n_rows_g

        fig_g, axes_g = plt.subplots(n_rows_g, n_cols_g,
                                     figsize=(pw, ph), squeeze=False)

        for idx, (lc, model_lbl) in enumerate(group):
            ax = axes_g[idx // n_cols_g][idx % n_cols_g]
            if lc not in colour_df.columns or colour_df[lc].isna().all():
                ax.set_visible(False)
                continue

            # Only plot rows with valid label (NaN = no prediction for that row)
            _lc_vals = colour_df[lc].fillna(-1).values.astype(int)
            _lc_mask = _lc_vals >= 0
            ml_vals  = _lc_vals[_lc_mask]
            _tsne_lc = tsne_xy[_lc_mask]
            uniq    = sorted(np.unique(ml_vals))
            pal     = {v: label_palette.get(v, COLORS[i % len(COLORS)])
                       for i, v in enumerate(uniq)}
            lm      = {0: 'FAIL', 1: 'PASS'}
            lbl_arr = np.array([lm.get(v, str(v)) for v in ml_vals])
            pal_str = {lm.get(k, str(k)): v for k, v in pal.items()}
            uniq_s  = [lm.get(v, str(v)) for v in uniq]

            _tsne_scatter(ax, _tsne_lc, lbl_arr,
                          title=model_lbl[:55],
                          labels_uniq=uniq_s, palette=pal_str)

        # Hide unused axes
        for idx in range(n, n_rows_g * n_cols_g):
            axes_g[idx // n_cols_g][idx % n_cols_g].set_visible(False)

        sup = f"{title_prefix}t-SNE ({src_lbl})  |  {title_suffix}"
        fig_g.suptitle(sup, fontsize=SIZE_AXIS, fontweight='bold', y=1.01)
        plt.tight_layout()

        p_tiff = f"{out}_{src_tag}_{out_suffix}.tiff"
        p_png  = f"{out}_{src_tag}_{out_suffix}_preview.png"
        fig_g.savefig(p_tiff, dpi=300, format='tiff',
                      bbox_inches='tight', facecolor='white')
        fig_g.savefig(p_png,  dpi=150,
                      bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  t-SNE ({src_tag}, {group_name}) → {p_tiff}")

    _save_tsne_group(std_cols, 'standard labels',
                     'tsne_labels',
                     'ML prediction labels  (threshold = 0.5)')
    _save_tsne_group(opt_cols, 'optimal labels',
                     'tsne_optimallabels',
                     'ML optimal threshold labels')

# ── Patch run() to call new features ──────────────────────────────────────────
# We wrap the original run() and call it, then add the new outputs.

_original_run = run


def run(files: list, assay_cols: list, out: str = None,
        title: str = None, xlabel: str = None,
        logit_trans: bool = False,
        list_scores: bool = False,
        target_col: str = None,
        tsne_source: str = 'assay',
        embedding_file: str = None):
    """
    Extended run() — calls original correlation analysis then adds:
      • Assay-vs-assay Spearman correlation matrix (printed + saved to CSV)
      • Two correlation heatmaps (assay-only + assay+ML)
      • Boxplots per assay coloured by target_col
      • t-SNE on assay features (coloured by target + by ML labels)

    target_col : str — label column (e.g. 'psr_filter', 'sec_filter').
                       Required for boxplots and t-SNE.
                       Auto-detected from common names if not provided.
    """
    # Auto-detect output stem before calling original
    if out is None:
        out = _auto_out_stem(files, assay_cols)

    # ── Run original correlation analysis ─────────────────────────────────────
    _original_run(files=files, assay_cols=assay_cols, out=out,
                  title=title, xlabel=xlabel,
                  logit_trans=logit_trans, list_scores=list_scores)

    if list_scores:
        return

    # ── Load merged data for new analyses ─────────────────────────────────────
    # Load all files, merge on BARCODE (or row order if no BARCODE col)
    dfs = []
    for fpath in files:
        try:
            if fpath.lower().endswith('.csv'):
                df = pd.read_csv(fpath)
            else:
                xl    = pd.ExcelFile(fpath)
                sheet = _preferred_sheet(xl)
                df    = xl.parse(sheet)
            dfs.append(df)
        except Exception as e:
            print(f"  [extra] Warning: could not load {fpath}: {e}")

    if not dfs:
        return

    # Merge all files by BARCODE if possible, otherwise concat
    if len(dfs) == 1:
        df_full = dfs[0]
    else:
        if all('BARCODE' in d.columns for d in dfs):
            df_full = dfs[0]
            for d in dfs[1:]:
                new_cols = [c for c in d.columns
                            if c not in df_full.columns or c == 'BARCODE']
                df_full  = df_full.merge(d[new_cols], on='BARCODE', how='outer')
        else:
            df_full = pd.concat(dfs, axis=0, ignore_index=True)

    # Set BARCODE as index so embedding CSV alignment works
    # (PLM + hidden state CSVs always have BARCODE as index_col=0)
    if 'BARCODE' in df_full.columns:
        df_full = df_full.set_index('BARCODE')
        df_full.index = df_full.index.astype(str).str.strip()

    # ── Auto-detect target column if not provided ─────────────────────────────
    if target_col is None:
        _common_targets = ['psr_filter', 'psr_filter_cho', 'psr_filter_ova',
                           'sec_filter', 'hic_filter',
                           'acsins_filter', 'target', 'label', 'class',
                           'psr_label', 'sec_label']
        for cand in _common_targets:
            if cand in df_full.columns:
                target_col = cand
                print(f"\n  [extra] Auto-detected target column: '{target_col}'")
                break
        if target_col is None:
            print("  [extra] No target column found — "
                  "boxplots and t-SNE will be skipped. "
                  "Pass --target to specify one.")

    # Collect all_entries for heatmap + t-SNE
    all_entries_new = []
    for fpath in files:
        entries = load_file(fpath, assay_cols, list_scores=False)
        for df_e, sc, lbl, assay_pres in entries:
            if sc in df_full.columns:
                all_entries_new.append((df_e, sc, lbl, assay_pres,
                                        COLORS[len(all_entries_new) % len(COLORS)]))

    title_prefix = (f"{title}  —  " if title else "")
    W = 70
    print(f"\n{'═'*W}")
    print(f"  Extended analyses  (heatmaps · boxplots · t-SNE)")
    print(f"{'─'*W}")

    # 1. Heatmaps
    build_heatmaps(df_full, assay_cols, all_entries_new, out,
                   title_prefix=title_prefix)

    # 2. Boxplots
    if target_col:
        build_boxplots(df_full, assay_cols, target_col, out,
                       title_prefix=title_prefix)

    # 3. t-SNE
    if target_col:
        # Assay-based t-SNE (biological clustering)
        if tsne_source in ('assay', 'both'):
            build_tsne(df_full, assay_cols, target_col,
                       all_entries_new, out,
                       title_prefix=title_prefix,
                       feature_source='assay')

        # Score-based t-SNE (model decision space)
        if tsne_source in ('scores', 'both'):
            build_tsne(df_full, assay_cols, target_col,
                       all_entries_new, out,
                       title_prefix=title_prefix,
                       feature_source='scores')

        if tsne_source == 'embedding':
            if not embedding_file:
                print("  [tsne] --embedding-file required for --tsne-source embedding")
            else:
                build_tsne(df_full, assay_cols, target_col,
                           all_entries_new, out,
                           title_prefix=title_prefix,
                           feature_source='embedding',
                           embedding_file=embedding_file)

    print(f"{'═'*W}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='developability_correlation.py',
        description='IPI model score vs developability assay correlation (PSR, SEC, HIC, SPR ...)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
COMMON ASSAY COLUMN NAMES
──────────────────────────
  PSR / polyreactivity
    psr_norm_dna  psr_norm_avidin  psr_norm_insulin  psr_norm_smp  psr_norm_mean
    polyreactivity_prscore_ova_avg  polyreactivity_prscore_cho_avg
    "PSR  SMP Score (0-1)"  "BVP ELISA"  ELISA

  SEC (size exclusion chromatography)
    sec_monomer_pct  sec_aggregate_pct  sec_hm_pct
    "SEC %monomer"   "SEC %aggregate"

  HIC (hydrophobicity interaction chromatography)
    hic_retention_time  "HIC Retention Time (Min)a"

  AC-SINS (self-association)
    "SGAC-SINS AS100 ((NH4)2SO4 mM)"  acsins_delta_lambda

  SPR (surface plasmon resonance / affinity)
    kd_nM  kon_1_Ms  koff_1_s

  Tm / thermostability
    "Fab Tm by DSF (°C)"  fab_tm  ch2_tm

  Viscosity
    viscosity_cP  "viscosity at 150 mg/mL"

  Titer / expression
    "HEK Titer (mg/L)"  hek_titer  cho_titer

COMMON TARGET / LABEL COLUMNS
───────────────────────────────
  psr_filter  sec_filter  hic_filter  acsins_filter  target  label

RUN --list-scores TO SEE WHAT COLUMNS WERE DETECTED IN YOUR FILE
        """,
    )
    parser.add_argument(
        '--files', nargs='+', required=True,
        help='One or more prediction Excel (.xlsx) or CSV files')
    parser.add_argument(
        '--assay', nargs='+', required=True,
        help='Assay column name(s) to correlate against (any developability assay)')
    parser.add_argument(
        '--out', default=None,
        help=(
            'Output file stem. Default: auto-derived from input file path and assay type. ' 
            'e.g. test/manuscript/GDPa3_pred_all_psr_elisa_correlation'
        ))
    parser.add_argument(
        '--title', default=None,
        help='Custom figure title (default: auto-generated)')
    parser.add_argument(
        '--xlabel', default=None,
        help='Custom x-axis label (default: "IPI model score  (P(Pass))")')
    parser.add_argument(
        '--logit-trans', '--logit_trans', action='store_true', default=False,
        dest='logit_trans',
        help=(
            'Apply logit transform to scores before correlating: '
            'logit(p) = log(p / (1-p)). '
            'Spreads compressed tails from sigmoid saturation. '
            'Spearman rho unchanged (rank-invariant); '
            'Pearson r and scatter spread improve. '
            'Default: False (raw probability scores).'
        ))
    parser.add_argument(
        '--list-scores', action='store_true',
        help='Print detected IPI score columns and exit — no analysis')

    parser.add_argument(
        '--embedding-file', dest='embedding_file', default=None, metavar='EMB_CSV',
        help='PLM embedding CSV for --tsne-source embedding. '
             'e.g. test/dataset.xlsx.igbert.emb.csv')
    parser.add_argument(
        '--target', default=None,
        help=(
            'Label column name for boxplots and t-SNE colouring '
            '(e.g. psr_filter, sec_filter). '
            'Auto-detected from common names if not specified.'
        ))
    parser.add_argument(
        '--tsne-source', dest='tsne_source',
        choices=['assay', 'scores', 'embedding', 'both'],
        default='assay',
        help=(
            'Feature space for t-SNE embedding. '
            'assay  (default): t-SNE on assay columns — shows biological clustering. '
            'scores: t-SNE on all ML model scores — shows model decision space, '
            '        predicted labels should overlap perfectly with true labels. '
            'both  : run both assay-based and score-based t-SNE figures.'
        ))
    args = parser.parse_args()

    # ── Parse --files: support both space-separated and comma-separated ───────
    # e.g. --files a.xlsx b.xlsx  OR  --files a.xlsx,b.xlsx  OR  --files a.xlsx, b.xlsx
    _files = []
    for f in args.files:
        _files.extend([x.strip() for x in f.split(',') if x.strip()])

    # ── Parse --assay: support multi-word column names passed with commas ─────
    # Problem: argparse splits on spaces, so --assay SMP Score, OVA Score
    #   becomes ['SMP', 'Score,', 'OVA', 'Score'] instead of ['SMP Score', 'OVA Score']
    #
    # Strategy:
    #   • If ANY token contains a comma → user is using comma as delimiter
    #     → join all tokens with space, then split on comma
    #     e.g. ['SMP', 'Score,', 'OVA', 'Score'] → 'SMP Score, OVA Score'
    #          → ['SMP Score', 'OVA Score'] ✓
    #
    #   • If NO token contains a comma → user is using space-separated quoted args
    #     → keep as-is (e.g. ["SMP Score", "OVA Score"] from --assay "SMP Score" "OVA Score")
    #
    _raw_assay = args.assay
    if any(',' in token for token in _raw_assay):
        # Re-join and split on comma
        _assay_cols = [x.strip() for x in ' '.join(_raw_assay).split(',') if x.strip()]
    else:
        _assay_cols = [x.strip() for x in _raw_assay if x.strip()]

    run(
        files       = _files,
        assay_cols  = _assay_cols,
        out         = args.out,
        title       = args.title,
        xlabel      = args.xlabel,
        logit_trans = args.logit_trans,
        list_scores = args.list_scores,
        target_col  = args.target,
        tsne_source    = args.tsne_source,
        embedding_file = getattr(args, 'embedding_file', None),
    )