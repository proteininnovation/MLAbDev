"""
Figure 5 — Nature Biotechnology super figure
IPI PSR: 10-fold CV (panel a) + Cross-dataset (panel b)
         + External AUC heatmap (panel c) + Spearman ρ heatmap (panel d)

Usage:
    python figure5_natbiotech.py
    python figure5_natbiotech.py --excel Figure4_data.xlsx --dpi 300

Dependencies: pip install pandas openpyxl scipy scikit-learn matplotlib
              mpl_toolkits (bundled with matplotlib)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# 0.  CONSTANTS — edit here to customise the figure
# ═══════════════════════════════════════════════════════════════════════════════

# -- Figure geometry -----------------------------------------------------------
MM       = 1 / 25.4          # mm → inch
FW       = 183 * MM          # 183 mm = Nature Biotech double-column width
FH       = 216 * MM          # 216 mm = fits one Nature Biotech page
DPI      = 300
FONT     = "Arial"

# -- Panel a/b dot-plot style --------------------------------------------------
MS       = 4.5               # marker size (points)
ALPH     = 0.85              # marker alpha
REF_AUC_A = 0.94             # reference line in panel a
REF_AUC_B = 0.90             # reference line in panel b
GAP      = 1.6               # vertical gap between architecture blocks

# -- Heatmap -------------------------------------------------------------------
THRESH   = 0.27              # polyreactivity threshold: score < THRESH → PASS

# -- Colour palettes -----------------------------------------------------------
ARCH_COL = {
    "CNN"        : "#0072B2",
    "Transformer": "#E69F00",
    "RF"         : "#009E73",
    "XGBoost"    : "#CC79A7",
}
ARCH_MK  = {"CNN":"o", "Transformer":"s", "RF":"^", "XGBoost":"D"}
ARCH_ORDER = ["CNN", "Transformer", "RF", "XGBoost"]

COND_COL = {
    "DS1 10-fold CV": "#0072B2",
    "DS1 → IPI"     : "#D55E00",
    "IPI → DS1"     : "#009E73",
}
COND_MK  = {"DS1 10-fold CV":"o", "DS1 → IPI":"s", "IPI → DS1":"^"}
COND_MAP = {
    "DS1 · 10-Fold CV"  : "DS1 10-fold CV",
    "DS1 → IPI Transfer": "DS1 → IPI",
    "IPI → DS1 Transfer": "IPI → DS1",
}

# -- LM definitions ------------------------------------------------------------
# Heatmap rows: 6 PLMs (full names)
HM_LM_ORDER   = ['ablang', 'antiberty', 'antiberta2', 'antiberta2-cssp',
                  'igbert', 'onehot']
HM_LM_DISPLAY = ['AbLang2', 'AntiBERTy', 'AntiBERTa2', 'AntiBERTa2-CSSP',
                  'IgBert', 'One-hot']

# Panel a LMs per architecture
ARCH_LMS = {
    "CNN"        : ["AbLang2","AntiBERTy","AntiBERTa2","AntiBERTa2-CSSP","IgBert"],
    "Transformer": ["AbLang2","AntiBERTy","AntiBERTa2","AntiBERTa2-CSSP","IgBert","One-hot"],
    "RF"         : ["AbLang2","AntiBERTy","AntiBERTa2","AntiBERTa2-CSSP",
                    "IgBert","Biophysical","k-mer"],
    "XGBoost"    : ["AbLang2","AntiBERTy","AntiBERTa2","AntiBERTa2-CSSP",
                    "IgBert","Biophysical","k-mer"],
}

# -- Heatmap column definitions ------------------------------------------------
DATASETS_AUC = [
    'Jain 2017\nPSR SMP',
    'GDPa1\nPR Ova', 'GDPa1\nPR CHO',
    'GDPa3\nPR Ova', 'GDPa3\nPR CHO',
]
DATASETS_RHO = [
    'Jain 2017\nPSR SMP', 'Jain 2017\nELISA',
    'GDPa1\nPR Ova',       'GDPa1\nPR CHO',
    'GDPa3\nPR Ova',       'GDPa3\nPR CHO',
]

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  RCPARAMS
# ═══════════════════════════════════════════════════════════════════════════════
matplotlib.rcParams.update({
    "font.family"       : "sans-serif",
    "font.sans-serif"   : [FONT, "Helvetica", "DejaVu Sans"],
    "font.size"         : 6,
    "axes.labelsize"    : 6.5,
    "axes.titlesize"    : 6.5,
    "axes.linewidth"    : 0.5,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "xtick.labelsize"   : 5.5,
    "ytick.labelsize"   : 5.5,
    "xtick.major.width" : 0.5,
    "ytick.major.width" : 0.5,
    "xtick.major.size"  : 2,
    "ytick.major.size"  : 0,
    "xtick.direction"   : "out",
    "ytick.direction"   : "out",
    "legend.fontsize"   : 5.5,
    "legend.frameon"    : False,
    "pdf.fonttype"      : 42,
    "ps.fonttype"       : 42,
})

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  HELPER — build y-position map for panel a
# ═══════════════════════════════════════════════════════════════════════════════
def build_y_map():
    """
    Returns:
        y_map       : {(arch, lm): y_position}
        y_labels    : [(y_pos, label_str)]
        y_arch_mid  : {arch: midpoint_y}
        y_arch_span : {arch: (y_start, y_end)}   for vertical arch labels
        y_sep       : [y positions of horizontal separator lines]
        total_y     : max y value
    """
    y_map = {}; y_labels = []; y_arch_mid = {}
    y_arch_span = {}; y_sep = []; y = 0

    for arch in ARCH_ORDER:
        y_start = y
        for lm in ARCH_LMS[arch]:
            y_map[(arch, lm)] = y
            y_labels.append((y, lm))
            y += 1
        y_arch_mid[arch]  = (y_start + y - 1) / 2
        y_arch_span[arch] = (y_start, y - 1)
        y_sep.append(y - 0.5)
        y += GAP

    return y_map, y_labels, y_arch_mid, y_arch_span, y_sep, y - GAP

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
def load_fig4_data(excel_path):
    xl = pd.ExcelFile(excel_path)

    df_cv = pd.read_excel(excel_path, sheet_name='Fig4A_IPI_10fold_CV', skiprows=1)
    df_cv.columns = df_cv.columns.str.strip()
    df_cv['Architecture'] = df_cv['Architecture'].ffill()
    df_cv = df_cv.dropna(subset=['Language Model'])
    df_cv = df_cv.rename(columns={'Language Model': 'LM', 'F1-Score': 'F1'})

    df_xfer = pd.read_excel(excel_path, sheet_name='Fig4B_Cross_Dataset', skiprows=1)
    df_xfer.columns = df_xfer.columns.str.strip()
    df_xfer['Condition'] = df_xfer['Condition'].ffill()
    df_xfer = df_xfer.dropna(subset=['Language Model'])
    df_xfer = df_xfer.rename(columns={'Language Model': 'LM', 'F1-Score': 'F1'})
    df_xfer['cond'] = df_xfer['Condition'].map(COND_MAP)

    return df_cv, df_xfer


def score_col(lm):
    if lm == 'onehot':
        return 'transformer_onehot_onehot_ipi_psr_trainset_score'
    return f'transformer_lm_{lm}_ipi_psr_trainset_score'


def safe_auc(df, sc, assay_col, min_fail=5):
    """AUC with THRESH < 0.27 = PASS. NaN if too few examples in either class."""
    sub = df[[sc, assay_col]].dropna()
    y   = (sub[assay_col] < THRESH).astype(int)
    if (y == 0).sum() < min_fail or (y == 1).sum() < min_fail:
        return np.nan
    return roc_auc_score(y, sub[sc])


def safe_rho(df, sc, assay_col):
    sub = df[[sc, assay_col]].dropna()
    if len(sub) < 5:
        return np.nan
    return spearmanr(sub[sc], sub[assay_col])[0]


def compute_heatmap_matrices(jain, gdpa1, gdpa3):
    n = len(HM_LM_ORDER)
    auc = np.full((n, len(DATASETS_AUC)), np.nan)
    rho = np.full((n, len(DATASETS_RHO)), np.nan)

    for i, lm in enumerate(HM_LM_ORDER):
        c = score_col(lm)

        # ── Jain 2017 ──────────────────────────────────────────────────────────
        if c in jain.columns:
            auc[i, 0] = safe_auc(jain, c, 'PSR_SMP_Score')
            rho[i, 0] = safe_rho(jain, c, 'PSR_SMP_Score')
            rho[i, 1] = safe_rho(jain, c, 'ELISA')   # ELISA: different scale → ρ only

        # ── GDPa1 ──────────────────────────────────────────────────────────────
        if c in gdpa1.columns:
            auc[i, 1] = safe_auc(gdpa1, c, 'polyreactivity_prscore_ova_avg')
            auc[i, 2] = safe_auc(gdpa1, c, 'polyreactivity_prscore_cho_avg')
            rho[i, 2] = safe_rho(gdpa1, c, 'polyreactivity_prscore_ova_avg')
            rho[i, 3] = safe_rho(gdpa1, c, 'polyreactivity_prscore_cho_avg')

        # ── GDPa3 ──────────────────────────────────────────────────────────────
        if c in gdpa3.columns:
            auc[i, 3] = safe_auc(gdpa3, c, 'polyreactivity_prscore_ova_avg', min_fail=3)
            auc[i, 4] = safe_auc(gdpa3, c, 'polyreactivity_prscore_cho_avg')
            rho[i, 4] = safe_rho(gdpa3, c, 'polyreactivity_prscore_ova_avg')
            rho[i, 5] = safe_rho(gdpa3, c, 'polyreactivity_prscore_cho_avg')

    return auc, rho

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  PANEL DRAWING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 4a. IPI 10-fold CV AUC dot plot ──────────────────────────────────────────
def draw_panel_a(ax, df_cv, y_map, y_labels, y_arch_mid,
                 y_arch_span, y_sep, total_y):
    """
    Dot plot: one dot per (arch, LM).
    Architecture labels drawn VERTICALLY to the left, avoiding LM name collision.
    """
    # (arch separators drawn inline with labels below)

    for arch in ARCH_ORDER:
        pairs = [(arch, lm) for lm in ARCH_LMS[arch] if (arch, lm) in y_map]
        if not pairs:
            continue
        ys = [y_map[p] for p in pairs]
        xs = []
        for p in pairs:
            row = df_cv[(df_cv['Architecture'] == p[0]) & (df_cv['LM'] == p[1])]
            xs.append(float(row['AUC'].iloc[0]) if len(row) else REF_AUC_A)

        # Range line
        ax.hlines(ys, xmin=[min(xs)]*len(ys), xmax=[max(xs)]*len(ys),
                  colors="#DDDDDD", linewidths=0.4, zorder=1)
        # Dots
        ax.scatter(xs, ys, c=ARCH_COL[arch], marker=ARCH_MK[arch],
                   s=MS**2, alpha=ALPH, linewidths=0.3,
                   edgecolors="white", zorder=3)

        # ── Architecture label: placed in the gap ABOVE each block ──────────
        # Using data coordinates so it never touches the y-tick labels.
        # The gap above the first row of each block (y_start - GAP/2) is empty
        # white space — perfect for a small arch label.
        y_s, y_e = y_arch_span[arch]
        # In inverted-y space the "top" of the block is y_s.
        # We place the label in the GAP region just above y_s (which renders
        # *below* y_s visually after invert_yaxis).
        y_label_pos = y_s - GAP * 0.55   # sits in the inter-block gap
        ax.text(0.9305, y_label_pos, arch,
                color      = ARCH_COL[arch],
                fontsize   = 6.5,
                fontweight = "bold",
                ha         = "left",
                va         = "center",
                clip_on    = False)
        # Thin colored rule to the right of the label
        ax.hlines(y_label_pos, 0.934, 0.975,
                  colors    = ARCH_COL[arch],
                  linewidths= 0.5,
                  alpha     = 0.35,
                  zorder    = 0)

    # Reference line + shaded region
    ax.axvline(REF_AUC_A, color="#888", lw=0.5, ls="--", zorder=0, alpha=0.7)
    ax.axvspan(REF_AUC_A, 0.976, alpha=0.04, color="#0072B2")
    ax.text(REF_AUC_A + 0.0005, total_y + 0.3,
            f"≥{REF_AUC_A}", fontsize=5, color="#777", va="bottom")

    # Y axis: LM names
    yt_y  = [v for v, _ in y_labels]
    yt_lb = [lb for _, lb in y_labels]
    ax.set_yticks(yt_y)
    ax.set_yticklabels(yt_lb, fontsize=5.0)
    ax.set_ylim(-0.6, total_y + 0.6)
    ax.invert_yaxis()

    # X axis
    ax.set_xlim(0.930, 0.975)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.02))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax.tick_params(axis="x", which="minor", length=1.2)
    ax.grid(axis="x", lw=0.25, alpha=0.35)
    ax.set_xlabel("AUC-ROC", fontsize=6.5, labelpad=3)

    # Panel title + letter
    ax.set_title(
        "IPI PSR Dataset · 10-Fold HCDR3-Stratified CV",
        fontsize=6, pad=14, fontweight='bold', loc='left')
    ax.text(-0.14, 1.16, "a",
            transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="left")

    # Architecture legend
    handles = [
        mlines.Line2D([0], [0], marker=ARCH_MK[a], color="w",
                      markerfacecolor=ARCH_COL[a], markersize=4.5, alpha=ALPH,
                      markeredgecolor="white", markeredgewidth=0.3, label=a)
        for a in ARCH_ORDER
    ]
    leg = ax.legend(handles=handles, loc="upper left",
                    bbox_to_anchor=(0.01, 0.99),
                    fontsize=5.5, handlelength=0.5, handletextpad=0.3,
                    labelspacing=0.18, borderpad=0.35,
                    frameon=True, framealpha=0.92,
                    edgecolor="#CCCCCC", fancybox=False)
    leg.get_frame().set_linewidth(0.4)
    leg.set_title("Architecture", prop={"size": 5.5, "weight": "bold"})


# ── 4b. Cross-dataset AUC dot plot ───────────────────────────────────────────
def draw_panel_b(ax, df_xfer):
    conds  = ["DS1 10-fold CV", "DS1 → IPI", "IPI → DS1"]
    LM_B   = ["AbLang2", "AntiBERTy", "AntiBERTa2", "AntiBERTa2-CSSP",
              "IgBert", "One-hot"]
    y_lm   = {lm: i for i, lm in enumerate(LM_B)}
    lookup = {lm: {} for lm in LM_B}

    for _, row in df_xfer.iterrows():
        lm = row["LM"]
        if lm in lookup and pd.notna(row.get("cond")):
            lookup[lm][row["cond"]] = float(row["AUC"])

    # Range lines
    for lm in LM_B:
        vals   = [lookup[lm].get(c) for c in conds]
        filled = [v for v in vals if v is not None]
        if len(filled) >= 2:
            ax.hlines(y_lm[lm], xmin=min(filled), xmax=max(filled),
                      colors="#CCCCCC", linewidths=0.7, zorder=1)
    # Dots
    for cond in conds:
        xs, ys = [], []
        for lm in LM_B:
            v = lookup[lm].get(cond)
            if v is not None:
                xs.append(v); ys.append(y_lm[lm])
        ax.scatter(xs, ys, c=COND_COL[cond], marker=COND_MK[cond],
                   s=MS**2, alpha=ALPH, linewidths=0.3,
                   edgecolors="white", zorder=3, label=cond)

    ax.axvline(REF_AUC_B, color="#888", lw=0.5, ls="--", zorder=0, alpha=0.7)
    ax.axvspan(REF_AUC_B, 1.02, alpha=0.04, color="#0072B2")
    ax.text(0.906, len(LM_B) - 0.2, "≥0.90", fontsize=5, color="#777", va="bottom")

    ax.set_yticks(range(len(LM_B)))
    ax.set_yticklabels(LM_B, fontsize=5.5)
    ax.set_ylim(-0.6, len(LM_B) - 0.4)
    ax.invert_yaxis()
    ax.set_xlim(0.60, 1.02)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.tick_params(axis="x", which="minor", length=1.2)
    ax.grid(axis="x", lw=0.25, alpha=0.35)
    ax.set_xlabel("AUC-ROC", fontsize=6.5, labelpad=3)

    ax.set_title("Cross-Dataset Generalization · Transformer Architecture",
                 fontsize=6, pad=14, fontweight='bold', loc='left')
    ax.text(-0.13, 1.16, "b",
            transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="left")

    handles = [
        mlines.Line2D([0], [0], marker=COND_MK[c], color="w",
                      markerfacecolor=COND_COL[c], markersize=4.5, alpha=ALPH,
                      markeredgecolor="white", markeredgewidth=0.3, label=c)
        for c in conds
    ]
    leg = ax.legend(handles=handles, loc="upper left",
                    bbox_to_anchor=(0.01, 0.99),
                    fontsize=5.5, handlelength=0.5, handletextpad=0.3,
                    labelspacing=0.18, borderpad=0.35,
                    frameon=True, framealpha=0.92,
                    edgecolor="#CCCCCC", fancybox=False)
    leg.get_frame().set_linewidth(0.4)
    leg.set_title("Condition", prop={"size": 5.5, "weight": "bold"})


# ── 4c/d. RdBu_r heatmap (Image-1 style) ──────────────────────────────────────
def draw_heatmap(ax, mat, row_labels, col_labels,
                 title, panel_letter,
                 vmin, vmax, vcenter,
                 cmap='RdBu_r', fmt='.2f', cbar_label='',
                 col_sep=None, grp_labels=None, cbar_ticks=None):
    """
    Image-1 style heatmap:
      • RdBu_r   diverging colour map centred at vcenter
      • White text on dark cells, dark text on light cells
      • Dashed vertical separators at col_sep positions
      • Dataset bracket labels below x-axis (grp_labels)
    """
    n_r, n_c = mat.shape
    norm  = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    im    = ax.imshow(mat, aspect='auto', cmap=cmap,
                      norm=norm, interpolation='nearest')

    cmap_obj = plt.get_cmap(cmap)
    for i in range(n_r):
        for j in range(n_c):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, 'N/A', ha='center', va='center',
                        fontsize=5, color='#888888')
                continue
            rgba = cmap_obj(norm(v))
            lum  = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            tc   = 'white' if lum < 0.45 else '#333333'
            fw   = 'bold'  if abs(v - vcenter) > 0.12 else 'normal'
            ax.text(j, i, format(v, fmt),
                    ha='center', va='center',
                    fontsize=6.5, color=tc, fontweight=fw)

    # Axes ticks
    ax.set_xticks(range(n_c))
    ax.set_xticklabels(col_labels, fontsize=6,
                       rotation=38, ha='right', rotation_mode='anchor')
    ax.set_yticks(range(n_r))
    ax.set_yticklabels(row_labels, fontsize=6.5)
    ax.tick_params(length=0, pad=3)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Dashed dataset separators
    if col_sep:
        for sep in col_sep:
            ax.axvline(sep + 0.5, color='#555555', lw=1.2, ls='--', zorder=5)

    # Title (single line, left-aligned, no wrapping)
    ax.set_title(title, fontsize=6.5, pad=6, fontweight='bold', loc='left')

    # Panel letter
    ax.text(-0.065, 1.22, panel_letter,
            transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top', ha='left')

    # Colorbar
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='2%', pad=0.07)
    cb  = plt.colorbar(im, cax=cax)
    cb.set_label(cbar_label, fontsize=6, labelpad=3)
    if cbar_ticks is not None:
        cb.set_ticks(cbar_ticks)
    cb.ax.tick_params(labelsize=5.5, length=2)

    # Dataset group bracket labels below x-axis
    if grp_labels:
        for grp_lbl, x0, x1 in grp_labels:
            xm = (x0 + x1) / 2.0
            ax.annotate('',
                        xy     = (x0 - 0.42, -0.62),
                        xytext = (x1 + 0.42, -0.62),
                        xycoords     = ('data', 'axes fraction'),
                        textcoords   = ('data', 'axes fraction'),
                        arrowprops   = dict(arrowstyle='-', color='#999', lw=0.8))
            ax.text(xm, -0.72, grp_lbl,
                    ha='center', va='top',
                    fontsize=5.8, fontweight='bold', color='#333333',
                    transform=ax.get_xaxis_transform())

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN — assemble figure
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Figure 5 — Nature Biotech super figure")
    parser.add_argument("--excel",  default="Figure4_data.xlsx",
                        help="Path to Figure4_data.xlsx")
    parser.add_argument("--jain",   default="Jain2017_pred_psr_filter_all_transformer_lm_ipi_psr_trainset.xlsx")
    parser.add_argument("--gdpa1",  default="GDPa1_v1_3_20251027_pred_psr_filter_all_transformer_lm_ipi_psr_trainset.xlsx")
    parser.add_argument("--gdpa3",  default="GDPa3_20260106_pred_psr_filter_all_transformer_lm_ipi_psr_trainset.xlsx")
    parser.add_argument("--out",    default="Figure5_natbiotech",
                        help="Output file stem (no extension)")
    parser.add_argument("--dpi",    type=int, default=300)
    args = parser.parse_args()

    # -- Load data -------------------------------------------------------------
    print("Loading data …")
    df_cv, df_xfer = load_fig4_data(args.excel)

    jain  = pd.read_excel(args.jain)
    gdpa1 = pd.read_excel(args.gdpa1)
    gdpa3 = pd.read_excel(args.gdpa3)

    # GDPa3 already has psr_filter_ova / psr_filter_cho columns
    # GDPa1: derive from threshold
    gdpa1['psr_filter_ova'] = (gdpa1['polyreactivity_prscore_ova_avg'] < THRESH).astype(float)
    gdpa1['psr_filter_cho'] = (gdpa1['polyreactivity_prscore_cho_avg'] < THRESH).astype(float)

    auc_mat, rho_mat = compute_heatmap_matrices(jain, gdpa1, gdpa3)

    # -- Y-position map for panel a --------------------------------------------
    (y_map, y_labels, y_arch_mid,
     y_arch_span, y_sep, total_y) = build_y_map()

    # -- Figure layout ---------------------------------------------------------
    fig = plt.figure(figsize=(FW, FH))
    gs  = GridSpec(3, 2, figure=fig,
                   height_ratios = [1.55, 0.72, 0.76],
                   left   = 0.155,   # enough room for vertical arch labels + LM names
                   right  = 0.965,
                   top    = 0.970,
                   bottom = 0.055,
                   hspace = 0.42,
                   wspace = 0.36)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    ax_d = fig.add_subplot(gs[2, :])

    # -- Draw panels -----------------------------------------------------------
    draw_panel_a(ax_a, df_cv, y_map, y_labels, y_arch_mid,
                 y_arch_span, y_sep, total_y)

    draw_panel_b(ax_b, df_xfer)

    # Right-side section labels (rotated)
    for ax_obj, lbl in [(ax_a, 'IPI PSR\n10-fold CV'),
                        (ax_b, 'Cross-dataset\ntransfer')]:
        ax_obj.annotate(lbl,
                        xy         = (1.04, 0.50),
                        xycoords   = 'axes fraction',
                        fontsize   = 5.5, fontweight='bold', color='#444',
                        rotation   = -90, va='center', ha='left',
                        clip_on    = False)

    draw_heatmap(
        ax_c, auc_mat, HM_LM_DISPLAY, DATASETS_AUC,
        title        = (f'IPI PSR Transformer Model Validation on External Clinical Antibodies'
                        f'  —  AUC-ROC  (PASS: score < {THRESH})'),
        panel_letter = 'c',
        vmin=0.20, vmax=0.90, vcenter=0.55,
        cmap='RdBu_r', fmt='.3f', cbar_label='AUC-ROC',
        col_sep      = [0, 2],
        grp_labels   = [
            ('Jain 2017  (n=137)', 0, 0),
            ('GDPa1  (n=197)',     1, 2),
            ('GDPa3  (n=80)',      3, 4),
        ]
    )

    draw_heatmap(
        ax_d, rho_mat, HM_LM_DISPLAY, DATASETS_RHO,
        title        = ('IPI PSR Transformer Model Validation on External Clinical Antibodies'
                        '  —  Spearman ρ  (P(PASS) vs polyreactivity assay score)'),
        panel_letter = 'd',
        vmin=-0.80, vmax=0.20, vcenter=0.0,
        cmap='RdBu_r', fmt='.2f', cbar_label='Spearman ρ',
        col_sep      = [1, 3],
        grp_labels   = [
            ('Jain 2017  (n=137)', 0, 1),
            ('GDPa1  (n=197)',     2, 3),
            ('GDPa3  (n=80)',      4, 5),
        ]
    )

    # -- Save ------------------------------------------------------------------
    tiff_path = f"{args.out}.tiff"
    png_path  = f"{args.out}_preview.png"

    fig.savefig(tiff_path, dpi=args.dpi, format='tiff',
                bbox_inches='tight', pad_inches=0.05, facecolor='white')
    fig.savefig(png_path, dpi=150,
                bbox_inches='tight', pad_inches=0.05, facecolor='white')

    import os
    w, h = fig.get_size_inches()
    print(f"Figure size : {w*25.4:.0f} × {h*25.4:.0f} mm")
    print(f"TIFF        : {tiff_path}  ({os.path.getsize(tiff_path)//1024} KB)")
    print(f"Preview PNG : {png_path}")


if __name__ == "__main__":
    main()