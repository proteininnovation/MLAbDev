#!/usr/bin/env python3
"""
plot_biophysical.py — Biophysical property panel figures for prediction output
IPI Antibody Developability Prediction Platform

Identical style to ipiabdev_figures.py (same rcParams, seaborn histplot,
COLOR_PASS, COLOR_FAIL, panel labels, legend helpers).

Two TIFF figures are produced:
  *_biophysical_{label}_true.tiff  — split by ground-truth label
  *_biophysical_{label}_pred.tiff  — split by predicted (optimallabel) column

Output: TIFF only (300 DPI). No PDF, no PNG.

Usage (standalone)
------------------
    python utils/plot_biophysical.py \\
        --file   DS1_pred_psr_filter_antiberta2-cssp_transformer_lm_ipi_psr_trainset.xlsx \\
        --target psr_filter

Called automatically from predict_developability.py when --predict finds
a ground-truth label column in the input file.
"""

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Import liabilities module ─────────────────────────────────────────────────
_liabilities = None
for _lib_path in [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'liabilities.py'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils', 'liabilities.py'),
]:
    if os.path.exists(_lib_path):
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location('liabilities', _lib_path)
        _liabilities = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_liabilities)
        break

if _liabilities is None:
    try:
        import liabilities as _liabilities
    except ImportError:
        _liabilities = None

_HAS_LIABILITIES = _liabilities is not None
if not _HAS_LIABILITIES:
    print("[biophys] WARNING: liabilities module not found.")


# ── Nature Biotechnology style constants — identical to ipiabdev_figures.py ───
MM_TO_INCH  = 1 / 25.4
SINGLE_COL  = 89  * MM_TO_INCH
DOUBLE_COL  = 183 * MM_TO_INCH
MAX_DEPTH   = 247 * MM_TO_INCH

DPI_SUBMIT  = 300

FONT_FAMILY = "Arial"
SIZE_PANEL  = 8
SIZE_AXIS   = 7
SIZE_TICK   = 6
SIZE_LEGEND = 6

COLOR_PASS  = "#4C9BE8"   # blue   — Pass (1)
COLOR_FAIL  = "#F28C38"   # orange — Fail (0)
ALPHA       = 0.75
LINEWIDTH   = 0.6


def set_nature_style():
    """Apply Nature Biotechnology-compliant rcParams — identical to ipiabdev_figures.py."""
    matplotlib.rcParams.update({
        "font.family"         : "sans-serif",
        "font.sans-serif"     : [FONT_FAMILY, "Helvetica", "DejaVu Sans"],
        "font.size"           : SIZE_TICK,
        "axes.labelsize"      : SIZE_AXIS,
        "axes.titlesize"      : SIZE_AXIS,
        "axes.linewidth"      : LINEWIDTH,
        "axes.spines.top"     : False,
        "axes.spines.right"   : False,
        "xtick.labelsize"     : SIZE_TICK,
        "ytick.labelsize"     : SIZE_TICK,
        "xtick.major.width"   : LINEWIDTH,
        "ytick.major.width"   : LINEWIDTH,
        "xtick.major.size"    : 2.5,
        "ytick.major.size"    : 2.5,
        "xtick.direction"     : "out",
        "ytick.direction"     : "out",
        "legend.fontsize"     : SIZE_LEGEND,
        "legend.frameon"      : False,
        "legend.handlelength" : 1.0,
        "legend.handleheight" : 0.7,
        "legend.handletextpad": 0.4,
        "legend.borderpad"    : 0.3,
        "savefig.dpi"         : DPI_SUBMIT,
        "pdf.fonttype"        : 42,
        "ps.fonttype"         : 42,
    })


def _legend_handles(label_pass="Pass (1)", label_fail="Fail (0)"):
    """Proxy Line2D artists for the shared legend — identical to ipiabdev_figures.py."""
    return [
        Line2D([0], [0], color=COLOR_PASS, lw=4, alpha=ALPHA, label=label_pass),
        Line2D([0], [0], color=COLOR_FAIL, lw=4, alpha=ALPHA, label=label_fail),
    ]


def _panel_label(ax, letter, x=-0.22, y=1.10):
    """Bold lowercase panel label — identical to ipiabdev_figures.py."""
    ax.text(x, y, letter,
            transform=ax.transAxes,
            fontsize=SIZE_PANEL, fontweight="bold",
            va="top", ha="left", fontfamily=FONT_FAMILY)


def _hist_panel(ax, data, col_filter, val_pass, val_fail,
                x_var, x_label,
                discrete=True, bins=None, x_lim=None,
                panel_letter=None, show_legend=False):
    """
    Overlapping density histograms for Pass vs Fail.
    Identical signature and implementation to ipiabdev_figures.py.
    Uses seaborn histplot (stat='density').
    """
    d_pass = data[data[col_filter] == val_pass]
    d_fail = data[data[col_filter] == val_fail]

    hist_kw = dict(stat="density", alpha=ALPHA, linewidth=0)

    if bins is not None:
        sns.histplot(d_pass, x=x_var, bins=bins, color=COLOR_PASS, ax=ax, **hist_kw)
        sns.histplot(d_fail, x=x_var, bins=bins, color=COLOR_FAIL, ax=ax, **hist_kw)
    else:
        sns.histplot(d_pass, x=x_var, discrete=discrete, color=COLOR_PASS, ax=ax, **hist_kw)
        sns.histplot(d_fail, x=x_var, discrete=discrete, color=COLOR_FAIL, ax=ax, **hist_kw)

    ax.set_xlabel(x_label,    fontsize=SIZE_AXIS, labelpad=2)
    ax.set_ylabel("Density",  fontsize=SIZE_AXIS, labelpad=2)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    if x_lim:
        ax.set_xlim(x_lim)

    if show_legend:
        ax.legend(handles=_legend_handles(), loc="upper right", fontsize=SIZE_LEGEND)
    elif ax.get_legend():
        ax.get_legend().remove()

    if panel_letter:
        _panel_label(ax, panel_letter)

    ax.tick_params(axis="both", labelsize=SIZE_TICK, width=LINEWIDTH, length=2.5)


# ── Panel definitions — identical to generate_extended_figure1 ────────────────
# (col, x_label, subset_R1, discrete, bins, x_lim)
_PANELS = [
    ("R",                     "Arginine count (HCDR3)",                False, True,  None,            None),
    ("D",                     "Aspartic acid count (HCDR3)",           False, True,  None,            None),
    ("CDR3_W_count",          "Tryptophan count (HCDR3)\n(Arg count=1)", True, True, None,            None),
    ("E",                     "Glutamic acid count (HCDR3)",           False, True,  None,            None),
    ("CDR3_len",              "HCDR3 loop length",                      False, True,  None,            None),
    ("HCDR3_charge",          "Net charge (HCDR3)",                    False, True,  None,            None),
    ("HCDR3_isoelectricpoint","Isoelectric point (HCDR3)",             False, True,  None,            None),
    ("VH_isoelectricpoint",   "Isoelectric point\n(heavy chain)",      False, False, list(range(1,12)), None),
]


# ── Annotation ────────────────────────────────────────────────────────────────

def annotate_biophysical(df, cdr3_col='CDR3', hseq_col='HSEQ'):
    """
    Annotate with liabilities module + AA count columns.
    Mirrors load_* functions in ipiabdev_figures.py exactly.
    """
    if not _HAS_LIABILITIES:
        return df

    df = df.copy()

    # Strip leading C — same as in load_ipi_psr_trainset / load_ds1
    if cdr3_col in df.columns:
        mask = df[cdr3_col].str.startswith('C', na=False)
        if mask.mean() > 0.90:
            df.loc[mask, cdr3_col] = df.loc[mask, cdr3_col].str[1:]
            print(f"[biophys] Stripped leading C from '{cdr3_col}'")

    # VH annotation
    if hseq_col in df.columns:
        try:
            df = _liabilities.annotate_liabilities_2(df, cdr3_col=hseq_col, label='VH')
        except Exception as e:
            print(f"[biophys] VH annotation: {e}")

    # HCDR3 annotation
    if cdr3_col in df.columns:
        try:
            df = _liabilities.annotate_liabilities_2(df, cdr3_col=cdr3_col, label='HCDR3')
        except Exception as e:
            print(f"[biophys] HCDR3 annotation: {e}")

    # AA count columns — same as _prepare_data in ipiabdev_figures.py
    if cdr3_col in df.columns:
        cdr3 = df[cdr3_col].fillna('').astype(str).str.upper()
        df['R']            = cdr3.str.count('R')
        df['D']            = cdr3.str.count('D')
        df['E']            = cdr3.str.count('E')
        df['CDR3_W_count'] = cdr3.str.count('W')
        df['CDR3_len']     = cdr3.str.len()

    return df


# ── Figure builder ────────────────────────────────────────────────────────────

def _build_figure(df, label_col, val_pass, val_fail, title,
                  pass_label="Pass (1)", fail_label="Fail (0)"):
    """
    Build 2×4 biophysical panel figure.

    Layout (GridSpec):
      Col 0           = dedicated legend axis (left of panel a)
      Cols 1–4 row 0  = panels a b c d
      Cols 1–4 row 1  = panels e f g h

    Style: seaborn histplot, same as generate_extended_figure1.
    """
    set_nature_style()

    panels = [(c,xl,r1,d,bins,xlim)
              for c,xl,r1,d,bins,xlim in _PANELS if c in df.columns]
    n_panels = len(panels)

    if n_panels == 0:
        print(f"[biophys] No biophysical columns for '{label_col}'.")
        return None

    N_COLS = 4
    n_rows = int(np.ceil(n_panels / N_COLS))
    fig_w  = DOUBLE_COL
    fig_h  = min(fig_w * 0.52, MAX_DEPTH)

    n1 = (df[label_col] == val_pass).sum()
    n0 = (df[label_col] == val_fail).sum()

    fig = plt.figure(figsize=(fig_w, fig_h))

    # GridSpec: col 0 = legend, cols 1..N_COLS = data
    gs = GridSpec(
        n_rows, N_COLS + 1,
        figure=fig,
        width_ratios=[0.14] + [1.0] * N_COLS,
        left=0.03, right=0.99,
        top=0.90,  bottom=0.14,
        hspace=0.55,
        wspace=0.68,
    )

    # ── Dedicated legend axis (left column, all rows) ─────────────────────────
    ax_leg = fig.add_subplot(gs[:, 0])
    ax_leg.axis('off')
    ax_leg.legend(
        handles=_legend_handles(
            label_pass=f"{pass_label}\n(n={n1:,})",
            label_fail=f"{fail_label}\n(n={n0:,})",
        ),
        loc='center',
        fontsize=SIZE_LEGEND,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.5,
        labelspacing=1.0,
    )

    # ── Data panels ───────────────────────────────────────────────────────────
    axes_grid = [fig.add_subplot(gs[r, c + 1])
                 for r in range(n_rows) for c in range(N_COLS)]

    letters = list('abcdefghijklmnop')

    for idx, (col, x_label, subset_r1, discrete, bins, x_lim) in enumerate(panels):
        ax  = axes_grid[idx]
        sub = df[df['R'] == 1] if (subset_r1 and 'R' in df.columns) else df

        _hist_panel(ax, sub, label_col, val_pass, val_fail,
                    col, x_label,
                    discrete=discrete, bins=bins, x_lim=x_lim,
                    panel_letter=letters[idx],
                    show_legend=False)

    for idx in range(n_panels, len(axes_grid)):
        axes_grid[idx].set_visible(False)

    fig.suptitle(title, fontsize=SIZE_AXIS, fontweight='bold', y=0.97)
    return fig


def _save_tiff(fig, out_tiff):
    """Save as TIFF only — no PDF, no PNG."""
    fig.savefig(out_tiff, dpi=DPI_SUBMIT, format='tiff',
                bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    plt.rcdefaults()
    print(f"[biophys] TIFF 300 DPI → {out_tiff}")


# ── Main entry point ──────────────────────────────────────────────────────────

def plot_biophysical_report(
    file,
    target,
    score_col=None,
    out=None,
    test_target=None,
    dataset_name=None,
    cdr3_col='CDR3',
    hseq_col='HSEQ',
    max_cdr3_len=25,
    val_pass=1,
    val_fail=0,
):
    """
    Generate two biophysical TIFF figures from a prediction output file.

      Figure A — split by ground-truth label  (*_biophysical_{label}_true.tiff)
      Figure B — split by predicted label     (*_biophysical_{label}_pred.tiff)

    Parameters
    ----------
    file         : prediction output file (.xlsx or .csv)
    target       : ground-truth label column (e.g. 'psr_filter')
    score_col    : predicted label column — auto-detected if None
    out          : output file stem (default: derived from file)
    test_target  : alternative ground-truth column (--test_target)
    dataset_name : clean dataset name for figure title
    cdr3_col     : CDR3 column (default 'CDR3')
    hseq_col     : heavy chain sequence column (default 'HSEQ')
    max_cdr3_len : CDR3 length filter (removes artefacts)
    val_pass / val_fail : label values for Pass/Fail (default 1/0)
    """
    # ── Load ──────────────────────────────────────────────────────────────────
    ext = os.path.splitext(file)[1].lower()
    df  = pd.read_excel(file) if ext in ('.xlsx', '.xls') else pd.read_csv(file)

    # Resolve ground-truth column
    _label_col = (test_target if (test_target and test_target in df.columns)
                  else target)
    if _label_col not in df.columns:
        print(f"[biophys] Label column '{_label_col}' not found — skipping.")
        return

    df[_label_col] = pd.to_numeric(df[_label_col], errors='coerce')
    df = df.dropna(subset=[_label_col])
    df[_label_col] = df[_label_col].astype(int)

    # Resolve predicted label column (auto-detect *_optimallabel → *_label)
    _pred_col = None
    if score_col and score_col in df.columns:
        _pred_col = score_col
    else:
        for col in df.columns:
            if col.endswith('_optimallabel'):
                _pred_col = col; break
    if _pred_col is None:
        for col in df.columns:
            if col.endswith('_label') and col != _label_col:
                _pred_col = col; break

    if _pred_col:
        print(f"[biophys] Predicted label column: '{_pred_col}'")
    else:
        print("[biophys] No predicted label column found — Figure B skipped.")

    # ── Annotate ──────────────────────────────────────────────────────────────
    if not _HAS_LIABILITIES:
        print("[biophys] liabilities module not available — skipping.")
        return

    print("[biophys] Annotating biophysical properties...")
    df = annotate_biophysical(df, cdr3_col=cdr3_col, hseq_col=hseq_col)

    if 'CDR3_len' in df.columns:
        before = len(df)
        df     = df[df['CDR3_len'] < max_cdr3_len]
        print(f"[biophys] CDR3 filter (<{max_cdr3_len}): {before:,} → {len(df):,}")

    n1 = (df[_label_col] == val_pass).sum()
    n0 = (df[_label_col] == val_fail).sum()
    print(f"[biophys] n={len(df):,}  pass={n1:,}  fail={n0:,}  label='{_label_col}'")

    # ── Output stem ───────────────────────────────────────────────────────────
    stem = os.path.splitext(out or file)[0].replace(' ', '_')
    ds   = dataset_name or Path(file).stem
    target_label = ('PSR polyreactivity' if 'psr' in _label_col.lower()
                    else 'SEC aggregation' if 'sec' in _label_col.lower()
                    else _label_col)

    # ── Figure A: ground-truth label ──────────────────────────────────────────
    print(f"\n[biophys] Figure A — ground-truth: '{_label_col}'")
    title_a = (f"{ds}  |  {target_label}  |  Ground-truth  '{_label_col}'"
               f"  (Pass n={n1:,}, Fail n={n0:,})")
    fig_a = _build_figure(df, _label_col, val_pass, val_fail, title_a,
                          pass_label="Pass (1) — non-polyreactive",
                          fail_label="Fail (0) — polyreactive")
    if fig_a:
        _save_tiff(fig_a, f"{stem}_biophysical_{_label_col}_true.tiff")

    # ── Figure B: predicted label ─────────────────────────────────────────────
    if _pred_col and _pred_col in df.columns:
        df[_pred_col] = pd.to_numeric(df[_pred_col], errors='coerce')
        df_pred = df.dropna(subset=[_pred_col]).copy()
        df_pred[_pred_col] = df_pred[_pred_col].astype(int)
        np1 = (df_pred[_pred_col] == val_pass).sum()
        np0 = (df_pred[_pred_col] == val_fail).sum()

        _short = (_pred_col.replace('_optimallabel', ' (Youden)')
                            .replace('_label', ' (t=0.5)'))
        print(f"\n[biophys] Figure B — predicted: '{_pred_col}'")
        title_b = (f"{ds}  |  {target_label}  |  Predicted  '{_short}'"
                   f"  (Pass n={np1:,}, Fail n={np0:,})")
        fig_b = _build_figure(df_pred, _pred_col, val_pass, val_fail, title_b,
                              pass_label="Pass (1) — predicted",
                              fail_label="Fail (0) — predicted")
        if fig_b:
            _save_tiff(fig_b, f"{stem}_biophysical_{_label_col}_pred.tiff")
    else:
        print("[biophys] Figure B skipped — no predicted label column.")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Biophysical property panels from a prediction file'
    )
    parser.add_argument('--file',        required=True)
    parser.add_argument('--target',      required=True)
    parser.add_argument('--score_col',   default=None)
    parser.add_argument('--test_target', default=None)
    parser.add_argument('--cdr3_col',    default='CDR3')
    parser.add_argument('--hseq_col',    default='HSEQ')
    parser.add_argument('--out',         default=None)
    args = parser.parse_args()

    plot_biophysical_report(
        file        = args.file,
        target      = args.target,
        score_col   = args.score_col,
        test_target = args.test_target,
        cdr3_col    = args.cdr3_col,
        hseq_col    = args.hseq_col,
        out         = args.out,
    )