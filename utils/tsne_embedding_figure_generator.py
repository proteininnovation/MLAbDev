"""
IPIAbDev — Figure 3: t-SNE PLM Embedding Quality (Nature Biotechnology format)
================================================================================
Generates Figure 3: t-SNE visualizations of IPI PSR antibody embeddings
colored by VH germline (top row, panels a–e) and VL germline (bottom row,
panels f–j) for five PLMs:
    AbLang2 | AntiBERTy | AntiBERTa2 | AntiBERTa2-CSSP | IgBert

Layout  : 2 rows × 5 columns = 10 panels
Width   : double column (183 mm)
Colors  : Qualitative palette (colorblind-safe, up to 9 germlines)

Nature Biotechnology figure specifications:
  - Font       : Arial, 7pt axis labels, 6pt tick labels, 8pt panel letters
  - Resolution : 300 dpi (TIFF) submission | 600 dpi final
  - Width      : double column = 183 mm
  - Format     : TIFF + PDF

Changes from 4-PLM version
---------------------------
  1.  plm_names default      : added "igbert"
  2.  plm_labels default     : added "IgBert"
  3.  panel_letters_top      : "abcd"  → "abcde"
  4.  panel_letters_bot      : "efgh"  → "fghij"
  5.  GridSpec wspace        : 0.45   → 0.38   (tighter for 5 cols)
  6.  fig_h ratio            : 0.50   → 0.46   (stay within MAX_DEPTH)
  7.  legend bbox_to_anchor x: 0.895  → 0.885
  8.  compute_tsne docstring : IgBert 1,024-dim note added

Required embedding file (generate with --build-embedding):
  {DATA_FILE}.igbert.emb.csv   (1,024-dim, BARCODE index)

Usage
-----
    python ipiabdev_tsne_figure3_igbert.py

Or import:
    from ipiabdev_tsne_figure3_igbert import generate_figure3
    generate_figure3(main_path=..., data_file=...)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─── Paths ───────────────────────────────────────────────────────────────────
MAIN_PATH  = "/Users/Hoan.Nguyen/ComBio/MachineLearning/IPIPred/"
DATA_FILE  = "data/ipi_antibodydb.xlsx"
OUT_DIR    = Path("images/figures_natbiotech")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Nature Biotechnology style constants ────────────────────────────────────
MM_TO_INCH  = 1 / 25.4
DOUBLE_COL  = 183 * MM_TO_INCH
MAX_DEPTH   = 247 * MM_TO_INCH

DPI_SUBMIT  = 300
DPI_FINAL   = 600

FONT_FAMILY = "Arial"
SIZE_PANEL  = 8     # panel letter (a, b ...)
SIZE_AXIS   = 7     # axis title
SIZE_TICK   = 6     # tick labels
SIZE_LEGEND = 6     # legend text
SIZE_TITLE  = 7     # PLM column title
LINEWIDTH   = 0.6
MARKER_SIZE = 1.5   # pt
ALPHA       = 0.7

# Colorblind-safe qualitative palette (Wong 2011 + extended)
GERMLINE_COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#999999",  # grey
    "#000000",  # black
]


# ─── Global matplotlib style ─────────────────────────────────────────────────
def set_nature_style():
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
        "xtick.major.size"    : 2.0,
        "ytick.major.size"    : 2.0,
        "xtick.direction"     : "out",
        "ytick.direction"     : "out",
        "legend.fontsize"     : SIZE_LEGEND,
        "legend.frameon"      : False,
        "figure.dpi"          : 150,
        "pdf.fonttype"        : 42,
        "ps.fonttype"         : 42,
    })

set_nature_style()


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_ipi_elisa_dataset(main_path: str, data_file: str = DATA_FILE) -> pd.DataFrame:
    """
    Load IPI PSR ELISA-only dataset with denoising filters.
    Returns ~8,019 high-confidence antibodies with germline annotations.

    Filters applied (matching Online Methods Section A):
      - All 4 ELISA scores present (no NA)
      - psr_filter label present and consistent with RF prediction
      - Excludes test antigens
      - ID2 > 8700
    """
    data = pd.read_excel(os.path.join(main_path, data_file))
    data = data[
        pd.notna(data["psr_norm_insulin"]) &
        pd.notna(data["psr_norm_dna"])     &
        pd.notna(data["psr_norm_smp"])     &
        pd.notna(data["psr_norm_avidin"])
    ]
    data = data[pd.notna(data["psr_filter"])]
    data = data[data["psr_filter"] == data["psr_rf_elisa"]]
    data = data[~data["antigen"].str.contains("test", na=False, case=False)]
    data = data[data["ID2"] > 8700]
    data = data.reset_index(drop=True)
    print(f"  Loaded {len(data):,} antibodies after denoising filters")
    return data


def load_embedding(main_path: str, data_file: str, plm_name: str) -> pd.DataFrame:
    """
    Load pre-computed PLM embedding CSV indexed by BARCODE.

    Naming convention: {data_file}.{plm_name}.emb.csv

    Embedding dimensions per PLM:
      ablang          : 480-dim
      antiberty       : 512-dim
      antiberta2      : 1,024-dim
      antiberta2-cssp : 1,024-dim
      igbert          : 1,024-dim   ← NEW
    """
    path = os.path.join(main_path, f"{data_file}.{plm_name}.emb.csv")
    emb  = pd.read_csv(path).set_index("BARCODE")
    print(f"  Loaded {plm_name} embedding: {emb.shape}")
    return emb


def align_data_embedding(
    data: pd.DataFrame, emb: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Align antibody metadata with embedding matrix by BARCODE.
    Returns aligned data DataFrame and embedding numpy array.
    """
    common       = list(set(data["BARCODE"]).intersection(emb.index))
    data_aligned = data.set_index("BARCODE").loc[common].reset_index()
    emb_aligned  = emb.loc[common].values
    return data_aligned, emb_aligned


# ─── t-SNE computation ────────────────────────────────────────────────────────
def compute_tsne(
    X: np.ndarray,
    n_components: int = 2,
    perplexity: float = 7,
    learning_rate: str = "auto",
    max_iter: int = 1000,
    early_exaggeration: float = 12.0,
    metric: str = "euclidean",
    random_state: int = 42,
    scale: bool = False,
) -> np.ndarray:
    """
    Compute t-SNE on embedding matrix.

    Default parameters match the original IPIAbDev publication figure:
      TSNE(n_components=2, perplexity=7, learning_rate='auto',
           random_state=42, init='pca')

      - perplexity=7      : LOW is correct for IPI fixed-framework library.
                            Frameworks are fixed per germline — only HCDR3
                            varies — so antibodies within a germline are
                            genuinely tight neighbors.
      - metric='euclidean': sklearn default — enables init='pca'
      - init='pca'        : stable, reproducible initialization
                            (only valid with euclidean metric)
      - learning_rate='auto': sklearn sets lr = max(n/early_exag/4, 50)
      - max_iter=1000     : sklearn default; sufficient at perplexity=7
      - scale=False       : no StandardScaler (matches original code)

    IgBert note (CHANGE 8):
      IgBert produces 1,024-dim embeddings — identical dimensionality to
      AntiBERTa2 and AntiBERTa2-CSSP. No change to any t-SNE parameter
      is required. PCA initialisation (init='pca') is stable at 1,024-dim
      and fast because sklearn uses a randomized SVD internally.

    Note: if metric='cosine' is passed, init is automatically switched
    to 'random' since PCA init requires euclidean metric.
    """
    if scale:
        X = StandardScaler().fit_transform(X)

    init = "pca" if metric == "euclidean" else "random"

    tsne = TSNE(
        n_components       = n_components,
        perplexity         = perplexity,
        learning_rate      = learning_rate,
        max_iter           = max_iter,
        early_exaggeration = early_exaggeration,
        metric             = metric,
        random_state       = random_state,
        init               = init,
    )
    X_tsne = tsne.fit_transform(X)
    print(f"    t-SNE KL divergence: {tsne.kl_divergence_:.4f}")
    return X_tsne


# ─── Single panel helper ──────────────────────────────────────────────────────
def _tsne_panel(
    ax,
    X_tsne: np.ndarray,
    labels: pd.Series,
    color_map: dict,
    panel_letter: str = None,
    show_legend: bool = False,
    legend_title: str = "",
    title: str = "",
):
    """
    Plot t-SNE scatter colored by germline label on a single Axes.

    Parameters
    ----------
    ax           : matplotlib Axes
    X_tsne       : (n, 2) array of t-SNE coordinates
    labels       : pd.Series of germline labels (e.g. 'IGHV3-23')
    color_map    : dict mapping label → color
    panel_letter : e.g. 'a'
    show_legend  : bool
    legend_title : legend header text
    title        : column title (PLM name), shown above top row only
    """
    unique_labels = sorted(labels.unique())

    for germ in unique_labels:
        mask = labels == germ
        ax.scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            c          = color_map[germ],
            s          = MARKER_SIZE,
            alpha      = ALPHA,
            linewidths = 0,
            label      = germ,
            rasterized = True,
        )

    ax.set_xlabel("t-SNE 1", fontsize=SIZE_AXIS, labelpad=2)
    ax.set_ylabel("t-SNE 2", fontsize=SIZE_AXIS, labelpad=2)
    ax.tick_params(axis="both", labelsize=SIZE_TICK, width=LINEWIDTH, length=2.0)
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.set_xticks([]);       ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=SIZE_TITLE, fontweight="bold",
                     pad=4, fontfamily=FONT_FAMILY)

    if panel_letter:
        ax.text(
            -0.12, 1.08, panel_letter,
            transform  = ax.transAxes,
            fontsize   = SIZE_PANEL,
            fontweight = "bold",
            va="top", ha="left",
            fontfamily = FONT_FAMILY,
        )

    if show_legend:
        handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_map[g], markersize=4,
                   alpha=ALPHA, label=g)
            for g in unique_labels
        ]
        leg = ax.legend(
            handles        = handles,
            fontsize       = SIZE_LEGEND,
            loc            = "upper left",
            bbox_to_anchor = (1.08, 1.0),
            borderaxespad  = 0,
            handlelength   = 0.8,
            handletextpad  = 0.3,
            borderpad      = 0.3,
            labelspacing   = 0.2,
            frameon        = False,
        )
        leg.set_title(legend_title)
        leg.get_title().set_fontsize(SIZE_LEGEND)
        leg.get_title().set_fontweight("bold")
        leg.get_title().set_fontfamily(FONT_FAMILY)


# ─── FIGURE 3 ─────────────────────────────────────────────────────────────────
def generate_figure3(
    main_path: str = MAIN_PATH,
    data_file: str = DATA_FILE,
    plm_names: list = None,
    plm_labels: list = None,
    vh_col: str = "heavy",
    vl_col: str = "light",
    tsne_params: dict = None,
    output_prefix: str = "Figure3",
    dpi: int = DPI_SUBMIT,
):
    """
    Figure 3 — PLM embedding quality (t-SNE of IPI PSR antibodies).

    Top row    (a–e) : colored by VH germline
    Bottom row (f–j) : colored by VL germline
    Columns: AbLang2 | AntiBERTy | AntiBERTa2 | AntiBERTa2-CSSP | IgBert

    Parameters
    ----------
    main_path     : root data path
    data_file     : relative path to data Excel/CSV file from main_path
    plm_names     : list of PLM embedding file suffixes
                    Default: ["ablang","antiberty","antiberta2",
                              "antiberta2-cssp","igbert"]
    plm_labels    : display names for column headers
                    Default: ["AbLang2","AntiBERTy","AntiBERTa2",
                              "AntiBERTa2-CSSP","IgBert"]
    vh_col        : DataFrame column name for VH germline annotation
    vl_col        : DataFrame column name for VL germline annotation
    tsne_params   : dict of t-SNE parameters (see compute_tsne defaults)
    output_prefix : output filename prefix
    dpi           : output resolution (300 submit, 600 final)
    """
    # ── CHANGES 1 & 2: default PLM lists now include igbert ──────────────────
    if plm_names is None:
        plm_names = [
            "ablang",
            "antiberty",
            "antiberta2",
            "antiberta2-cssp",
            "igbert",           # NEW
        ]
    if plm_labels is None:
        plm_labels = [
            "AbLang2",
            "AntiBERTy",
            "AntiBERTa2",
            "AntiBERTa2-CSSP",
            "IgBert",           # NEW
        ]
    if tsne_params is None:
        tsne_params = {}

    n_cols = len(plm_names)   # 5

    # ── CHANGES 3 & 4: panel letters for 5 columns ───────────────────────────
    panel_letters_top = list("abcde")   # was "abcd"
    panel_letters_bot = list("fghij")   # was "efgh"

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading IPI PSR ELISA dataset...")
    data = load_ipi_elisa_dataset(main_path, data_file)

    # ── Germline color maps ───────────────────────────────────────────────────
    vh_germlines = sorted(data[vh_col].dropna().unique())
    vl_germlines = sorted(data[vl_col].dropna().unique())

    vh_color_map = {g: GERMLINE_COLORS[i % len(GERMLINE_COLORS)]
                    for i, g in enumerate(vh_germlines)}
    vl_color_map = {g: GERMLINE_COLORS[i % len(GERMLINE_COLORS)]
                    for i, g in enumerate(vl_germlines)}

    print(f"  VH germlines ({len(vh_germlines)}): {vh_germlines}")
    print(f"  VL germlines ({len(vl_germlines)}): {vl_germlines}")

    # ── t-SNE for each PLM ────────────────────────────────────────────────────
    tsne_results = {}
    aligned_data = {}

    for plm in plm_names:
        print(f"\nProcessing {plm}...")
        emb  = load_embedding(main_path, data_file, plm)
        d, X = align_data_embedding(data, emb)
        print(f"  Aligned: {len(d):,} antibodies × {X.shape[1]} dimensions")
        print(f"  Computing t-SNE...")
        tsne_results[plm] = compute_tsne(X, **tsne_params)
        aligned_data[plm] = d

    # ── Figure geometry ───────────────────────────────────────────────────────
    fig_w = DOUBLE_COL
    # CHANGE 6: ratio 0.50 → 0.46 to stay within MAX_DEPTH with 5 columns
    fig_h = min(fig_w * 0.46, MAX_DEPTH)

    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = GridSpec(
        2, n_cols + 1,               # 2 rows × 6 cols (1 label + 5 data)
        figure       = fig,
        width_ratios = [0.10] + [1] * n_cols,
        left    = 0.04, right  = 0.88,
        top     = 0.91, bottom = 0.08,
        hspace  = 0.40,
        wspace  = 0.38,   # CHANGE 5: was 0.45 — tighter for 5th column
    )

    # Row label axes — invisible, text only
    for row, label in enumerate(["VH\ngermline", "VL\ngermline"]):
        ax_lbl = fig.add_subplot(gs[row, 0])
        ax_lbl.axis("off")
        ax_lbl.text(
            0.9, 0.5, label,
            transform   = ax_lbl.transAxes,
            fontsize    = SIZE_AXIS,
            fontweight  = "bold",
            rotation    = 90,
            va="center", ha="center",
            fontfamily  = FONT_FAMILY,
            linespacing = 1.4,
        )

    # Data panel axes — columns 1 to n_cols
    axes = np.array([
        [fig.add_subplot(gs[row, col + 1]) for col in range(n_cols)]
        for row in range(2)
    ])

    for col_idx, (plm, plm_label) in enumerate(zip(plm_names, plm_labels)):
        X_tsne = tsne_results[plm]
        d      = aligned_data[plm]

        # Top row — VH germline
        _tsne_panel(
            axes[0, col_idx],
            X_tsne,
            labels       = d[vh_col].astype(str),
            color_map    = vh_color_map,
            panel_letter = panel_letters_top[col_idx],
            show_legend  = False,
            legend_title = "VH germline",
            title        = plm_label,
        )

        # Bottom row — VL germline
        _tsne_panel(
            axes[1, col_idx],
            X_tsne,
            labels       = d[vl_col].astype(str),
            color_map    = vl_color_map,
            panel_letter = panel_letters_bot[col_idx],
            show_legend  = False,
            legend_title = "VL germline",
            title        = "",
        )

    # ── Figure-level legends — fixed figure coordinates ───────────────────────
    def _make_legend_handles(color_map, unique_labels):
        return [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_map[g], markersize=4,
                   alpha=ALPHA, label=g)
            for g in sorted(unique_labels)
        ]

    vh_labels = sorted(aligned_data[plm_names[0]][vh_col].astype(str).unique())
    vl_labels = sorted(aligned_data[plm_names[0]][vl_col].astype(str).unique())

    vh_handles = _make_legend_handles(vh_color_map, vh_labels)
    vl_handles = _make_legend_handles(vl_color_map, vl_labels)

    for handles, title_text, y_pos in [
        (vh_handles, "VH germline", 0.93),
        (vl_handles, "VL germline", 0.46),
    ]:
        leg = fig.legend(
            handles        = handles,
            fontsize       = SIZE_LEGEND,
            loc            = "upper left",
            # CHANGE 7: x 0.895 → 0.885 to fit 5-column layout
            bbox_to_anchor = (0.885, y_pos),
            bbox_transform = fig.transFigure,
            borderaxespad  = 0,
            handlelength   = 0.8,
            handletextpad  = 0.3,
            borderpad      = 0.3,
            labelspacing   = 0.2,
            frameon        = False,
        )
        leg.set_title(title_text)
        leg.get_title().set_fontsize(SIZE_LEGEND)
        leg.get_title().set_fontweight("bold")
        leg.get_title().set_fontfamily(FONT_FAMILY)

    _save_figure(fig, output_prefix, dpi)
    return fig


def _save_figure(fig, prefix: str, dpi: int = DPI_SUBMIT):
    tiff_path = OUT_DIR / f"{prefix}.tiff"
    pdf_path  = OUT_DIR / f"{prefix}.pdf"
    fig.savefig(tiff_path, dpi=dpi, format="tiff",
                bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_path,  dpi=dpi, format="pdf",
                bbox_inches="tight", pad_inches=0.05)
    print(f"\n  Saved: {tiff_path}  ({tiff_path.stat().st_size // 1024} KB)")
    print(f"  Saved: {pdf_path}")
    plt.close(fig)


# ─── t-SNE parameter guide ────────────────────────────────────────────────────
TSNE_PARAMS_GUIDE = """
t-SNE Parameter Reference — IPIAbDev Figure 3 (5-PLM version)
==============================================================

DEFAULT (matching original publication):
  perplexity=7  learning_rate='auto'  max_iter=1000
  metric='euclidean'  init='pca'  random_state=42  scale=False

PLM embedding dimensions:
  ablang           :   480-dim
  antiberty        :   512-dim
  antiberta2       : 1,024-dim
  antiberta2-cssp  : 1,024-dim
  igbert           : 1,024-dim  ← no parameter change needed vs AntiBERTa2

Required embedding files:
  {DATA_FILE}.ablang.emb.csv
  {DATA_FILE}.antiberty.emb.csv
  {DATA_FILE}.antiberta2.emb.csv
  {DATA_FILE}.antiberta2-cssp.emb.csv
  {DATA_FILE}.igbert.emb.csv          ← NEW

Panel layout (10 panels total):
  Top row    : a   b   c   d   e     ← VH germline
  Bottom row : f   g   h   i   j     ← VL germline
  Columns    : AbLang2 | AntiBERTy | AntiBERTa2 | AntiBERTa2-CSSP | IgBert
"""


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Figure 3 t-SNE — 5 PLMs incl. IgBert (Nature Biotechnology format)"
    )
    parser.add_argument("--data_path",     type=str,   default=MAIN_PATH)
    parser.add_argument("--data_file",     type=str,   default=DATA_FILE)
    parser.add_argument("--perplexity",    type=float, default=7)
    parser.add_argument("--max_iter",      type=int,   default=1000)
    parser.add_argument("--metric",        type=str,   default="euclidean",
                        choices=["cosine", "euclidean"])
    parser.add_argument("--vh_col",        type=str,   default="heavy")
    parser.add_argument("--vl_col",        type=str,   default="light")
    parser.add_argument("--dpi",           type=int,   default=300,
                        choices=[300, 600])
    parser.add_argument("--output_prefix", type=str,   default="Figure3")
    args = parser.parse_args()

    print(TSNE_PARAMS_GUIDE)

    generate_figure3(
        main_path     = args.data_path,
        data_file     = args.data_file,
        vh_col        = args.vh_col,
        vl_col        = args.vl_col,
        tsne_params   = {
            "perplexity"    : args.perplexity,
            "max_iter"      : args.max_iter,
            "metric"        : args.metric,
            "learning_rate" : "auto",
        },
        output_prefix = args.output_prefix,
        dpi           = args.dpi,
    )