# kmer_regression_hit_expansion_ngs.py
# this is part of IPIAbDiscov and IPIAbDev package
# FINAL VERSION - All features + optional CDR3 labels on dendrogram
# Author: Hoan Nguyen
# Created: 2025-03-20
# Version: 3.1

# This script implements a k-mer regression model for hit expansion in NGS data, with optional BLOSUM62 features. It includes functions for feature extraction, model training, diversity-based lead selection, and various evaluation plots including logomaker logos and Levenshtein distance heatmaps. The script is designed to be flexible with different training modes and diversity metrics.
# kmer regession + optional BLOSUM62 features   
#Optional BLOSUM62 in model
#3 training modes
#Levenshtein / BLOSUM62 diversity
#logomaker professional logos
#MACS baseline as germline reference
#Difference logo
#Position-specific stats (IMGT)
#Shannon entropy
#KL divergence + KL contribution heatmap
#Entropy delta plot
#Top-N LV heatmap + dendrogram (with optional real CDR3 labels + mapping CSV)

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from itertools import product
from rapidfuzz.distance import Levenshtein
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import logomaker
    LOGOMAKER_AVAILABLE = True
except ImportError:
    LOGOMAKER_AVAILABLE = False
    print("⚠️  pip install logomaker for fancy logos")

# ====================== BLOSUM62 MATRIX ======================
BLOSUM62_DICT = {
    'A': {'A': 4, 'C': 0, 'D': -2, 'E': -1, 'F': -2, 'G': 0, 'H': -2, 'I': -1, 'K': -1, 'L': -1, 'M': -1, 'N': -2, 'P': -1, 'Q': -1, 'R': -1, 'S': 1, 'T': 0, 'V': 0, 'W': -3, 'Y': -2},
    'C': {'A': 0, 'C': 9, 'D': -3, 'E': -4, 'F': -2, 'G': -3, 'H': -3, 'I': -1, 'K': -3, 'L': -1, 'M': -1, 'N': -3, 'P': -3, 'Q': -3, 'R': -3, 'S': -1, 'T': -1, 'V': -1, 'W': -2, 'Y': -2},
    'D': {'A': -2, 'C': -3, 'D': 6, 'E': 2, 'F': -3, 'G': -1, 'H': -1, 'I': -3, 'K': -1, 'L': -4, 'M': -3, 'N': 1, 'P': -1, 'Q': 0, 'R': -2, 'S': 0, 'T': -1, 'V': -3, 'W': -4, 'Y': -3},
    'E': {'A': -1, 'C': -4, 'D': 2, 'E': 5, 'F': -3, 'G': -2, 'H': 0, 'I': -3, 'K': 1, 'L': -3, 'M': -2, 'N': 0, 'P': -1, 'Q': 2, 'R': 0, 'S': 0, 'T': -1, 'V': -2, 'W': -3, 'Y': -2},
    'F': {'A': -2, 'C': -2, 'D': -3, 'E': -3, 'F': 6, 'G': -3, 'H': -1, 'I': 0, 'K': -3, 'L': 0, 'M': 0, 'N': -3, 'P': -4, 'Q': -3, 'R': -3, 'S': -2, 'T': -2, 'V': -1, 'W': 1, 'Y': 3},
    'G': {'A': 0, 'C': -3, 'D': -1, 'E': -2, 'F': -3, 'G': 6, 'H': -2, 'I': -4, 'K': -2, 'L': -4, 'M': -3, 'N': 0, 'P': -2, 'Q': -2, 'R': -2, 'S': 0, 'T': -2, 'V': -3, 'W': -2, 'Y': -3},
    'H': {'A': -2, 'C': -3, 'D': -1, 'E': 0, 'F': -1, 'G': -2, 'H': 8, 'I': -3, 'K': -1, 'L': -3, 'M': -2, 'N': 1, 'P': -2, 'Q': 0, 'R': 0, 'S': -1, 'T': -2, 'V': -3, 'W': -2, 'Y': 2},
    'I': {'A': -1, 'C': -1, 'D': -3, 'E': -3, 'F': 0, 'G': -4, 'H': -3, 'I': 4, 'K': -3, 'L': 2, 'M': 1, 'N': -3, 'P': -3, 'Q': -3, 'R': -3, 'S': -2, 'T': -1, 'V': 3, 'W': -3, 'Y': -1},
    'K': {'A': -1, 'C': -3, 'D': -1, 'E': 1, 'F': -3, 'G': -2, 'H': -1, 'I': -3, 'K': 5, 'L': -2, 'M': -1, 'N': 0, 'P': -1, 'Q': 1, 'R': 2, 'S': 0, 'T': -1, 'V': -2, 'W': -3, 'Y': -2},
    'L': {'A': -1, 'C': -1, 'D': -4, 'E': -3, 'F': 0, 'G': -4, 'H': -3, 'I': 2, 'K': -2, 'L': 4, 'M': 2, 'N': -3, 'P': -3, 'Q': -2, 'R': -2, 'S': -2, 'T': -1, 'V': 1, 'W': -2, 'Y': -1},
    'M': {'A': -1, 'C': -1, 'D': -3, 'E': -2, 'F': 0, 'G': -3, 'H': -2, 'I': 1, 'K': -1, 'L': 2, 'M': 5, 'N': -2, 'P': -2, 'Q': 0, 'R': -1, 'S': -1, 'T': -1, 'V': 1, 'W': -1, 'Y': -1},
    'N': {'A': -2, 'C': -3, 'D': 1, 'E': 0, 'F': -3, 'G': 0, 'H': 1, 'I': -3, 'K': 0, 'L': -3, 'M': -2, 'N': 6, 'P': -2, 'Q': 0, 'R': 0, 'S': 1, 'T': 0, 'V': -3, 'W': -4, 'Y': -2},
    'P': {'A': -1, 'C': -3, 'D': -1, 'E': -1, 'F': -4, 'G': -2, 'H': -2, 'I': -3, 'K': -1, 'L': -3, 'M': -2, 'N': -2, 'P': 7, 'Q': -1, 'R': -2, 'S': -1, 'T': -1, 'V': -2, 'W': -4, 'Y': -3},
    'Q': {'A': -1, 'C': -3, 'D': 0, 'E': 2, 'F': -3, 'G': -2, 'H': 0, 'I': -3, 'K': 1, 'L': -2, 'M': 0, 'N': 0, 'P': -1, 'Q': 5, 'R': 1, 'S': 0, 'T': -1, 'V': -2, 'W': -2, 'Y': -1},
    'R': {'A': -1, 'C': -3, 'D': -2, 'E': 0, 'F': -3, 'G': -2, 'H': 0, 'I': -3, 'K': 2, 'L': -2, 'M': -1, 'N': 0, 'P': -2, 'Q': 1, 'R': 5, 'S': -1, 'T': -1, 'V': -3, 'W': -3, 'Y': -2},
    'S': {'A': 1, 'C': -1, 'D': 0, 'E': 0, 'F': -2, 'G': 0, 'H': -1, 'I': -2, 'K': 0, 'L': -2, 'M': -1, 'N': 1, 'P': -1, 'Q': 0, 'R': -1, 'S': 4, 'T': 1, 'V': -2, 'W': -3, 'Y': -2},
    'T': {'A': 0, 'C': -1, 'D': -1, 'E': -1, 'F': -2, 'G': -2, 'H': -2, 'I': -1, 'K': -1, 'L': -1, 'M': -1, 'N': 0, 'P': -1, 'Q': -1, 'R': -1, 'S': 1, 'T': 5, 'V': 0, 'W': -2, 'Y': -2},
    'V': {'A': 0, 'C': -1, 'D': -3, 'E': -2, 'F': -1, 'G': -3, 'H': -3, 'I': 3, 'K': -2, 'L': 1, 'M': 1, 'N': -3, 'P': -2, 'Q': -2, 'R': -3, 'S': -2, 'T': 0, 'V': 4, 'W': -3, 'Y': -1},
    'W': {'A': -3, 'C': -2, 'D': -4, 'E': -3, 'F': 1, 'G': -2, 'H': -2, 'I': -3, 'K': -3, 'L': -2, 'M': -1, 'N': -4, 'P': -4, 'Q': -2, 'R': -3, 'S': -3, 'T': -2, 'V': -3, 'W': 11, 'Y': 2},
    'Y': {'A': -2, 'C': -2, 'D': -3, 'E': -2, 'F': 3, 'G': -3, 'H': 2, 'I': -1, 'K': -2, 'L': -1, 'M': -1, 'N': -2, 'P': -3, 'Q': -1, 'R': -2, 'S': -2, 'T': -2, 'V': -1, 'W': 2, 'Y': 7},
}

alphabet = 'ACDEFGHIKLMNPQRSTVWY'

# ====================== 1. FEATURES ======================
def cdr3s_to_features(cdr3_series: pd.Series, use_blosum: bool = False) -> np.ndarray:
    n_kmer = 8420
    extra = 21 if use_blosum else 0
    X = np.zeros((len(cdr3_series), n_kmer + extra), dtype=np.float32)
    kmer_list = [''.join(p) for k in [1,2,3] for p in product(alphabet, repeat=k)]
    kmer_idx = {k: i for i, k in enumerate(kmer_list)}
    aa_idx = {aa: i for i, aa in enumerate(alphabet)}
    for i, cdr3 in enumerate(cdr3_series):
        if pd.isna(cdr3) or len(str(cdr3)) < 4: continue
        seq = 'C' + str(cdr3) + 'W'
        seq = ''.join(aa for aa in seq if aa in alphabet)
        counts = np.zeros(n_kmer, dtype=np.float32)
        for k in [1, 2, 3]:
            for j in range(len(seq) - k + 1):
                km = seq[j:j+k]
                if km in kmer_idx:
                    counts[kmer_idx[km]] += 1
        norm = np.linalg.norm(counts)
        if norm > 0: counts /= norm
        if not use_blosum:
            X[i, :n_kmer] = counts
            continue
        aa_counts = np.zeros(20, dtype=np.float32)
        for aa in seq:
            if aa in aa_idx: aa_counts[aa_idx[aa]] += 1
        aa_norm = np.linalg.norm(aa_counts)
        if aa_norm > 0: aa_counts /= aa_norm
        blosum_sum = 0.0
        if len(seq) >= 2:
            for j in range(len(seq)-1):
                blosum_sum += BLOSUM62_DICT.get(seq[j], {}).get(seq[j+1], -4)
            mean_blosum = blosum_sum / (len(seq)-1)
        else:
            mean_blosum = 0.0
        full = np.concatenate([counts, aa_counts, [mean_blosum]])
        full_norm = np.linalg.norm(full)
        if full_norm > 0: X[i] = full / full_norm
    return X

# ====================== 2. ML TRAINING ======================
def add_kmer_logreg_score(df, cdr3_col="HCDR3", macs_col="Macs_count", facs1_col="FACS1_count",
                          use_blosum_features=False, min_pos_count=5, min_fold_change=1.5,
                          training_mode="binary_strong", score_col=None):
    if score_col is None:
        prefix = "kmer_blosum" if use_blosum_features else "kmer"
        score_col = f"{prefix}_logreg_score" if "binary" in training_mode else f"{prefix}_oneclass_score"
    df = df.copy()
    for col in [macs_col, facs1_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    total_macs = df[macs_col].sum()
    total_facs1 = df[facs1_col].sum()
    df['freq_macs'] = df[macs_col] / total_macs if total_macs > 0 else 0
    df['freq_facs1'] = df[facs1_col] / total_facs1 if total_facs1 > 0 else 0
    df['fold_change'] = df['freq_facs1'] / (df['freq_macs'] + 1e-8)
    pos_mask = (df[facs1_col] > min_pos_count) & (df['freq_facs1'] >= df['freq_macs'] * min_fold_change)
    X = cdr3s_to_features(df[cdr3_col], use_blosum=use_blosum_features)
    train_mask = (df[macs_col] > 0) | (df[facs1_col] > 0)
    if training_mode == "one_class":
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(X[train_mask & pos_mask])
        df[score_col] = -model.decision_function(X)
    else:
        if training_mode == "binary_strong":
            neg_mask = (df[facs1_col] > min_pos_count) & (df['freq_facs1'] <= df['freq_macs'] * 0.5)
            y = pd.Series(0, index=df.index)
            y[pos_mask] = 1
            y[neg_mask] = 0
        else:
            y = pos_mask.astype(int)
        model = LogisticRegression(penalty='l1', C=1.0, solver='liblinear',
                                   class_weight='balanced', max_iter=1000, random_state=42)
        model.fit(X[train_mask], y[train_mask])
        df[score_col] = model.predict_proba(X)[:, 1]
    return df, model, X

# ====================== 3. LOAD PREVIOUS & DIVERSITY ======================
def load_previous_cdr3s(file_list, cdr3_column="HCDR3"):
    all_cdr3 = set()
    for f in file_list:
        if not f: continue
        try:
            temp = pd.read_excel(f) if f.lower().endswith(('.xlsx','.xls')) else pd.read_csv(f)
            if cdr3_column in temp.columns:
                new = temp[cdr3_column].dropna().astype(str).str.strip().tolist()
                all_cdr3.update(new)
        except: pass
    return list(all_cdr3)

def blosum_similarity(seq1, seq2):
    min_len = min(len(seq1), len(seq2))
    score = sum(BLOSUM62_DICT.get(seq1[i], {}).get(seq2[i], -4) for i in range(min_len))
    return score / min_len if min_len else -10.0

def select_diverse_leads(df, previous_cdr3s, score_col, cdr3_col="HCDR3", count_col="FACS1_count",
                         min_score=0.8, min_cpm=200, diversity_metric="levenshtein",
                         min_levenshtein_dist=5, max_blosum_similarity=1.5,
                         selected_col="selected_for_synthesis"):
    df = df.copy()
    total = df[count_col].sum()
    df['cpm'] = df[count_col] / total * 1_000_000
    candidates = df[(df[score_col] >= min_score) & (df['cpm'] >= min_cpm)].sort_values(score_col, ascending=False).copy()
    selected_indices, selected_cdr3s = [], []
    for idx, row in candidates.iterrows():
        cdr3 = str(row[cdr3_col]).strip()
        if pd.isna(cdr3) or len(cdr3) < 4: continue
        if diversity_metric == "levenshtein":
            d_prev = min((Levenshtein.distance(cdr3, p) for p in previous_cdr3s), default=999)
            d_pair = min((Levenshtein.distance(cdr3, s) for s in selected_cdr3s), default=999)
            if d_prev < min_levenshtein_dist or d_pair < min_levenshtein_dist: continue
        else:
            s_prev = max((blosum_similarity(cdr3, p) for p in previous_cdr3s), default=-999)
            s_pair = max((blosum_similarity(cdr3, s) for s in selected_cdr3s), default=-999)
            if s_prev > max_blosum_similarity or s_pair > max_blosum_similarity: continue
        selected_indices.append(idx)
        selected_cdr3s.append(cdr3)
    df[selected_col] = False
    df.loc[selected_indices, selected_col] = True
    return df

# ====================== 4. LOGOMAKER ======================
def plot_fancy_logo(cdr3_list, title, filename, output_folder):
    if not cdr3_list or not LOGOMAKER_AVAILABLE: return
    max_len = max((len(s) for s in cdr3_list), default=0)
    counts = pd.DataFrame(0, index=list(alphabet), columns=range(max_len))
    for seq in cdr3_list:
        for i, aa in enumerate(seq):
            if aa in alphabet and i < max_len:
                counts.loc[aa, i] += 1
    prob = counts / counts.sum(axis=0)
    fig, ax = plt.subplots(figsize=(14, 5))
    logo = logomaker.Logo(prob.T, ax=ax, color_scheme='chemistry', vpad=0.1, width=0.9)
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    ax.set_title(title)
    ax.set_xlabel("CDR3 Position")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename), dpi=300)
    plt.close()

# ====================== 5. POSITION STATS + KL HEATMAP ======================
def generate_position_specific_stats(df, score_col, output_folder, cdr3_col="HCDR3", macs_col="Macs_count", facs1_col="FACS1_count"):
    os.makedirs(output_folder, exist_ok=True)
    print("📊 Computing Position-Specific Stats + KL Heatmap...")
    groups = {
        'MACS_Baseline': df[df[macs_col] > 0][cdr3_col].dropna().astype(str).str.strip().tolist(),
        'Training_Positive': df[(df[facs1_col] > 5) & (df['freq_facs1'] >= df['freq_macs'] * 1.5)][cdr3_col].dropna().astype(str).str.strip().tolist(),
        'High_Score': df[df[score_col] > 0.8][cdr3_col].dropna().astype(str).str.strip().tolist()
    }
    max_len = max((max(len(s) for s in seqs) if seqs else 0) for seqs in groups.values())
    stats_rows = []
    entropy_data = {name: [] for name in groups}
    kl_data, delta_data = [], []
    kl_contrib_matrix = np.zeros((len(alphabet), max_len))

    for pos in range(max_len):
        row = {'IMGT_Position': pos + 105}
        pos_probs = {}
        for group_name, seqs in groups.items():
            aa_count = {aa: 0 for aa in alphabet}
            total = 0
            for seq in seqs:
                if pos < len(seq) and seq[pos] in alphabet:
                    aa_count[seq[pos]] += 1
                    total += 1
            probs = np.array(list(aa_count.values())) / total if total > 0 else np.zeros(20)
            pos_probs[group_name] = probs
            probs_nz = probs[probs > 0]
            entropy = -np.sum(probs_nz * np.log2(probs_nz)) if len(probs_nz) > 0 else 0
            entropy_data[group_name].append(entropy)
            row[f"{group_name}_Entropy"] = round(entropy, 3)
            if total > 0:
                top_idx = np.argmax(list(aa_count.values()))
                row[f"{group_name}_Top_AA"] = f"{alphabet[top_idx]} ({max(aa_count.values())/total*100:.1f}%)"
        if 'High_Score' in pos_probs and 'MACS_Baseline' in pos_probs:
            p = pos_probs['High_Score']
            q = pos_probs['MACS_Baseline']
            q[q == 0] = 1e-10
            p[p == 0] = 1e-10
            kl = np.sum(p * np.log2(p / q))
            kl_data.append(kl)
            row['KL_Divergence_High_vs_MACS'] = round(kl, 3)
            for i, aa in enumerate(alphabet):
                contrib = p[i] * np.log2(p[i] / q[i]) if p[i] > 0 else 0
                kl_contrib_matrix[i, pos] = contrib
        delta = entropy_data['MACS_Baseline'][-1] - entropy_data['High_Score'][-1]
        delta_data.append(delta)
        row['Entropy_Delta_MACS_minus_High'] = round(delta, 3)
        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(os.path.join(output_folder, "11_position_specific_stats.csv"), index=False)

    # Entropy Delta Plot
    positions = range(105, 105 + len(delta_data))
    plt.figure(figsize=(14, 7))
    plt.bar(positions, delta_data, color=['green' if d > 0 else 'red' for d in delta_data], alpha=0.8)
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Entropy Delta: MACS → High-Score Rescued Clones")
    plt.xlabel("IMGT Position")
    plt.ylabel("Entropy Delta (bits)")
    plt.savefig(os.path.join(output_folder, "14_entropy_delta_plot.png"), dpi=300)
    plt.close()

    # KL Line Plot
    plt.figure(figsize=(14, 7))
    plt.plot(positions, kl_data, color='purple', linewidth=3, marker='o')
    plt.title("Relative Entropy (KL Divergence) High-Score vs MACS")
    plt.xlabel("IMGT Position")
    plt.ylabel("KL Divergence (bits)")
    plt.savefig(os.path.join(output_folder, "15_kl_divergence_vs_macs.png"), dpi=300)
    plt.close()

    # KL Contribution Heatmap
    plt.figure(figsize=(16, 10))
    sns.heatmap(kl_contrib_matrix, annot=False, cmap="RdBu_r", center=0,
                xticklabels=[f"{p}" for p in positions],
                yticklabels=list(alphabet),
                cbar_kws={'label': 'Contribution to KL divergence'})
    plt.title("KL Divergence Heatmap\n(Blue = enriched in rescued clones | Red = depleted)")
    plt.xlabel("IMGT Position (CDR3)")
    plt.ylabel("Amino Acid")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "16_kl_divergence_contribution_heatmap.png"), dpi=300)
    plt.close()

# ====================== 6. TOP-N LV CLUSTER ======================
def plot_top_n_cluster(df, score_col, output_folder, cdr3_col="HCDR3", top_n=50, label_with_cdr3=True):
    if top_n <= 1: return
    os.makedirs(output_folder, exist_ok=True)
    top_df = df.nlargest(top_n, score_col).copy()
    cdr3_list = top_df[cdr3_col].dropna().astype(str).str.strip().tolist()
    scores = top_df[score_col].round(3).tolist()
    n = len(cdr3_list)

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = Levenshtein.distance(cdr3_list[i], cdr3_list[j])
            dist_matrix[i,j] = dist_matrix[j,i] = d

    # Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(dist_matrix, cmap="viridis", xticklabels=False, yticklabels=False)
    plt.title(f"Levenshtein Distance Heatmap — Top {n} High-Score CDR3s")
    plt.savefig(os.path.join(output_folder, "17_lv_distance_heatmap_top50.png"), dpi=300)
    plt.close()

    # Dendrogram
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method='average')
    labels = [f"#{i+1:02d}: {s[:8]}...{s[-4:]} ({scores[i]})" if label_with_cdr3 else f"#{i+1:02d}" for i, s in enumerate(cdr3_list)]

    plt.figure(figsize=(18, 10))
    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=9)
    plt.title(f"Dendrogram — Top {n} High-Score CDR3s")
    plt.xlabel("Clones")
    plt.ylabel("Levenshtein Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "18_lv_dendrogram_top50.png"), dpi=300)
    plt.close()

    # Mapping CSV
    mapping = pd.DataFrame({"Rank": range(1,n+1), "CDR3": cdr3_list, "ML_Score": scores})
    mapping.to_csv(os.path.join(output_folder, "top50_cluster_mapping.csv"), index=False)

# ====================== 7. MAIN EVALUATION PLOTS ======================
def generate_evaluation_plots(df, score_col, training_mode, output_folder, model=None, X=None,
                              cdr3_col="HCDR3", macs_col="Macs_count", facs1_col="FACS1_count"):
    os.makedirs(output_folder, exist_ok=True)
    # Log fold-change
    df['log_fold_change'] = np.log10(df['fold_change'] + 1e-8)
    plt.figure(figsize=(10,6))
    sns.histplot(df[df['fold_change']>0]['log_fold_change'], bins=50, kde=True)
    plt.axvline(np.log10(1.5), color='red', linestyle='--')
    plt.title("Log10 Fold-Change Distribution")
    plt.savefig(os.path.join(output_folder, "01_log_fold_change.png"))
    plt.close()

    # Fancy logos
    all_cdr3s = df[cdr3_col].dropna().astype(str).tolist()
    plot_fancy_logo(all_cdr3s, "ALL RAW CDR3s", "07_logo_all_raw.png", output_folder)
    pos_cdr3s = df[(df[facs1_col]>5)&(df['freq_facs1']>=df['freq_macs']*1.5)][cdr3_col].dropna().astype(str).tolist()
    plot_fancy_logo(pos_cdr3s, "Training Positives", "04_logo_train_positive.png", output_folder)
    high_cdr3s = df[df[score_col]>0.8][cdr3_col].dropna().astype(str).tolist()
    plot_fancy_logo(high_cdr3s, "High-Score Predicted", "05_logo_high_score.png", output_folder)

    # Position stats + KL heatmap
    generate_position_specific_stats(df, score_col, output_folder, cdr3_col, macs_col, facs1_col)

    # Top-N LV cluster
    if PLOT_TOP_N_CLUSTER > 0:
        plot_top_n_cluster(df, score_col, output_folder, cdr3_col, PLOT_TOP_N_CLUSTER, LABEL_DENDROGRAM_WITH_CDR3)

# ====================== USER CONFIG ======================
if __name__ == "__main__":

    INPUT_FILE = "/Users/Hoan.Nguyen/ComBio/NGS/Projects/AntibodyDiscovery/Miseq108/processed/mPTPRO_440-827_AH_Block198_clones.csv"

    USE_BLOSUM_IN_MODEL = False
    TRAINING_MODE = "binary_strong"
    USE_MACS_AS_GERMLINE = True

    DIVERSITY_METRIC = "levenshtein"
    MIN_LEVENSHTEIN_DIST = 5

    PLOT_TOP_N_CLUSTER = 50
    LABEL_DENDROGRAM_WITH_CDR3 = True

    # ================== CUSTOM OUTPUT FOLDER ==================
    OUTPUT_FOLDER = "/Users/Hoan.Nguyen/ComBio/NGS/Projects/AntibodyDiscovery/Miseq108"          
    # ========================================================

    PREVIOUS_FILES = ["/Users/Hoan.Nguyen/ComBio/AntigenDB/datasources/ipi_data/processed/ipi_antibodydb_july2025.csv"]
    PREVIOUS_CDR3_COLUMN = "CDR3"
    facs_col = 'count mPTPRO_440-827_AH__Block198__Round5__F_P__4nM_Block198.csv'
    macs_col = 'count mPTPRO_440-827_AH__Block198__Round3__M_P__100nM_Block198.csv'

    # ====================== RUN ======================
    #df = pd.read_csv(INPUT_FILE)
    #df, model, X = add_kmer_logreg_score(df, use_blosum_features=USE_BLOSUM_IN_MODEL, training_mode=TRAINING_MODE)
    #previous_list = load_previous_cdr3s(PREVIOUS_FILES, PREVIOUS_CDR3_COLUMN)
    #df = select_diverse_leads(df, previous_cdr3s=previous_list, score_col=df.columns[-1], diversity_metric=DIVERSITY_METRIC)
    #generate_evaluation_plots(df, df.columns[-2], TRAINING_MODE, OUTPUT_FOLDER, model=model, X=X)

    df = pd.read_csv(INPUT_FILE)
    df, model, X = add_kmer_logreg_score(df, use_blosum_features=USE_BLOSUM_IN_MODEL, training_mode=TRAINING_MODE,cdr3_col='cdr3_aa',macs_col='count mPTPRO_440-827_AH__Block198__Round3__M_P__100nM_Block198.csv',facs1_col=facs_col)
    previous_list = load_previous_cdr3s(PREVIOUS_FILES, PREVIOUS_CDR3_COLUMN)
    df = select_diverse_leads(df, previous_cdr3s=previous_list, score_col=df.columns[-1], diversity_metric=DIVERSITY_METRIC,cdr3_col='cdr3_aa',count_col=facs_col, min_levenshtein_dist=MIN_LEVENSHTEIN_DIST)
    generate_evaluation_plots(df, df.columns[-2], TRAINING_MODE,OUTPUT_FOLDER, model=model, X=X,cdr3_col='cdr3_aa', macs_col=macs_col, facs1_col=facs_col)



    # Final CSVs
    df.to_csv(os.path.join(OUTPUT_FOLDER, "leads_with_ml_score_and_selection.csv"), index=False)
    df[df["selected_for_synthesis"]].to_csv(os.path.join(OUTPUT_FOLDER, "final_clones_for_synthesis.csv"), index=False)

    print(f"🎉 v6.9 COMPLETE! All results saved in: {OUTPUT_FOLDER}")