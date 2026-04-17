"""
IPI Antibody Developability Prediction Platform
Final Production Version — DEC-2025 / Updated APR-2026
Supports 7 PLMs:
  ablang          → 480 columns    (ablang2, paired VH+VL)
  antiberty       → 512 columns    (AntiBERTy, mean pool)
  antiberta2      → 1024 columns   (alchemab/antiberta2)
  antiberta2-cssp → 1024 columns   (alchemab/antiberta2-cssp)
  igbert          → 1024 columns   (Exscientia/IgBert, ProtBert backbone, paired)
  igt5            → 1024 columns   (Exscientia/IgT5, ProtT5-XL backbone, paired)
  abmap           → auto-dim       (rs239/abmap, ESM-2+CDR mutagenesis, ANARCI required)

Installation for new PLMs:
  pip install transformers       # igbert, igt5 (already needed for antiberta2)
  pip install abmap              # abmap (also needs ANARCI + hmmer)
  conda install -c bioconda hmmer=3.3.2 -y
  git clone https://github.com/oxpig/ANARCI && cd ANARCI && python setup.py install
"""

# embedding_generator.py

# [FIX-MACOS] Must be set BEFORE transformers/sentencepiece is imported.
# SentencePiece (used by T5Tokenizer) spawns background pthreads that
# conflict with macOS fork() → "mutex lock failed: Invalid argument" / abort.
# Setting here (module load time) ensures it takes effect before any import.
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")  # macOS fork safety

import math
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ── Soft imports: only fail at call time if the LM is actually requested ──────

# ABLANG2
try:
    import ablang2
    ABLANG_OK = True
except ImportError:
    ABLANG_OK = False

# ANTIBERTY
try:
    from antiberty import AntiBERTyRunner
    ANTIBERTY_OK = True
except ImportError:
    ANTIBERTY_OK = False

# ANTIBERTA2 / ANTIBERTA2-CSSP  (RoFormer)
try:
    from transformers import RoFormerTokenizer, RoFormerForMaskedLM
    TRANSFORMERS_OK = True
except ImportError:
    TRANSFORMERS_OK = False

# IGBERT  (ProtBert backbone, Exscientia/IgBert)
try:
    from transformers import BertModel, BertTokenizer
    IGBERT_OK = True
except ImportError:
    IGBERT_OK = False

# IGT5  (ProtT5-XL backbone, Exscientia/IgT5)
try:
    from transformers import T5EncoderModel, T5Tokenizer
    IGT5_OK = True
except ImportError:
    IGT5_OK = False

# ABMAP  (MIT/Sanofi, ESM-2 + CDR mutagenesis, pip install abmap)
try:
    import abmap
    from abmap.embed import embed_seq as _abmap_embed_seq
    ABMAP_OK = True
except ImportError:
    ABMAP_OK = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _insert_space_every_other_except_cls(input_string: str) -> str:
    """Space tokeniser for AntiBERTy / AntiBERTa2."""
    parts = input_string.split('[CLS]')
    modified = [''.join(c + ' ' for c in part).strip() for part in parts]
    return ' [CLS] '.join(modified)


def batch_loader(data, batch_size):
    n = len(data)
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        yield i, end, data[i:end]


def _space_seq(seq: str) -> str:
    """Space-separate a single-letter AA sequence (ProtBert/ProtT5 format)."""
    return ' '.join(seq.strip().upper())


def _clean_seq(seq) -> str:
    """Coerce to str, replace nan-like → empty."""
    s = str(seq).strip()
    if s.lower() in ('nan', 'none', 'null', '', 'n/a'):
        return ''
    return s.upper()


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool last hidden state over non-padding tokens.
    hidden : (B, L, D)
    mask   : (B, L)  — 1 = real token, 0 = padding
    Returns (B, D)
    """
    m = mask.unsqueeze(-1).float()          # (B, L, 1)
    return (hidden * m).sum(1) / m.sum(1)   # (B, D)


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_embedding(input_file: str,
                       lm: str = "antiberta2",
                       batch_size: int = 64,
                       device: str = "cpu",
                       mode: str = "VHVL") -> str:
    """
    Generate embeddings for antibody sequences in input_file.

    Parameters
    ----------
    input_file : str   path to .csv / .xlsx / .xls
    lm         : str   one of: ablang | antiberty | antiberta2 | antiberta2-cssp |
                               igbert | igt5 | abmap
    batch_size : int   sequences per GPU batch (reduce if OOM)
    device     : str   'cpu' | 'cuda' | 'mps'
    mode       : str   'VHVL' (default, paired) | 'VH' (heavy-chain only)

    Returns
    -------
    str : path to output CSV  ({input_file}.{lm}.emb.csv)
    """
    suffix   = f".{lm}.vh.emb.csv" if mode.upper() == "VH" else f".{lm}.emb.csv"
    out_path = f"{input_file}{suffix}"

    if os.path.exists(out_path):
        print(f"Embedding exists: {out_path}")
        return out_path

    print(f"\nGenerating {lm.upper()} embedding ({mode.upper()}) "
          f"→ {os.path.basename(out_path)}")

    # ── Load file ─────────────────────────────────────────────────────────────
    try:
        if input_file.lower().endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(input_file)
        else:
            raise ValueError(f"Unsupported format: {input_file}")
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    required = ['BARCODE', 'HSEQ'] + (['LSEQ'] if mode.upper() == 'VHVL' else [])
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
        return None

    df = df.set_index('BARCODE')

    # ──────────────────────────────────────────────────────────────────────────
    # ABLANG2 — 480 columns
    # ──────────────────────────────────────────────────────────────────────────
    if lm == "ablang":
        if not ABLANG_OK:
            print("Error: ablang2 not installed. Run: pip install ablang2")
            return None
        ablang_model = ablang2.pretrained(
            "ablang2-paired", device=device, random_init=False, ncpu=3
        )
        embeddings = []
        for idx in tqdm(df.index, desc="ABLang2"):
            try:
                seqs = ([str(df.loc[idx, 'HSEQ']), str(df.loc[idx, 'LSEQ'])]
                        if mode.upper() == 'VHVL' else [str(df.loc[idx, 'HSEQ'])])
                emb = ablang_model(seqs, mode='seqcoding')[0]
                embeddings.append(emb)
            except Exception as e:
                print(f"  Warning ABLang2 {idx}: {e} → zeros")
                embeddings.append(np.zeros(480))

        emb_df = pd.DataFrame(embeddings, index=df.index)
        emb_df.to_csv(out_path)
        print(f"SAVED ABLang2 ({mode}): {out_path} | Shape: {emb_df.shape}")
        return out_path

    # ──────────────────────────────────────────────────────────────────────────
    # ANTIBERTY — 512 columns
    # ──────────────────────────────────────────────────────────────────────────
    elif lm == "antiberty":
        if not ANTIBERTY_OK:
            print("Error: antiberty not installed. Run: pip install antiberty")
            return None
        runner = AntiBERTyRunner()
        df_tmp = df.copy()
        if mode.upper() == 'VHVL':
            df_tmp['HL'] = (df_tmp['HSEQ'].astype(str) + '[CLS][CLS]'
                            + df_tmp['LSEQ'].astype(str))
        else:
            df_tmp['HL'] = df_tmp['HSEQ'].astype(str) + '[CLS]'
        sequences = [s.replace('  ', ' ') for s in df_tmp['HL'].tolist()]

        n, dim = len(sequences), 512
        embeddings = torch.empty((n, dim))
        for i, (start, end, batch) in enumerate(batch_loader(sequences, 500), 1):
            print(f'  Batch {i}/{math.ceil(n / 500)}')
            try:
                embs   = runner.embed(batch)
                means  = [e.mean(axis=0) for e in embs]
                embeddings[start:end] = torch.stack(means)
            except Exception as e:
                print(f"  Batch {i} failed: {e} → zeros")
                embeddings[start:end] = torch.zeros(end - start, dim)

        emb_df = pd.DataFrame(embeddings.numpy(), index=df.index)
        emb_df.to_csv(out_path)
        print(f"SAVED AntiBERTy ({mode}): {out_path} | Shape: {emb_df.shape}")
        return out_path

    # ──────────────────────────────────────────────────────────────────────────
    # ANTIBERTA2 / ANTIBERTA2-CSSP — 1024 columns
    # ──────────────────────────────────────────────────────────────────────────
    elif lm in ("antiberta2", "antiberta2-cssp"):
        if not TRANSFORMERS_OK:
            print("Error: transformers not installed. Run: pip install transformers")
            return None
        model_name = ("alchemab/antiberta2-cssp" if lm == "antiberta2-cssp"
                      else "alchemab/antiberta2")
        tokenizer  = RoFormerTokenizer.from_pretrained(model_name)
        model      = RoFormerForMaskedLM.from_pretrained(model_name).to(device)
        model.eval()

        df_tmp = df.copy()
        if mode.upper() == 'VHVL':
            df_tmp['HL'] = (df_tmp['HSEQ'].astype(str) + '[CLS][CLS]'
                            + df_tmp['LSEQ'].astype(str))
        else:
            df_tmp['HL'] = df_tmp['HSEQ'].astype(str) + '[CLS]'
        sequences = [_insert_space_every_other_except_cls(s).replace('  ', ' ')
                     for s in df_tmp['HL'].tolist()]

        n, dim = len(sequences), 1024
        embeddings = torch.empty((n, dim))
        for i, (start, end, batch) in enumerate(batch_loader(sequences, 128), 1):
            print(f'  Batch {i}/{math.ceil(n / 128)}')
            try:
                enc = tokenizer(batch, padding=True, truncation=True,
                                max_length=512, return_tensors='pt').to(device)
                with torch.no_grad():
                    out    = model(**enc, output_hidden_states=True)
                    hidden = out.hidden_states[-1]
                    pooled = _mean_pool(hidden, enc['attention_mask'])
                    embeddings[start:end] = pooled.cpu()
            except Exception as e:
                print(f"  Batch {i} failed: {e} → zeros")
                embeddings[start:end] = torch.zeros(end - start, dim)

        emb_df = pd.DataFrame(embeddings.numpy(), index=df.index)
        emb_df.to_csv(out_path)
        print(f"SAVED {lm} ({mode}): {out_path} | Shape: {emb_df.shape}")
        return out_path

    # ──────────────────────────────────────────────────────────────────────────
    # IGBERT — 1024 columns
    # Exscientia/IgBert  (ProtBert backbone, fine-tuned on 2M+ paired OAS seqs)
    # Paper : Kenlay et al., PLOS Comput Biol 2024 (doi:10.1371/journal.pcbi.1012646)
    # Input : space-separated AAs; paired VH+VL fed as a BERT sequence pair
    #         → [CLS] VH_tokens [SEP] VL_tokens [SEP]
    # Pool  : mean over all non-padding tokens (incl. [CLS]/[SEP] per paper)
    # ──────────────────────────────────────────────────────────────────────────
    elif lm == "igbert":
        if not IGBERT_OK:
            print("Error: transformers not installed. pip install transformers")
            return None

        print("  Loading Exscientia/IgBert ...")
        tokenizer = BertTokenizer.from_pretrained(
            "Exscientia/IgBert", do_lower_case=False
        )
        model = BertModel.from_pretrained(
            "Exscientia/IgBert", add_pooling_layer=False
        ).to(device)
        model.eval()

        hseqs = [_clean_seq(df.loc[idx, 'HSEQ']) for idx in df.index]
        lseqs = ([_clean_seq(df.loc[idx, 'LSEQ']) for idx in df.index]
                 if mode.upper() == 'VHVL' else [''] * len(df))

        n, dim = len(hseqs), 1024
        embeddings = torch.empty((n, dim))

        for i, (start, end, _) in enumerate(batch_loader(hseqs, batch_size), 1):
            print(f'  Batch {i}/{math.ceil(n / batch_size)}')
            try:
                vh_batch = [_space_seq(s) for s in hseqs[start:end]]
                vl_batch = [_space_seq(s) for s in lseqs[start:end]]

                if mode.upper() == 'VHVL':
                    # BERT pair encoding: [CLS] VH [SEP] VL [SEP]
                    enc = tokenizer(
                        vh_batch, vl_batch,
                        padding=True, truncation=True,
                        max_length=512, return_tensors='pt'
                    ).to(device)
                else:
                    enc = tokenizer(
                        vh_batch,
                        padding=True, truncation=True,
                        max_length=512, return_tensors='pt'
                    ).to(device)

                with torch.no_grad():
                    out    = model(**enc)
                    hidden = out.last_hidden_state          # (B, L, 1024)
                    pooled = _mean_pool(hidden, enc['attention_mask'])
                    embeddings[start:end] = pooled.cpu()

            except Exception as e:
                print(f"  Batch {i} failed: {e} → zeros")
                embeddings[start:end] = torch.zeros(end - start, dim)

        emb_df = pd.DataFrame(embeddings.numpy(), index=df.index)
        emb_df.to_csv(out_path)
        print(f"SAVED IgBert ({mode}): {out_path} | Shape: {emb_df.shape}")
        return out_path

    # ──────────────────────────────────────────────────────────────────────────
    # IGT5 — 1024 columns
    # Exscientia/IgT5  (ProtT5-XL backbone, fine-tuned on 2M+ paired OAS seqs)
    # Paper : same as IgBert above
    # Input : space-separated AAs; VH and VL concatenated (T5 encoder-only)
    #         T5 does not use [CLS]/[SEP]; VH+VL separated by a single space
    # Pool  : mean over all non-padding tokens
    # ──────────────────────────────────────────────────────────────────────────
    elif lm == "igt5":
        if not IGT5_OK:
            print("Error: transformers not installed. pip install transformers")
            return None

        print("  Loading Exscientia/IgT5 ...")
        tokenizer = T5Tokenizer.from_pretrained(
            "Exscientia/IgT5", do_lower_case=False, legacy=False
        )
        model = T5EncoderModel.from_pretrained("Exscientia/IgT5").to(device)
        model.eval()

        hseqs = [_clean_seq(df.loc[idx, 'HSEQ']) for idx in df.index]
        lseqs = ([_clean_seq(df.loc[idx, 'LSEQ']) for idx in df.index]
                 if mode.upper() == 'VHVL' else [''] * len(df))

        n, dim = len(hseqs), 1024
        embeddings = torch.empty((n, dim))

        for i, (start, end, _) in enumerate(batch_loader(hseqs, batch_size), 1):
            print(f'  Batch {i}/{math.ceil(n / batch_size)}')
            try:
                batch_seqs = []
                for vh, vl in zip(hseqs[start:end], lseqs[start:end]):
                    if mode.upper() == 'VHVL' and vl:
                        # Concatenate VH + VL with space separator
                        batch_seqs.append(_space_seq(vh) + ' ' + _space_seq(vl))
                    else:
                        batch_seqs.append(_space_seq(vh))

                enc = tokenizer(
                    batch_seqs,
                    padding=True, truncation=True,
                    max_length=512, return_tensors='pt'
                ).to(device)

                with torch.no_grad():
                    out    = model(**enc)
                    hidden = out.last_hidden_state           # (B, L, 1024)
                    pooled = _mean_pool(hidden, enc['attention_mask'])
                    embeddings[start:end] = pooled.cpu()

            except Exception as e:
                print(f"  Batch {i} failed: {e} → zeros")
                embeddings[start:end] = torch.zeros(end - start, dim)

        emb_df = pd.DataFrame(embeddings.numpy(), index=df.index)
        emb_df.to_csv(out_path)
        print(f"SAVED IgT5 ({mode}): {out_path} | Shape: {emb_df.shape}")
        return out_path

    # ──────────────────────────────────────────────────────────────────────────
    # ABMAP — auto-detected dim (typically 256 or 512)
    # MIT/Sanofi, PNAS 2024  (doi:10.1073/pnas.2418918121)
    # Install : pip install abmap   (+ ANARCI + hmmer — see module docstring)
    # Method  : per-chain embed_seq (ESM-2 + CDR mutagenesis augmentation)
    #           VH and VL embedded separately then mean-averaged
    # ANARCI  : required to assign IMGT CDR labels (must be installed separately)
    # ──────────────────────────────────────────────────────────────────────────
    elif lm == "abmap":
        if not ABMAP_OK:
            print("Error: abmap not installed. pip install abmap")
            print("Also requires ANARCI + hmmer — see module docstring.")
            return None

        print("  Loading AbMAP (ESM-2 backbone, CDR mutagenesis augmentation) ...")
        hseqs = [_clean_seq(df.loc[idx, 'HSEQ']) for idx in df.index]
        lseqs = ([_clean_seq(df.loc[idx, 'LSEQ']) for idx in df.index]
                 if mode.upper() == 'VHVL' else [''] * len(df))

        embeddings = []
        dim_detected = None

        for j, (vh, vl) in enumerate(tqdm(zip(hseqs, lseqs),
                                          total=len(hseqs), desc="AbMAP")):
            try:
                # embed_seq(sequence, seq_type, plm_type)
                # seq_type: 'H' = heavy chain, 'L' = light chain
                vh_emb = _abmap_embed_seq(vh, seq_type='H', plm_type='esm2')

                if mode.upper() == 'VHVL' and vl:
                    vl_emb = _abmap_embed_seq(vl, seq_type='L', plm_type='esm2')
                    # Average VH + VL embeddings → single fixed-length vector
                    emb = (np.array(vh_emb) + np.array(vl_emb)) / 2.0
                else:
                    emb = np.array(vh_emb)

                if dim_detected is None:
                    dim_detected = len(emb)
                    print(f"  AbMAP output dim detected: {dim_detected}")

                embeddings.append(emb)

            except Exception as e:
                if dim_detected is None:
                    # Can't determine dim yet — record as None, fill after
                    print(f"  Warning AbMAP idx={j}: {e} → will fill zeros")
                    embeddings.append(None)
                else:
                    print(f"  Warning AbMAP idx={j}: {e} → zeros")
                    embeddings.append(np.zeros(dim_detected))

        # Fill None placeholders with zeros (only if first batch also failed)
        if dim_detected is None:
            print("  ERROR: AbMAP failed on all sequences — no output written.")
            return None

        embeddings = [e if e is not None else np.zeros(dim_detected)
                      for e in embeddings]

        emb_df = pd.DataFrame(np.vstack(embeddings), index=df.index)
        emb_df.to_csv(out_path)
        print(f"SAVED AbMAP ({mode}): {out_path} | "
              f"Shape: {emb_df.shape}  (dim={dim_detected})")
        return out_path

    else:
        print(f"Error: LM '{lm}' not supported or package missing.")
        print(f"Supported: ablang | antiberty | antiberta2 | antiberta2-cssp | "
              f"igbert | igt5 | abmap")
        return None


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="IPI embedding generator — 7 PLMs supported",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PLM summary
───────────────────────────────────────────────────────────
  ablang          480-dim   ablang2-paired (Oxford OPIG)
  antiberty       512-dim   AntiBERTy (Johns Hopkins)
  antiberta2      1024-dim  alchemab/antiberta2
  antiberta2-cssp 1024-dim  alchemab/antiberta2-cssp
  igbert          1024-dim  Exscientia/IgBert (PLOS CB 2024)
  igt5            1024-dim  Exscientia/IgT5   (PLOS CB 2024)
  abmap           auto-dim  rs239/abmap (PNAS 2025, needs ANARCI)

Installation (new PLMs only):
  pip install transformers     # already needed for antiberta2
  pip install abmap            # for abmap
  conda install -c bioconda hmmer=3.3.2 -y
  git clone https://github.com/oxpig/ANARCI && cd ANARCI && python setup.py install
        """,
    )
    parser.add_argument("--input", required=True, help="Input CSV/Excel file")
    parser.add_argument("--lm", default="antiberta2",
                        choices=["ablang", "antiberty", "antiberta2", "antiberta2-cssp",
                                 "igbert", "igt5", "abmap"])
    parser.add_argument("--mode", default="VHVL", choices=["VHVL", "VH"],
                        help="VHVL=paired VH+VL (default), VH=heavy chain only")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    generate_embedding(
        input_file=args.input,
        lm=args.lm,
        batch_size=args.batch_size,
        device=args.device,
        mode=args.mode,
    )