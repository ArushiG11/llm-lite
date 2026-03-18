#!/usr/bin/env python3
"""
Unified evaluation: Bigram vs Trigram vs Transformer.

Computes validation perplexity for each model (where available) and plots a bar chart.
Run from repo root: python scripts/evaluate_models.py [--valid-subset N] [--no-plot]
"""

import argparse
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

# Paths (from repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from model_causal import GPTMini

VALID_BIN = REPO_ROOT / "data/processed/valid.bin"
TOK_PATH = REPO_ROOT / "tokenizer/bpe_tokenizer.json"
BIGRAM_DENSE_COUNTS = REPO_ROOT / "models/bigram_counts.npy"
BIGRAM_DENSE_UNIGRAM = REPO_ROOT / "models/bigram_unigram.npy"
BIGRAM_SPARSE_PKL = REPO_ROOT / "models/bigram_sparse.pkl"
TRIGRAM_PKL = REPO_ROOT / "models/trigram.pkl"
TRANSFORMER_CKPT = REPO_ROOT / "models/transformer/ckpt.pt"

ALPHA_BIGRAM = 0.1


def load_valid_tokens(max_tokens=None):
    if not VALID_BIN.is_file():
        raise FileNotFoundError(f"Validation data not found: {VALID_BIN}. Run data prep first.")
    tokens = np.fromfile(VALID_BIN, dtype=np.uint16)
    if max_tokens is not None and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokens


# ---------- Bigram (dense: train_bigram.py) ----------
def eval_bigram_dense(valid_tokens, V):
    if not BIGRAM_DENSE_COUNTS.is_file() or not BIGRAM_DENSE_UNIGRAM.is_file():
        return None
    flat_counts = np.load(BIGRAM_DENSE_COUNTS)
    unigram = np.load(BIGRAM_DENSE_UNIGRAM)
    prev = valid_tokens[:-1].astype(np.uint64)
    nxt = valid_tokens[1:].astype(np.uint64)
    N = len(nxt)
    idx = prev * V + nxt
    counts = flat_counts[idx]
    unigram_prev = unigram[prev]
    denom = unigram_prev + ALPHA_BIGRAM * V
    probs = np.clip((counts.astype(np.float64) + ALPHA_BIGRAM) / denom, 1e-12, 1.0)
    avg_nll = -np.log(probs).sum() / max(N, 1)
    return math.exp(avg_nll), avg_nll


# ---------- Bigram (sparse: train_bigram_sparse.py) ----------
def eval_bigram_sparse(valid_tokens, V):
    if not BIGRAM_SPARSE_PKL.is_file():
        return None
    with open(BIGRAM_SPARSE_PKL, "rb") as f:
        data = pickle.load(f)
    bigram = data["bigram"]
    prev_count = data["prev_count"]
    alpha = data.get("alpha", ALPHA_BIGRAM)
    nll = 0.0
    n = 0
    for a, b in zip(valid_tokens[:-1], valid_tokens[1:]):
        a, b = int(a), int(b)
        p = (bigram[(a, b)] + alpha) / (prev_count[a] + alpha * V)
        p = max(p, 1e-12)
        nll += -math.log(p)
        n += 1
    avg_nll = nll / max(n, 1)
    return math.exp(avg_nll), avg_nll


# ---------- Trigram (train_trigram.py) ----------
def eval_trigram(valid_tokens, V):
    if not TRIGRAM_PKL.is_file():
        return None
    with open(TRIGRAM_PKL, "rb") as f:
        data = pickle.load(f)
    unigram = data["unigram"]
    bigram = data["bigram"]
    bigram_ctx = data["bigram_ctx"]
    trigram = data["trigram"]
    trigram_ctx = data["trigram_ctx"]
    total = data["total"]
    alpha = data.get("alpha", 0.1)
    lambdas = data.get("lambdas", (0.55, 0.30, 0.15))
    l3, l2, l1 = lambdas

    def p1(w):
        return (unigram.get(w, 0) + alpha) / (total + alpha * V)

    def p2(prev, w):
        return (bigram.get((prev, w), 0) + alpha) / (bigram_ctx.get(prev, 0) + alpha * V)

    def p3(prev2, prev1, w):
        return (trigram.get((prev2, prev1, w), 0) + alpha) / (trigram_ctx.get((prev2, prev1), 0) + alpha * V)

    nll = 0.0
    n = 0
    for i in range(2, len(valid_tokens)):
        a, b, c = int(valid_tokens[i - 2]), int(valid_tokens[i - 1]), int(valid_tokens[i])
        p = l3 * p3(a, b, c) + l2 * p2(b, c) + l1 * p1(c)
        p = max(p, 1e-12)
        nll += -math.log(p)
        n += 1
    avg_nll = nll / max(n, 1)
    return math.exp(avg_nll), avg_nll


# ---------- Transformer (train_transformer_causal.py) ----------
def eval_transformer(valid_tokens, device="cpu"):
    if not TRANSFORMER_CKPT.is_file():
        return None
    ckpt = torch.load(TRANSFORMER_CKPT, map_location=device)
    cfg = ckpt["config"]
    BLOCK_SIZE = cfg["BLOCK_SIZE"]
    VOCAB_SIZE = cfg["VOCAB_SIZE"]

    model = GPTMini.from_config(cfg, dropout_inference=0.0).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    losses = []
    with torch.no_grad():
        for start in range(0, len(valid_tokens) - BLOCK_SIZE - 1, BLOCK_SIZE):
            x = torch.tensor(valid_tokens[start : start + BLOCK_SIZE], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(valid_tokens[start + 1 : start + BLOCK_SIZE + 1], dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(x, targets=y)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            losses.append(loss.item())
    if not losses:
        return None
    avg_nll = sum(losses) / len(losses)
    return math.exp(avg_nll), avg_nll


def main():
    ap = argparse.ArgumentParser(description="Evaluate bigram, trigram, transformer (perplexity)")
    ap.add_argument("--valid-subset", type=int, default=None, help="Use first N validation tokens (default: all)")
    ap.add_argument("--no-plot", action="store_true", help="Skip saving the bar chart")
    ap.add_argument("--out", default=None, help="Output image path (default: models/evaluation.png)")
    args = ap.parse_args()

    print("Loading validation tokens...")
    valid_tokens = load_valid_tokens(max_tokens=args.valid_subset)
    tokenizer = Tokenizer.from_file(str(TOK_PATH))
    V = tokenizer.get_vocab_size()
    print(f"Valid tokens: {len(valid_tokens):,}, vocab size: {V}\n")

    results = {}

    # Bigram dense
    out = eval_bigram_dense(valid_tokens, V)
    if out is not None:
        ppl, nll = out
        results["Bigram (dense)"] = ppl
        print(f"Bigram (dense)     | perplexity: {ppl:.2f} | avg NLL: {nll:.4f}")
    else:
        print("Bigram (dense)     | (no model: run train_bigram.py)")

    # Bigram sparse
    out = eval_bigram_sparse(valid_tokens, V)
    if out is not None:
        ppl, nll = out
        results["Bigram (sparse)"] = ppl
        print(f"Bigram (sparse)    | perplexity: {ppl:.2f} | avg NLL: {nll:.4f}")
    else:
        print("Bigram (sparse)    | (no model: run train_bigram_sparse.py)")

    # Trigram
    out = eval_trigram(valid_tokens, V)
    if out is not None:
        ppl, nll = out
        results["Trigram"] = ppl
        print(f"Trigram            | perplexity: {ppl:.2f} | avg NLL: {nll:.4f}")
    else:
        print("Trigram            | (no model: run train_trigram.py)")

    # Transformer
    out = eval_transformer(valid_tokens)
    if out is not None:
        ppl, nll = out
        results["Transformer"] = ppl
        print(f"Transformer        | perplexity: {ppl:.2f} | avg NLL: {nll:.4f}")
    else:
        print("Transformer        | (no model: run train_transformer_causal.py)")

    if not results:
        print("\nNo models found. Train at least one model (bigram, trigram, or transformer).")
        return

    # Plot
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n(Install matplotlib for graphs: pip install matplotlib)")
        else:
            names = list(results.keys())
            values = list(results.values())
            colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"][: len(names)]
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=0.8)
            ax.set_ylabel("Perplexity (lower is better)", fontsize=12)
            ax.set_title("Validation perplexity: Bigram vs Trigram vs Transformer", fontsize=14)
            ax.set_ylim(0, max(values) * 1.15)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{val:.1f}", ha="center", fontsize=11)
            plt.tight_layout()
            out_path = Path(args.out) if args.out else REPO_ROOT / "models/evaluation.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=120)
            plt.close()
            print(f"\nSaved plot: {out_path}")


if __name__ == "__main__":
    main()
