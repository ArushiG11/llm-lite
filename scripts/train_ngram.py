"""
Train an N-gram language model with linear interpolation.

Interpolation blends N orders together so that a zero high-order count
never produces zero probability — lower-order models act as a safety net.

  P(next | ctx) = λ_N·P_N(next|full_ctx) + ... + λ_1·P_1(next)

Usage:
  python scripts/train_ngram.py          # trigram (N=3) by default
  python scripts/train_ngram.py --n 4
  python scripts/train_ngram.py --n 2   # same as bigram but with interpolation
"""

import argparse
import math
import pathlib
import pickle
from collections import defaultdict

import numpy as np
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

TRAIN_BIN = pathlib.Path("data/processed/train.bin")
VALID_BIN = pathlib.Path("data/processed/valid.bin")
TOK_PATH  = pathlib.Path("tokenizer/bpe_tokenizer.json")
OUT_DIR   = pathlib.Path("models")

DEFAULT_N      = 3
DEFAULT_ALPHA  = 0.001   # small alpha: adds only alpha*vocab pseudo-counts per context
SAMPLE_TOKENS  = 200
SEED           = 42


# ── data ───────────────────────────────────────────────────────────────────────

def load_tokens(path: pathlib.Path) -> list[int]:
    """Read flat uint16 binary into a Python list of ints."""
    return np.fromfile(path, dtype=np.uint16).astype(np.int32).tolist()


# ── model ──────────────────────────────────────────────────────────────────────

def build_ngram_counts(tokens: list[int], n: int) -> tuple[list[dict], list[dict]]:
    """
    Build count dicts for all orders from 1 up to n.

    Returns:
      ngram_counts  — list of dicts, index k holds k+1-gram counts
                       ngram_counts[0][()]         = {next: count}  ← unigram
                       ngram_counts[1][(w1,)]      = {next: count}  ← bigram
                       ngram_counts[2][(w1,w2)]    = {next: count}  ← trigram
      ctx_totals    — total tokens seen after each context
                       ctx_totals[1][(w1,)] = sum of ngram_counts[1][(w1,)].values()

    Using dicts (sparse) instead of dense arrays because for N≥3 most possible
    n-grams never appear in training — a dense array would be mostly zeros.
    """
    # defaultdict of defaultdict(int) for each order
    ngram_counts: list[dict] = [defaultdict(lambda: defaultdict(int)) for _ in range(n)]
    ctx_totals:   list[dict] = [defaultdict(int) for _ in range(n)]

    for i in range(len(tokens) - n + 1):
        next_tok = tokens[i + n - 1]
        for k in range(n):
            # context is the k tokens immediately before next_tok
            ctx = tuple(tokens[i + (n - 1 - k) : i + (n - 1)])  # length k
            ngram_counts[k][ctx][next_tok] += 1
            ctx_totals[k][ctx] += 1

    return ngram_counts, ctx_totals


def prob_kth_order(next_tok: int, ctx: tuple, ngram_counts: list[dict],
                   ctx_totals: list[dict], vocab_size: int, alpha: float) -> float:
    """
    P_k(next | ctx) with add-α smoothing for the k-th order model.
    ctx has length k (k=0 for unigram, k=1 for bigram, etc.)
    """
    k          = len(ctx)
    # Use .get(ctx, {}) because defaultdict factories aren't preserved by pickle.
    count_next = ngram_counts[k].get(ctx, {}).get(next_tok, 0)
    count_ctx  = ctx_totals[k].get(ctx, 0)
    return (count_next + alpha) / (count_ctx + alpha * vocab_size)


def interpolated_prob(next_tok: int, context: tuple, lambdas: list[float],
                      ngram_counts: list[dict], ctx_totals: list[dict],
                      vocab_size: int, alpha: float) -> float:
    """
    Blend all orders using interpolation weights (lambdas).

    context is the full N-1 length history.  For each order k, we take
    only the last k tokens as the context (suffix of the full context).
    """
    p = 0.0
    n = len(lambdas)  # number of orders
    for k in range(n):
        ctx_k = context[max(0, len(context) - k):]  # last k tokens
        p += lambdas[k] * prob_kth_order(next_tok, ctx_k, ngram_counts,
                                          ctx_totals, vocab_size, alpha)
    return p


def compute_perplexity(tokens: list[int], lambdas: list[float],
                       ngram_counts: list[dict], ctx_totals: list[dict],
                       vocab_size: int, alpha: float, n: int) -> float:
    """
    Evaluate perplexity over a token sequence using the interpolated model.
    """
    log_prob_sum = 0.0
    count        = 0

    for i in range(n - 1, len(tokens)):
        next_tok = tokens[i]
        context  = tuple(tokens[max(0, i - (n - 1)) : i])  # up to N-1 tokens
        p        = interpolated_prob(next_tok, context, lambdas, ngram_counts,
                                     ctx_totals, vocab_size, alpha)
        log_prob_sum += math.log(max(p, 1e-300))  # guard against log(0)
        count += 1

    return math.exp(-log_prob_sum / count)


# ── generation ─────────────────────────────────────────────────────────────────

def generate(lambdas: list[float], ngram_counts: list[dict], ctx_totals: list[dict],
             vocab_size: int, alpha: float, n: int,
             start_id: int, max_tokens: int, seed: int) -> list[int]:
    """
    Autoregressively sample using the interpolated N-gram model.

    At each step we compute the full vocabulary distribution, then sample.
    This is O(vocab_size) per step — fine for vocab_size=8000.
    """
    rng = np.random.default_rng(seed)
    ids = [start_id]

    for _ in range(max_tokens):
        context = tuple(ids[-(n - 1):])  # last N-1 tokens as context
        probs   = np.array([
            interpolated_prob(w, context, lambdas, ngram_counts, ctx_totals, vocab_size, alpha)
            for w in range(vocab_size)
        ], dtype=np.float64)
        probs  /= probs.sum()  # renormalise (smoothing can make them not sum exactly to 1)
        ids.append(int(rng.choice(vocab_size, p=probs)))

    return ids


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Train an interpolated N-gram model")
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help="N-gram order (2=bigram, 3=trigram, etc.)")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                    help="Add-α smoothing strength (default: 0.001)")
    ap.add_argument("--sample-tokens", type=int, default=SAMPLE_TOKENS)
    args = ap.parse_args()

    assert args.n >= 2, "N must be at least 2 (bigram)"

    tok         = Tokenizer.from_file(str(TOK_PATH))
    tok.decoder = ByteLevelDecoder()  # converts Ġ → space in tok.decode()
    vocab_size  = tok.get_vocab_size()
    bos_id      = tok.token_to_id("<bos>") or 0
    print(f"Order       : {args.n}-gram")
    print(f"Vocab size  : {vocab_size}")

    print("Loading tokens ...")
    train_tokens = load_tokens(TRAIN_BIN)
    valid_tokens = load_tokens(VALID_BIN)
    print(f"Train : {len(train_tokens):,} | Valid : {len(valid_tokens):,}")

    print(f"\nBuilding {args.n}-gram counts ...")
    # Use a subset for faster training; increase for better models
    ngram_counts, ctx_totals = build_ngram_counts(train_tokens[:2_000_000], args.n)
    for k in range(args.n):
        print(f"  order {k+1}: {len(ngram_counts[k]):,} unique contexts")

    # Interpolation weights: proportional to total observations at each order.
    # Higher-order models have sparser counts, so we weight them by how much
    # data they actually saw — contexts with more total counts get more weight.
    # This is better than a fixed exponential scheme when data is limited.
    raw = [sum(ctx_totals[k].values()) for k in range(args.n)]
    total   = sum(raw)
    lambdas = [r / total for r in raw]
    print(f"\nInterpolation weights: {[f'{l:.3f}' for l in lambdas]}")
    print(f"  (index 0 = unigram, index {args.n-1} = {args.n}-gram)")

    # Evaluate on a sample (full valid set, capped train sample for speed)
    print("\nEvaluating ...")
    eval_train = train_tokens[:50_000]
    train_ppl  = compute_perplexity(eval_train, lambdas, ngram_counts, ctx_totals,
                                    vocab_size, args.alpha, args.n)
    valid_ppl  = compute_perplexity(valid_tokens[:20_000], lambdas, ngram_counts,
                                    ctx_totals, vocab_size, args.alpha, args.n)
    print(f"Train perplexity : {train_ppl:>8.2f}")
    print(f"Valid perplexity : {valid_ppl:>8.2f}")
    print(f"(Random baseline : {vocab_size})")

    # Save model for evaluate_models.py
    OUT_DIR.mkdir(exist_ok=True)
    model_path = OUT_DIR / f"ngram_n{args.n}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "n": args.n,
            "lambdas": lambdas,
            "alpha": args.alpha,
            "vocab_size": vocab_size,
            "ngram_counts": [{k: dict(v) for k, v in d.items()} for d in ngram_counts],
            "ctx_totals": [dict(d) for d in ctx_totals],
        }, f)
    print(f"\nSaved → {model_path}")

    # Generate a sample
    print(f"\n--- {args.n}-gram sample ---\n")
    ids  = generate(lambdas, ngram_counts, ctx_totals, vocab_size, args.alpha,
                    args.n, bos_id, args.sample_tokens, SEED)
    text = tok.decode(ids)
    print(text)


if __name__ == "__main__":
    main()