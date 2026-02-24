"""
Generic N-gram language model with interpolated backoff.

Usage:
  python scripts/train_ngram.py --n 3   # trigram (same idea as train_trigram_interpolated)
  python scripts/train_ngram.py --n 4   # 4-gram
  python scripts/train_ngram.py --n 5   # 5-gram (needs more data)

Trains counts for 1-gram, 2-gram, ... N-gram and combines them with learned
interpolation weights (lambdas). Smoothing is add-alpha at each level.
"""

import argparse
import math
import pathlib
import pickle
import time
from collections import Counter
from collections import defaultdict

import numpy as np
from tokenizers import Tokenizer


def load_tokens(bin_path: str, dtype=np.uint16) -> np.ndarray:
    return np.fromfile(bin_path, dtype=dtype)


def build_ngram_counts(tokens: np.ndarray, n: int):
    """
    Build counts for 1-gram, 2-gram, ... n-gram.
    Returns:
      counts[k] = Counter for (k+1)-gram: counts[k][tuple of (k+1) tokens] = count
      ctx[k] = Counter for context of (k+1)-gram: ctx[k][tuple of k tokens] = count
      total = len(tokens)
    """
    counts = [Counter() for _ in range(n)]   # counts[0]=unigram, counts[1]=bigram, ...
    ctx = [Counter() for _ in range(n)]     # ctx[0] not used; ctx[1]=unigram ctx for bigram, ...

    # Unigram
    for x in tokens:
        counts[0][int(x)] += 1
    total = len(tokens)

    # Bigram through n-gram
    for k in range(1, n):
        # (k+1)-gram: (t_0, ..., t_k) where we predict t_k given (t_0,...,t_{k-1})
        for i in range(len(tokens) - k):
            key = tuple(int(tokens[i + j]) for j in range(k + 1))
            context_key = key[:-1]
            counts[k][key] += 1
            ctx[k][context_key] += 1

    return counts, ctx, total


def prob_k(counts: list, ctx: list, total: int, context: tuple, w: int, V: int, alpha: float, k: int) -> float:
    """
    P(w | context) for (k+1)-gram level.
    context has length k: (prev_k, ..., prev_1) so we predict w after context.
    For k=0 (unigram): context is empty, P(w) = (count(w)+alpha)/(total+alpha*V).
    """
    if k == 0:
        return (counts[0][w] + alpha) / (total + alpha * V)
    denom = ctx[k][context] + alpha * V
    key = context + (w,)
    return (counts[k][key] + alpha) / denom


def p_interp(counts, ctx, total, context_tuple, w, V, alpha, lambdas):
    """
    context_tuple = (prev_n_minus_1, ..., prev_1) of length n-1 for n-gram model.
    We have n levels: unigram (context length 0), bigram (1), ..., n-gram (n-1).
    """
    n = len(lambdas)
    p = 0.0
    for k in range(n):
        ctx_k = context_tuple[-(k):] if k > 0 else ()   # last k tokens for k+1-gram
        p += lambdas[k] * prob_k(counts, ctx, total, ctx_k, w, V, alpha, k)
    return p


def perplexity(valid_tokens: np.ndarray, counts, ctx, total, n: int, V: int, alpha: float, lambdas) -> tuple[float, float]:
    nll = 0.0
    N = 0
    for i in range(n - 1, len(valid_tokens)):
        w = int(valid_tokens[i])
        context = tuple(int(valid_tokens[i - j]) for j in range(n - 1, 0, -1))  # (t_{i-n+1}, ..., t_{i-1})
        p = p_interp(counts, ctx, total, context, w, V, alpha, lambdas)
        p = max(p, 1e-12)
        nll += -math.log(p)
        N += 1
    avg_nll = nll / max(N, 1)
    return math.exp(avg_nll), avg_nll


def build_followers(counts: Counter, n: int):
    """For n-gram Counter, map context (n-1 tokens) -> list of (follower, count)."""
    followers = defaultdict(list)  # context -> [(w, count), ...]
    for key, cnt in counts[n - 1].items():
        ctx = key[:-1]
        w = key[-1]
        followers[ctx].append((w, cnt))
    return dict(followers)


def sample_next(context_tuple, followers, counts, ctx, total, V, alpha, lambdas, rng: np.random.Generator) -> int:
    """Sample next token given context (length n-1). Fallback to lower-order when context unseen."""
    n = len(lambdas)
    # Try highest-order context first
    if context_tuple in followers:
        cands, cnts = zip(*followers[context_tuple])
        cnts = np.array(cnts, dtype=np.float64) + alpha
        cnts /= cnts.sum()
        return int(rng.choice(list(cands), p=cnts))
    # Fallback: sample from interpolated distribution over full vocab (expensive) or top unigrams
    # Cheap fallback: sample from unigram
    top = [w for w, _ in counts[0].most_common(min(2000, len(counts[0])))]
    if not top:
        return int(rng.integers(0, V))
    probs = np.array([p_interp(counts, ctx, total, context_tuple, w, V, alpha, lambdas) for w in top], dtype=np.float64)
    probs /= probs.sum()
    return int(rng.choice(top, p=probs))


def generate(tokenizer: Tokenizer, counts, ctx, total, followers, V: int, n: int, alpha: float, lambdas,
             max_tokens: int = 250, seed: int = 42):
    rng = np.random.default_rng(seed)
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    if bos_id is None:
        start = [int(rng.integers(0, V)) for _ in range(n - 1)]
    else:
        start = [int(bos_id)] * (n - 1)

    out = list(start)
    for _ in range(max_tokens - (n - 1)):
        context = tuple(out[-(n - 1):])
        nxt = sample_next(context, followers, counts, ctx, total, V, alpha, lambdas, rng)
        out.append(nxt)
        if eos_id is not None and nxt == eos_id:
            break
    return tokenizer.decode(out)


def main():
    parser = argparse.ArgumentParser(description="Train interpolated N-gram LM")
    parser.add_argument("--n", type=int, default=4, help="N-gram order (e.g. 4 for 4-gram)")
    parser.add_argument("--alpha", type=float, default=0.1, help="Add-alpha smoothing")
    parser.add_argument("--max-tokens", type=int, default=5_000_000, help="Cap training tokens (0 = use all). Default 0 for fair comparison with bigram.")
    parser.add_argument("--train-bin", type=str, default="data/processed/train.bin")
    parser.add_argument("--valid-bin", type=str, default="data/processed/valid.bin")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/bpe_tokenizer.json")
    parser.add_argument("--out-dir", type=str, default="models")
    args = parser.parse_args()

    n = max(2, args.n)
    alpha = args.alpha
    max_tokens = args.max_tokens

    train_bin = args.train_bin
    valid_bin = args.valid_bin
    tok_path = args.tokenizer

    start_time = time.time()

    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(tok_path)
    V = tokenizer.get_vocab_size()
    print(f"Vocab size: {V:,}")

    print("Loading tokens...")
    train_tokens = load_tokens(train_bin)
    valid_tokens = load_tokens(valid_bin)
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Valid tokens: {len(valid_tokens):,}")

    if max_tokens > 0 and len(train_tokens) > max_tokens:
        train_tokens = train_tokens[:max_tokens]
        print(f"Capped to first {max_tokens:,} train tokens")

    print(f"\nBuilding {n}-gram counts...")
    t0 = time.time()
    counts, ctx, total = build_ngram_counts(train_tokens, n)
    print(f"Counts built in {time.time() - t0:.2f}s")
    for k in range(n):
        print(f"  Level {k+1}-gram unique: {len(counts[k]):,}")

    # Default lambdas: more weight on higher order (sum = 1)
    lambdas = np.ones(n) / n
    # Slight bias toward higher n-gram
    for k in range(n):
        lambdas[k] = (k + 1) / ((n * (n + 1)) / 2)
    lambdas = tuple(lambdas.tolist())
    print(f"Lambdas: {lambdas}")

    print("\nEvaluating perplexity...")
    ppl, avg_nll = perplexity(valid_tokens, counts, ctx, total, n, V, alpha, lambdas)
    print(f"Valid avg NLL: {avg_nll:.4f}")
    print(f"Valid perplexity: {ppl:.2f}")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    model_path = out_dir / f"ngram_n{n}_interpolated.pkl"
    followers = build_followers(counts, n)
    with open(model_path, "wb") as f:
        pickle.dump({
            "n": n,
            "counts": counts,
            "ctx": ctx,
            "total": total,
            "V": V,
            "alpha": alpha,
            "lambdas": lambdas,
            "followers": followers,
        }, f)
    print(f"\nSaved: {model_path}")

    print("\n--- N-gram sample ---\n")
    text = generate(tokenizer, counts, ctx, total, followers, V, n, alpha, lambdas, max_tokens=250, seed=42)
    print(text)

    print(f"\n=== Total time: {time.time() - start_time:.2f}s ===")


if __name__ == "__main__":
    main()
