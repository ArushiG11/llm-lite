import math
import pathlib
import pickle
from collections import Counter

import numpy as np
from tokenizers import Tokenizer


def load_tokens(bin_path: str, dtype=np.uint16) -> np.ndarray:
    return np.fromfile(bin_path, dtype=dtype)


def build_counts(tokens: np.ndarray):
    """
    Build unigram, bigram, trigram counts (all sparse via Counter).
    Also build context counts for normalization.
    """
    unigram = Counter()
    bigram = Counter()
    bigram_ctx = Counter()
    trigram = Counter()
    trigram_ctx = Counter()

    # Unigram counts
    for x in tokens:
        unigram[int(x)] += 1

    # Bigram counts + bigram context
    for a, b in zip(tokens[:-1], tokens[1:]):
        a = int(a); b = int(b)
        bigram[(a, b)] += 1
        bigram_ctx[a] += 1

    # Trigram counts + trigram context
    for a, b, c in zip(tokens[:-2], tokens[1:-1], tokens[2:]):
        a = int(a); b = int(b); c = int(c)
        trigram[(a, b, c)] += 1
        trigram_ctx[(a, b)] += 1

    total = len(tokens)
    return unigram, bigram, bigram_ctx, trigram, trigram_ctx, total


def p1(unigram: Counter, total: int, w: int, V: int, alpha: float) -> float:
    return (unigram[w] + alpha) / (total + alpha * V)


def p2(bigram: Counter, bigram_ctx: Counter, prev: int, w: int, V: int, alpha: float) -> float:
    return (bigram[(prev, w)] + alpha) / (bigram_ctx[prev] + alpha * V)


def p3(trigram: Counter, trigram_ctx: Counter, prev2: int, prev1: int, w: int, V: int, alpha: float) -> float:
    return (trigram[(prev2, prev1, w)] + alpha) / (trigram_ctx[(prev2, prev1)] + alpha * V)


def p_interp(unigram, bigram, bigram_ctx, trigram, trigram_ctx, total,
             prev2, prev1, w, V, alpha, lambdas=(0.7, 0.25, 0.05)) -> float:
    l3, l2, l1 = lambdas
    return (
        l3 * p3(trigram, trigram_ctx, prev2, prev1, w, V, alpha)
        + l2 * p2(bigram, bigram_ctx, prev1, w, V, alpha)
        + l1 * p1(unigram, total, w, V, alpha)
    )


def perplexity(valid_tokens: np.ndarray, unigram, bigram, bigram_ctx, trigram, trigram_ctx, total,
               V: int, alpha: float, lambdas) -> tuple[float, float]:
    """
    Evaluate trigram interpolated perplexity on validation tokens:
    predict x_t using (x_{t-2}, x_{t-1})
    """
    nll = 0.0
    N = 0

    for a, b, c in zip(valid_tokens[:-2], valid_tokens[1:-1], valid_tokens[2:]):
        a = int(a); b = int(b); c = int(c)
        p = p_interp(unigram, bigram, bigram_ctx, trigram, trigram_ctx, total, a, b, c, V, alpha, lambdas)
        nll += -math.log(p)
        N += 1

    avg_nll = nll / max(N, 1)
    return math.exp(avg_nll), avg_nll


def build_trigram_followers(trigram: Counter):
    """
    followers[(a,b)] = list of (c, count)
    used to sample quickly without scanning whole trigram table each step.
    """
    followers = {}
    for (a, b, c), cnt in trigram.items():
        key = (a, b)
        if key not in followers:
            followers[key] = ([], [])
        followers[key][0].append(c)
        followers[key][1].append(cnt)
    return followers


def sample_next(prev2, prev1, followers, unigram, total, V, alpha, rng):
    key = (prev2, prev1)
    if key not in followers:
        # fallback: sample from frequent unigrams (fast)
        top = [w for w, _ in unigram.most_common(2000)]
        probs = np.array([(unigram[w] + alpha) for w in top], dtype=np.float64)
        probs /= probs.sum()
        return int(rng.choice(top, p=probs))

    cands, counts = followers[key]
    counts = np.array(counts, dtype=np.float64)

    # smoothed probabilities over candidates only
    probs = (counts + alpha)
    probs /= probs.sum()
    return int(rng.choice(cands, p=probs))


def generate(tokenizer: Tokenizer, trigram: Counter, unigram: Counter, total: int, V: int,
             max_tokens=250, seed=42, alpha=0.1):
    rng = np.random.default_rng(seed)
    followers = build_trigram_followers(trigram)

    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")

    if bos is None:
        prev2 = int(rng.integers(0, V))
        prev1 = int(rng.integers(0, V))
        out = [prev2, prev1]
    else:
        prev2 = prev1 = int(bos)
        out = [prev2, prev1]

    for _ in range(max_tokens - 2):
        nxt = sample_next(prev2, prev1, followers, unigram, total, V, alpha, rng)
        out.append(nxt)
        prev2, prev1 = prev1, nxt
        if eos is not None and nxt == eos:
            break

    return tokenizer.decode(out)


def main():
    train_bin = "data/processed/train.bin"
    valid_bin = "data/processed/valid.bin"
    tok_path = "tokenizer/bpe_tokenizer.json"

    # CPU safety cap (increase later if you want)
    MAX_TRAIN_TOKENS = 5_000_000

    # smoothing + interpolation
    alpha = 0.1
    lambdas = (0.7, 0.25, 0.05)

    tokenizer = Tokenizer.from_file(tok_path)
    V = tokenizer.get_vocab_size()
    print(f"Vocab size: {V:,}")

    train_tokens = load_tokens(train_bin)
    valid_tokens = load_tokens(valid_bin)

    print(f"Train tokens (full): {len(train_tokens):,}")
    print(f"Valid tokens:        {len(valid_tokens):,}")

    # cap train tokens
    if len(train_tokens) > MAX_TRAIN_TOKENS:
        train_tokens = train_tokens[:MAX_TRAIN_TOKENS]
        print(f"Using first {MAX_TRAIN_TOKENS:,} train tokens for trigram counting (CPU-safe).")

    print("\nCounting unigrams, bigrams, trigrams...")
    unigram, bigram, bigram_ctx, trigram, trigram_ctx, total = build_counts(train_tokens)

    print(f"Unique unigrams: {len(unigram):,}")
    print(f"Unique bigrams:  {len(bigram):,}")
    print(f"Unique trigrams: {len(trigram):,}")

    print("\nEvaluating trigram interpolated model...")
    ppl, avg_nll = perplexity(valid_tokens, unigram, bigram, bigram_ctx, trigram, trigram_ctx, total, V, alpha, lambdas)
    print(f"Valid avg NLL: {avg_nll:.4f}")
    print(f"Valid perplexity: {ppl:.2f}")

    pathlib.Path("models").mkdir(exist_ok=True)
    with open("models/trigram_interpolated.pkl", "wb") as f:
        pickle.dump(
            {
                "unigram": unigram,
                "bigram": bigram,
                "bigram_ctx": bigram_ctx,
                "trigram": trigram,
                "trigram_ctx": trigram_ctx,
                "total": total,
                "V": V,
                "alpha": alpha,
                "lambdas": lambdas,
                "max_train_tokens": MAX_TRAIN_TOKENS,
            },
            f,
        )
    print("\nSaved: models/trigram_interpolated.pkl")

    print("\n--- Trigram sample ---\n")
    print(generate(tokenizer, trigram, unigram, total, V, max_tokens=250, seed=42, alpha=alpha))


if __name__ == "__main__":
    main()
