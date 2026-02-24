import math
import pathlib
import pickle
from collections import Counter

import numpy as np
from tokenizers import Tokenizer

TRAIN_BIN = pathlib.Path("data/processed/train.bin")
VALID_BIN = pathlib.Path("data/processed/valid.bin")
TOK_PATH = pathlib.Path("tokenizer/bpe_tokenizer.json")
OUT_PATH = pathlib.Path("models/trigram.pkl")

ALPHA = 0.1  # smoothing
# (trigram, bigram, unigram). Less trigram weight often beats 0.7 when many contexts are rare.
LAMBDAS = (0.55, 0.30, 0.15)


def load_tokens(path: pathlib.Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint16)


def build_counts(tokens: np.ndarray):
    """
    Build unigram, bigram, trigram counts (sparse).
    Also store context totals needed for normalization.
    """
    unigram = Counter()
    bigram = Counter()
    bigram_ctx = Counter()
    trigram = Counter()
    trigram_ctx = Counter()

    # unigram counts
    for x in tokens:
        unigram[int(x)] += 1

    # bigram counts
    for a, b in zip(tokens[:-1], tokens[1:]):
        a = int(a); b = int(b)
        bigram[(a, b)] += 1
        bigram_ctx[a] += 1

    # trigram counts
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
             prev2, prev1, w, V, alpha, lambdas) -> float:
    l3, l2, l1 = lambdas
    return (
        l3 * p3(trigram, trigram_ctx, prev2, prev1, w, V, alpha)
        + l2 * p2(bigram, bigram_ctx, prev1, w, V, alpha)
        + l1 * p1(unigram, total, w, V, alpha)
    )


def perplexity(valid_tokens: np.ndarray, unigram, bigram, bigram_ctx, trigram, trigram_ctx, total,
               V: int, alpha: float, lambdas):
    """
    Evaluate interpolated trigram model on validation tokens.
    """
    nll = 0.0
    N = 0

    for a, b, c in zip(valid_tokens[:-2], valid_tokens[1:-1], valid_tokens[2:]):
        a = int(a); b = int(b); c = int(c)
        p = p_interp(unigram, bigram, bigram_ctx, trigram, trigram_ctx, total, a, b, c, V, alpha, lambdas)
        nll += -math.log(p)
        N += 1

    avg_nll = nll / max(N, 1)
    ppl = math.exp(avg_nll)
    return ppl, avg_nll


def build_trigram_followers(trigram: Counter):
    """
    followers[(a,b)] = (list_of_next_tokens, list_of_counts)
    for efficient sampling.
    """
    followers = {}
    for (a, b, c), cnt in trigram.items():
        key = (a, b)
        if key not in followers:
            followers[key] = ([], [])
        followers[key][0].append(c)
        followers[key][1].append(cnt)
    return followers


def sample_next(prev2, prev1, followers, unigram, V, rng: np.random.Generator) -> int:
    key = (prev2, prev1)
    if key not in followers:
        # fallback: sample from frequent unigrams
        top = [w for w, _ in unigram.most_common(200)]
        weights = np.array([unigram[w] for w in top], dtype=np.float64)
        weights /= weights.sum()
        return int(rng.choice(top, p=weights))

    cands, counts = followers[key]
    counts = np.array(counts, dtype=np.float64)
    probs = counts / counts.sum()
    return int(rng.choice(cands, p=probs))


def generate(tokenizer: Tokenizer, trigram: Counter, unigram: Counter,
             max_new_tokens=250, seed=42):
    rng = np.random.default_rng(seed)
    followers = build_trigram_followers(trigram)
    V = tokenizer.get_vocab_size()

    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")

    if bos is None:
        prev2 = int(rng.integers(0, V))
        prev1 = int(rng.integers(0, V))
        out = [prev2, prev1]
    else:
        prev2 = prev1 = int(bos)
        out = [prev2, prev1]

    for _ in range(max_new_tokens):
        nxt = sample_next(prev2, prev1, followers, unigram, V, rng)
        out.append(nxt)
        prev2, prev1 = prev1, nxt
        if eos is not None and nxt == eos:
            break

    return tokenizer.decode(out)


def main():
    tokenizer = Tokenizer.from_file(str(TOK_PATH))
    V = tokenizer.get_vocab_size()

    train_tokens = load_tokens(TRAIN_BIN)
    valid_tokens = load_tokens(VALID_BIN)

    print(f"Vocab size: {V}")
    print(f"Train tokens: {len(train_tokens):,} (using all — no cap)")
    print(f"Valid tokens: {len(valid_tokens):,}")

    print("\nCounting unigrams, bigrams, trigrams...")
    unigram, bigram, bigram_ctx, trigram, trigram_ctx, total = build_counts(train_tokens)

    print(f"Unique unigrams: {len(unigram):,}")
    print(f"Unique bigrams:  {len(bigram):,}")
    print(f"Unique trigrams: {len(trigram):,}")

    print("\nEvaluating interpolated trigram model...")
    ppl, avg_nll = perplexity(valid_tokens, unigram, bigram, bigram_ctx, trigram, trigram_ctx, total, V, ALPHA, LAMBDAS)
    print(f"Valid avg NLL: {avg_nll:.4f}")
    print(f"Valid perplexity: {ppl:.2f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(
            {
                "unigram": unigram,
                "bigram": bigram,
                "bigram_ctx": bigram_ctx,
                "trigram": trigram,
                "trigram_ctx": trigram_ctx,
                "total": total,
                "V": V,
                "alpha": ALPHA,
                "lambdas": LAMBDAS,
            },
            f,
        )
    print(f"\nSaved model: {OUT_PATH}")

    print("\n--- Trigram sample ---\n")
    print(generate(tokenizer, trigram, unigram, max_new_tokens=300, seed=42))


if __name__ == "__main__":
    main()