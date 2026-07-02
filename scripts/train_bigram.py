"""
Train a bigram language model.

A bigram estimates P(next | prev) from token pair counts.
Add-α smoothing prevents zero probabilities for unseen pairs.

Usage:
  python scripts/train_bigram.py
  python scripts/train_bigram.py --alpha 0.01
"""

import argparse
import math
import pathlib

import numpy as np
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

TRAIN_BIN = pathlib.Path("data/processed/train.bin")
VALID_BIN = pathlib.Path("data/processed/valid.bin")
TOK_PATH  = pathlib.Path("tokenizer/bpe_tokenizer.json")
OUT_DIR   = pathlib.Path("models")

DEFAULT_ALPHA  = 0.1   # smoothing strength: higher → more uniform distribution
SAMPLE_TOKENS  = 200
SEED           = 42


# ── data ───────────────────────────────────────────────────────────────────────

def load_tokens(path: pathlib.Path) -> np.ndarray:
    """Read the flat uint16 binary file into an int32 array."""
    return np.fromfile(path, dtype=np.uint16).astype(np.int32)


# ── model ──────────────────────────────────────────────────────────────────────

def build_counts(tokens: np.ndarray, vocab_size: int) -> np.ndarray:
    """
    Build a [vocab_size, vocab_size] count matrix.

    counts[i, j] = number of times token j followed token i in training data.

    np.add.at is used instead of a Python loop for speed — it handles repeated
    indices correctly (unlike counts[prev, next] += 1 which would miss duplicates).
    """
    counts = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    np.add.at(counts, (tokens[:-1], tokens[1:]), 1)
    return counts


def compute_perplexity(tokens: np.ndarray, counts: np.ndarray,
                       vocab_size: int, alpha: float) -> float:
    """
    Perplexity = exp( -mean( log P(next | prev) ) )

    Fully vectorised: looks up all consecutive pairs at once instead of looping.

    A perplexity of K means the model is as uncertain as if it picked uniformly
    from K tokens at each step.  Lower is better.  Random baseline = vocab_size.
    """
    prev     = tokens[:-1]              # shape [N]
    nxt      = tokens[1:]               # shape [N]
    row_sums = counts.sum(axis=1)       # total counts per previous token

    pair_counts = counts[prev, nxt]                         # count(prev→next)
    denom       = row_sums[prev] + alpha * vocab_size        # smoothed denominator
    log_probs   = np.log((pair_counts + alpha) / denom)     # log P(next|prev)

    return float(np.exp(-log_probs.mean()))


# ── generation ─────────────────────────────────────────────────────────────────

def generate(counts: np.ndarray, vocab_size: int, alpha: float,
             start_id: int, max_tokens: int, seed: int) -> list[int]:
    """
    Autoregressively sample token IDs from P(next | current).

    At each step: look up the row for the current token, apply smoothing,
    normalise to a probability distribution, then sample.
    """
    rng      = np.random.default_rng(seed)
    row_sums = counts.sum(axis=1).astype(np.float64)

    ids     = [start_id]
    current = start_id

    for _ in range(max_tokens):
        # Smoothed probability row for the current token
        row  = (counts[current].astype(np.float64) + alpha)
        row /= (row_sums[current] + alpha * vocab_size)
        current = int(rng.choice(vocab_size, p=row))
        ids.append(current)

    return ids


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Train a bigram language model")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                    help="Add-α smoothing (default: 0.1)")
    ap.add_argument("--sample-tokens", type=int, default=SAMPLE_TOKENS)
    args = ap.parse_args()

    # Load tokenizer just to get vocab size and special token IDs
    tok        = Tokenizer.from_file(str(TOK_PATH))
    tok.decoder = ByteLevelDecoder()  # converts Ġ → space in tok.decode()
    vocab_size = tok.get_vocab_size()
    bos_id     = tok.token_to_id("<bos>") or 0
    print(f"vocab_size : {vocab_size}")

    # Load token sequences from binary files
    print("Loading tokens ...")
    train_tokens = load_tokens(TRAIN_BIN)
    valid_tokens = load_tokens(VALID_BIN)
    print(f"train : {len(train_tokens):>10,} tokens")
    print(f"valid : {len(valid_tokens):>10,} tokens")

    # Build count matrix
    print("\nBuilding bigram counts ...")
    counts = build_counts(train_tokens, vocab_size)
    nonzero = int(np.count_nonzero(counts))
    total   = vocab_size ** 2
    print(f"Non-zero pairs : {nonzero:,} / {total:,}  ({100*nonzero/total:.1f}% dense)")

    # Save for use by evaluate_models.py
    OUT_DIR.mkdir(exist_ok=True)
    np.save(OUT_DIR / "bigram_counts.npy", counts)
    print(f"Saved → {OUT_DIR}/bigram_counts.npy")

    # Evaluate
    print("\nEvaluating ...")
    # Use a 500k-token sample of train to keep this fast
    train_ppl = compute_perplexity(train_tokens[:500_000], counts, vocab_size, args.alpha)
    valid_ppl = compute_perplexity(valid_tokens, counts, vocab_size, args.alpha)
    print(f"Train perplexity : {train_ppl:>8.2f}")
    print(f"Valid perplexity : {valid_ppl:>8.2f}")
    print(f"(Random baseline : {vocab_size})")

    # Generate a sample
    print("\n--- Bigram sample ---\n")
    ids  = generate(counts, vocab_size, args.alpha, bos_id, args.sample_tokens, SEED)
    text = tok.decode(ids)
    print(text)


if __name__ == "__main__":
    main()
