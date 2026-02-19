import math
import pathlib
import time
import numpy as np
from tokenizers import Tokenizer


def load_tokens(bin_path: str, dtype=np.uint16) -> np.ndarray:
    """Load a .bin file of token IDs into a numpy array."""
    return np.fromfile(bin_path, dtype=dtype)

def build_bigram_counts(tokens: np.ndarray, vocab_size: int):
    """
    Build bigram counts:
      count[a, b] = how often token b follows token a
    We'll store counts in a flat array for speed:
      idx = a * V + b
    """
    V = vocab_size
    flat_counts = np.zeros(V * V, dtype=np.uint32)
    unigram = np.zeros(V, dtype=np.uint32)  # count of "previous token" occurrences

    prev = tokens[:-1]
    nxt = tokens[1:]

    # Track how many times each prev token appears (row totals)
    # unigram[a] = number of times token a was seen as a previous token
    np.add.at(unigram, prev, 1)

    # Bigram index trick: (prev, nxt) -> prev*V + nxt
    idx = prev.astype(np.uint64) * V + nxt.astype(np.uint64)
    np.add.at(flat_counts, idx, 1)

    return flat_counts, unigram

def bigram_prob(flat_counts: np.ndarray, unigram: np.ndarray, prev_id: int, next_id: int, V: int, alpha: float):
    """
    Add-alpha smoothing:
      P(next|prev) = (count(prev,next) + alpha) / (count(prev) + alpha*V)
    """
    c = flat_counts[prev_id * V + next_id]
    denom = unigram[prev_id] + alpha * V
    return (c + alpha) / denom

def perplexity(tokens: np.ndarray, flat_counts: np.ndarray, unigram: np.ndarray, V: int, alpha: float):
    """
    Compute perplexity on a token sequence using bigram probabilities.
    Perplexity = exp( average negative log likelihood )
    OPTIMIZED: Uses vectorized operations instead of Python loop
    """
    prev = tokens[:-1].astype(np.uint64)
    nxt = tokens[1:].astype(np.uint64)
    N = len(nxt)

    # Vectorized bigram probability computation
    # Get counts for all (prev, next) pairs at once
    idx = prev * V + nxt
    counts = flat_counts[idx]
    
    # Compute probabilities: (count + alpha) / (unigram[prev] + alpha*V)
    unigram_prev = unigram[prev]
    denom = unigram_prev + alpha * V
    probs = (counts.astype(np.float64) + alpha) / denom
    
    # Compute negative log likelihood (vectorized)
    # Clip probabilities to avoid log(0)
    probs = np.clip(probs, 1e-10, 1.0)
    nll = -np.log(probs).sum()

    avg_nll = nll / max(N, 1)
    ppl = math.exp(avg_nll)
    return ppl, avg_nll

def sample_next(flat_counts: np.ndarray, unigram: np.ndarray, prev_id: int, V: int, alpha: float, rng: np.random.Generator):
    """
    Sample next token from P(next|prev).
    """
    row = flat_counts[prev_id * V : (prev_id + 1) * V].astype(np.float64)
    probs = (row + alpha) / (unigram[prev_id] + alpha * V)

    # Numerical safety (should already sum to ~1)
    probs = probs / probs.sum()
    return int(rng.choice(V, p=probs))

def generate_text(tokenizer: Tokenizer, flat_counts: np.ndarray, unigram: np.ndarray, V: int, alpha: float,
                  max_tokens: int = 200, seed: int = 0):
    """
    Generate text by sampling tokens one-by-one.
    We'll start from <bos> if present; otherwise start from a random token.
    """
    rng = np.random.default_rng(seed)

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    if bos_id is None:
        # fallback if tokenizer doesn't have <bos>
        prev_id = int(rng.integers(0, V))
        out_ids = [prev_id]
    else:
        prev_id = int(bos_id)
        out_ids = [prev_id]

    for _ in range(max_tokens - 1):
        nxt = sample_next(flat_counts, unigram, prev_id, V, alpha, rng)
        out_ids.append(nxt)
        prev_id = nxt

        if eos_id is not None and nxt == eos_id:
            break

    return tokenizer.decode(out_ids)

# ---------- Main script ----------

def main():
    start_time = time.time()
    
    # Files from Step 1
    train_bin = "data/processed/train.bin"
    valid_bin = "data/processed/valid.bin"
    tok_path = "tokenizer/bpe_tokenizer.json"

    # Bigram smoothing (prevents zero probabilities)
    alpha = 0.1

    # Load tokenizer (to get vocab size + decode generated tokens)
    print("Loading tokenizer...")
    t0 = time.time()
    tokenizer = Tokenizer.from_file(tok_path)
    V = tokenizer.get_vocab_size()
    print(f"Vocab size: {V:,} (took {time.time() - t0:.2f}s)")

    print("\nLoading tokens...")
    t0 = time.time()
    train_tokens = load_tokens(train_bin)
    valid_tokens = load_tokens(valid_bin)
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Valid tokens: {len(valid_tokens):,}")
    print(f"Loading took {time.time() - t0:.2f}s")

    print("\nBuilding bigram counts (this is the 'training' step)...")
    t0 = time.time()
    flat_counts, unigram = build_bigram_counts(train_tokens, V)
    print(f"Building counts took {time.time() - t0:.2f}s")

    # Evaluate
    print("\nEvaluating on validation set...")
    t0 = time.time()
    ppl, avg_nll = perplexity(valid_tokens, flat_counts, unigram, V, alpha)
    print(f"Evaluation took {time.time() - t0:.2f}s")
    print(f"Validation avg NLL: {avg_nll:.4f}")
    print(f"Validation perplexity: {ppl:.2f}")

    # Save model
    print("\nSaving model...")
    t0 = time.time()
    out_dir = pathlib.Path("models")
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "bigram_counts.npy", flat_counts)
    np.save(out_dir / "bigram_unigram.npy", unigram)
    print(f"Saving took {time.time() - t0:.2f}s")
    print("Saved model to models/bigram_counts.npy and models/bigram_unigram.npy")

    # Generate samples
    print("\n--- Bigram sample (generated text) ---\n")
    t0 = time.time()
    text = generate_text(tokenizer, flat_counts, unigram, V, alpha, max_tokens=200, seed=42)
    print(f"Generation took {time.time() - t0:.2f}s\n")
    print(text)
    
    print(f"\n=== Total time: {time.time() - start_time:.2f}s ===")

if __name__ == "__main__":
    main()
