# Import math module for mathematical operations (log, exp)
import math
# Import pathlib for cross-platform path handling
import pathlib
# Import pickle for saving/loading Python objects to/from binary files
import pickle
# Import Counter from collections for counting token frequencies efficiently
from collections import Counter

# Import numpy for efficient array operations and random number generation
import numpy as np
# Import Tokenizer from HuggingFace tokenizers library for encoding/decoding text
from tokenizers import Tokenizer


def load_tokens(bin_path: str, dtype=np.uint16) -> np.ndarray:
    """
    Load token IDs from a binary file.
    
    Args:
        bin_path: Path to the binary file containing token IDs
        dtype: Data type to read (default: uint16, which supports vocab sizes up to 65535)
    
    Returns:
        numpy array of token IDs
    """
    # Read binary file and convert to numpy array with specified data type
    return np.fromfile(bin_path, dtype=dtype)


def build_bigram_sparse(tokens: np.ndarray):
    """
    Build a sparse bigram model by counting token pairs.
    
    A bigram model predicts the next token given the previous token.
    Uses sparse representation (only stores observed pairs) for memory efficiency.
    
    Args:
        tokens: Array of token IDs
    
    Returns:
        bigram: Counter mapping (token_a, token_b) -> count of how often b follows a
        prev_count: Counter mapping token_a -> count of how often a appears as previous token
    
    Note:
        bigram[(a,b)] = how often token b follows token a
        prev_count[a] = how often token a appears as a "previous token"
    """
    # Initialize counters to store bigram frequencies and previous token counts
    bigram = Counter()
    prev_count = Counter()

    # Iterate through all consecutive token pairs (bigrams)
    # tokens[:-1] = all tokens except the last one
    # tokens[1:] = all tokens except the first one
    # zip pairs them up: (token[0], token[1]), (token[1], token[2]), etc.
    for a, b in zip(tokens[:-1], tokens[1:]):
        # Convert numpy types to Python ints for dictionary keys
        a = int(a)
        b = int(b)
        # Increment count for this bigram pair (a followed by b)
        bigram[(a, b)] += 1
        # Increment count for token a appearing as a previous token
        prev_count[a] += 1

    # Return both counters: bigram frequencies and previous token counts
    return bigram, prev_count


def prob(bigram: Counter, prev_count: Counter, a: int, b: int, V: int, alpha: float) -> float:
    """
    Calculate the probability P(b|a) using add-alpha (Laplace) smoothing.
    
    Add-alpha smoothing prevents zero probabilities by adding a small value (alpha)
    to all counts. This ensures even unseen bigrams get a small probability.
    
    Args:
        bigram: Counter of bigram frequencies
        prev_count: Counter of previous token frequencies
        a: Previous token ID
        b: Next token ID
        V: Vocabulary size (total number of possible tokens)
        alpha: Smoothing parameter (small positive number)
    
    Returns:
        Probability P(b|a) = P(next token is b | previous token is a)
    
    Formula:
        P(b|a) = (count(a,b) + alpha) / (count(a) + alpha * V)
    """
    # Add-alpha smoothing: add alpha to numerator and alpha*V to denominator
    # This ensures we never get probability 0, even for unseen bigrams
    return (bigram[(a, b)] + alpha) / (prev_count[a] + alpha * V)


def perplexity(tokens: np.ndarray, bigram: Counter, prev_count: Counter, V: int, alpha: float):
    """
    Calculate perplexity on a sequence of tokens.
    
    Perplexity measures how 'surprised' the model is on validation text.
    Lower perplexity = model is less surprised = better predictions.
    Perplexity = 2^(average negative log-likelihood)
    
    Args:
        tokens: Array of token IDs to evaluate
        bigram: Counter of bigram frequencies from training
        prev_count: Counter of previous token frequencies from training
        V: Vocabulary size
        alpha: Smoothing parameter
    
    Returns:
        Tuple of (perplexity, average_negative_log_likelihood)
        Lower is better for both metrics.
    """
    # Initialize negative log-likelihood accumulator
    nll = 0.0
    # Initialize counter for number of bigrams processed
    N = 0

    # Iterate through all consecutive token pairs in the sequence
    for a, b in zip(tokens[:-1], tokens[1:]):
        # Convert numpy types to Python ints
        a = int(a)
        b = int(b)
        # Calculate probability of this bigram using smoothed probability
        p = prob(bigram, prev_count, a, b, V, alpha)
        # Add negative log probability to NLL accumulator
        # Lower probability = higher negative log = more surprised
        nll += -math.log(p)
        # Increment bigram counter
        N += 1

    # Calculate average negative log-likelihood (avoid division by zero)
    avg_nll = nll / max(N, 1)
    # Perplexity = e^(average NLL) = 2^(average NLL in base 2)
    # This measures the "effective vocabulary size" the model thinks it's choosing from
    return math.exp(avg_nll), avg_nll


def sample_next(a: int, bigram: Counter, prev_count: Counter, V: int, alpha: float, rng: np.random.Generator) -> int:
    """
    Sample the next token given the previous token a.
    
    For speed, we only sample among tokens that were observed following 'a' in training.
    This is an approximation but much faster than sampling from all V tokens.
    
    Args:
        a: Previous token ID
        bigram: Counter of bigram frequencies
        prev_count: Counter of previous token frequencies
        V: Vocabulary size
        alpha: Smoothing parameter
        rng: Random number generator for sampling
    
    Returns:
        Sampled next token ID
    """
    # Lists to store tokens that follow 'a' and their counts
    followers = []
    counts = []

    # Iterate through all bigrams to find ones that start with token 'a'
    for (x, y), c in bigram.items():
        # If this bigram starts with 'a', record the follower and its count
        if x == a:
            followers.append(y)
            counts.append(c)

    # If no followers were found (unseen token), sample uniformly from vocabulary
    if not followers:
        return int(rng.integers(0, V))

    # Convert counts list to numpy array for efficient computation
    counts = np.array(counts, dtype=np.float64)

    # Calculate smoothed probabilities for observed followers
    # Denominator includes smoothing: count(a) + alpha * V
    denom = prev_count[a] + alpha * V
    # Numerator: count(a,b) + alpha for each follower b
    probs = (counts + alpha) / denom
    # Renormalize probabilities so they sum to 1 (only over observed followers)
    # This is an approximation - we're ignoring unseen followers for speed
    probs = probs / probs.sum()

    # Sample one follower according to the calculated probabilities
    return int(rng.choice(followers, p=probs))


def generate(tokenizer: Tokenizer, bigram: Counter, prev_count: Counter, V: int, alpha: float, max_tokens=200, seed=42) -> str:
    """
    Generate text using the bigram model.
    
    Starts with a beginning-of-sequence token (or random token) and samples
    next tokens one by one until end-of-sequence token or max_tokens reached.
    
    Args:
        tokenizer: Tokenizer for encoding/decoding text
        bigram: Counter of bigram frequencies
        prev_count: Counter of previous token frequencies
        V: Vocabulary size
        alpha: Smoothing parameter
        max_tokens: Maximum number of tokens to generate
        seed: Random seed for reproducibility
    
    Returns:
        Generated text string
    """
    # Initialize random number generator with seed for reproducibility
    rng = np.random.default_rng(seed)
    # Get token ID for beginning-of-sequence token (if it exists)
    bos = tokenizer.token_to_id("<bos>")
    # Get token ID for end-of-sequence token (if it exists)
    eos = tokenizer.token_to_id("<eos>")

    # If no BOS token exists, start with a random token
    if bos is None:
        # Sample a random token from the vocabulary
        prev = int(rng.integers(0, V))
        # Initialize output list with the starting token
        out = [prev]
    else:
        # Start with the beginning-of-sequence token
        prev = int(bos)
        # Initialize output list with BOS token
        out = [prev]

    # Generate tokens one by one (max_tokens - 1 because we already have the first token)
    for _ in range(max_tokens - 1):
        # Sample the next token given the previous token
        nxt = sample_next(prev, bigram, prev_count, V, alpha, rng)
        # Add the sampled token to output
        out.append(nxt)
        # Update previous token for next iteration
        prev = nxt
        # Stop early if we generate the end-of-sequence token
        if eos is not None and nxt == eos:
            break

    # Decode the list of token IDs back into text
    return tokenizer.decode(out)


def main():
    """
    Main function: train bigram model, evaluate it, save it, and generate sample text.
    """
    # Path to training token binary file (created by tokenize_to_bin.py)
    train_bin = "data/processed/train.bin"
    # Path to validation token binary file (created by tokenize_to_bin.py)
    valid_bin = "data/processed/valid.bin"
    # Path to trained tokenizer JSON file (created by train_tokenizer.py)
    tok_path = "tokenizer/bpe_tokenizer.json"

    # Smoothing value (small positive number)
    # Alpha = 0.1 means we add 0.1 to all counts for smoothing
    # Larger alpha = more uniform probabilities, smaller alpha = more confident in training data
    alpha = 0.1

    # Load the trained tokenizer from JSON file
    tokenizer = Tokenizer.from_file(tok_path)
    # Get vocabulary size (number of unique tokens)
    V = tokenizer.get_vocab_size()
    # Print vocabulary size with thousand separators for readability
    print(f"Vocab size: {V:,}")

    # Load token IDs from training binary file
    train_tokens = load_tokens(train_bin)
    # Load token IDs from validation binary file
    valid_tokens = load_tokens(valid_bin)
    # Print number of training tokens
    print(f"Train tokens: {len(train_tokens):,}")
    # Print number of validation tokens
    print(f"Valid tokens: {len(valid_tokens):,}")

    # Build the bigram model by counting token pairs in training data
    print("\nTraining (counting bigrams)...")
    bigram, prev_count = build_bigram_sparse(train_tokens)
    # Print number of unique bigram pairs found
    print(f"Unique bigrams: {len(bigram):,}")

    # Evaluate the model on validation data using perplexity
    print("\nEvaluating...")
    ppl, avg_nll = perplexity(valid_tokens, bigram, prev_count, V, alpha)
    # Print average negative log-likelihood (lower is better)
    print(f"Valid avg NLL: {avg_nll:.4f}")
    # Print perplexity (lower is better, measures model's "surprise")
    print(f"Valid perplexity: {ppl:.2f}")

    # Create models directory if it doesn't exist
    pathlib.Path("models").mkdir(exist_ok=True)
    # Save the trained model to a pickle file
    with open("models/bigram_sparse.pkl", "wb") as f:
        # Save dictionary containing all model components needed for generation
        pickle.dump({"bigram": bigram, "prev_count": prev_count, "V": V, "alpha": alpha}, f)
    # Confirm model was saved
    print("\nSaved: models/bigram_sparse.pkl")

    # Generate and print a sample text to demonstrate the model
    print("\n--- Bigram sample ---\n")
    print(generate(tokenizer, bigram, prev_count, V, alpha, max_tokens=200, seed=42))


# Only run main() if this script is executed directly (not imported as a module)
if __name__ == "__main__":
    main()
