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
    
    Why:
        Binary format is much faster to read than text files, especially for large datasets.
        uint16 uses 2 bytes per token, which is efficient for vocab sizes up to 65,535.
    """
    # Read binary file and convert to numpy array with specified data type
    return np.fromfile(bin_path, dtype=dtype)


def build_counts(tokens: np.ndarray):
    """
    Build unigram, bigram, and trigram counts from token sequence.
    
    A trigram model uses 3 tokens of context: (token_{t-2}, token_{t-1}, token_t)
    We also need unigram and bigram counts for interpolation (combining all three models).
    
    Args:
        tokens: Array of token IDs from training data
    
    Returns:
        unigram: Counter mapping token -> count (how often each token appears)
        bigram: Counter mapping (token_a, token_b) -> count (how often b follows a)
        bigram_ctx: Counter mapping token_a -> count (how often a appears as first in bigram)
        trigram: Counter mapping (a, b, c) -> count (how often c follows a,b)
        trigram_ctx: Counter mapping (a, b) -> count (how often (a,b) appears as context)
        total: Total number of tokens in the sequence
    
    Why sparse (Counter):
        Most n-grams never appear in training data. Storing only observed ones saves memory.
        For example, with vocab size 8000, there are 8000^3 = 512 billion possible trigrams,
        but we only see a tiny fraction of them. Counter stores only what we actually see.
    
    How:
        - Unigrams: count each token individually
        - Bigrams: count pairs of consecutive tokens
        - Trigrams: count triplets of consecutive tokens
        - Context counts: needed for probability normalization
    """
    # Initialize counters for all n-gram types (all start empty)
    unigram = Counter()
    bigram = Counter()
    # Context counter: how many times each token appears as the first token in a bigram
    bigram_ctx = Counter()
    trigram = Counter()
    # Context counter: how many times each (token_a, token_b) pair appears as context
    trigram_ctx = Counter()

    # Unigram counts: count how often each individual token appears
    for x in tokens:
        # Convert numpy type to Python int (required for dictionary keys)
        # Increment count for this token
        unigram[int(x)] += 1

    # Bigram counts: count pairs of consecutive tokens (a, b)
    # Also count how often each token 'a' appears as the first token in a bigram
    # tokens[:-1] = all tokens except the last one
    # tokens[1:] = all tokens except the first one
    # zip pairs them: (token[0], token[1]), (token[1], token[2]), etc.
    for a, b in zip(tokens[:-1], tokens[1:]):
        # Convert to Python ints for dictionary keys
        a = int(a); b = int(b)
        # Increment count for this bigram pair (a followed by b)
        bigram[(a, b)] += 1
        # Increment count for token 'a' appearing as context (first token in bigram)
        bigram_ctx[a] += 1

    # Trigram counts: count triplets of consecutive tokens (a, b, c)
    # Also count how often each (a, b) pair appears as context for trigrams
    # tokens[:-2] = all tokens except last two
    # tokens[1:-1] = all tokens except first and last
    # tokens[2:] = all tokens except first two
    # zip creates triplets: (token[0], token[1], token[2]), (token[1], token[2], token[3]), etc.
    for a, b, c in zip(tokens[:-2], tokens[1:-1], tokens[2:]):
        # Convert to Python ints for dictionary keys
        a = int(a); b = int(b); c = int(c)
        # Increment count for this trigram (a, b, c)
        trigram[(a, b, c)] += 1
        # Increment count for (a, b) appearing as context (first two tokens in trigram)
        trigram_ctx[(a, b)] += 1

    # Calculate total number of tokens (needed for unigram probability normalization)
    total = len(tokens)
    # Return all counters and total count
    return unigram, bigram, bigram_ctx, trigram, trigram_ctx, total


def p1(unigram: Counter, total: int, w: int, V: int, alpha: float) -> float:
    """
    Calculate unigram probability P(w) using add-alpha smoothing.
    
    Unigram model: predicts next token based only on overall token frequency,
    ignoring all context. This is the simplest language model.
    
    Args:
        unigram: Counter of token frequencies
        total: Total number of tokens in training data
        w: Token ID to calculate probability for
        V: Vocabulary size (total number of possible tokens)
        alpha: Smoothing parameter (small positive number)
    
    Returns:
        Probability P(w) = probability of token w appearing
    
    Formula:
        P(w) = (count(w) + alpha) / (total + alpha * V)
    
    Why smoothing:
        Without smoothing, unseen tokens get probability 0, which breaks log calculations.
        Smoothing ensures every token has at least a tiny probability.
    """
    # Add-alpha smoothing: add alpha to numerator, alpha*V to denominator
    # This ensures even unseen tokens get a small probability
    return (unigram[w] + alpha) / (total + alpha * V)


def p2(bigram: Counter, bigram_ctx: Counter, prev: int, w: int, V: int, alpha: float) -> float:
    """
    Calculate bigram probability P(w | prev) using add-alpha smoothing.
    
    Bigram model: predicts next token based on the previous token.
    More context-aware than unigram, but less than trigram.
    
    Args:
        bigram: Counter of bigram frequencies
        bigram_ctx: Counter of how often each token appears as previous token
        prev: Previous token ID
        w: Next token ID to calculate probability for
        V: Vocabulary size
        alpha: Smoothing parameter
    
    Returns:
        Probability P(w | prev) = probability of token w given previous token is prev
    
    Formula:
        P(w | prev) = (count(prev, w) + alpha) / (count(prev) + alpha * V)
    
    Why:
        Bigrams capture local dependencies between adjacent tokens.
        For example, "the" is more likely to be followed by a noun than a verb.
    """
    # Calculate conditional probability: P(w | prev)
    # Numerator: count of (prev, w) bigram + smoothing
    # Denominator: count of prev appearing as context + smoothing
    return (bigram[(prev, w)] + alpha) / (bigram_ctx[prev] + alpha * V)


def p3(trigram: Counter, trigram_ctx: Counter, prev2: int, prev1: int, w: int, V: int, alpha: float) -> float:
    """
    Calculate trigram probability P(w | prev2, prev1) using add-alpha smoothing.
    
    Trigram model: predicts next token based on the previous TWO tokens.
    This captures longer-range dependencies than bigrams.
    
    Args:
        trigram: Counter of trigram frequencies
        trigram_ctx: Counter of how often each (prev2, prev1) pair appears as context
        prev2: Token two positions back
        prev1: Token one position back (immediately previous)
        w: Next token ID to calculate probability for
        V: Vocabulary size
        alpha: Smoothing parameter
    
    Returns:
        Probability P(w | prev2, prev1) = probability of token w given previous two tokens
    
    Formula:
        P(w | prev2, prev1) = (count(prev2, prev1, w) + alpha) / (count(prev2, prev1) + alpha * V)
    
    Why:
        Trigrams capture more context. For example, "I am" is more likely to be followed
        by "happy" than "the", but "I am the" might be followed by "king" or "one".
        More context = better predictions, but also more data sparsity (many trigrams never seen).
    """
    # Calculate conditional probability: P(w | prev2, prev1)
    # Numerator: count of (prev2, prev1, w) trigram + smoothing
    # Denominator: count of (prev2, prev1) appearing as context + smoothing
    return (trigram[(prev2, prev1, w)] + alpha) / (trigram_ctx[(prev2, prev1)] + alpha * V)


def p_interp(unigram, bigram, bigram_ctx, trigram, trigram_ctx, total,
             prev2, prev1, w, V, alpha, lambdas=(0.7, 0.25, 0.05)) -> float:
    """
    Calculate interpolated probability combining unigram, bigram, and trigram models.
    
    Interpolation: combine multiple n-gram models by weighted average.
    This is better than using trigrams alone because:
    1. Trigrams are sparse (many never seen) - fall back to bigrams/unigrams
    2. Unigrams are robust (always have data) - provide baseline
    3. Weighted combination balances specificity vs. reliability
    
    Args:
        unigram, bigram, bigram_ctx, trigram, trigram_ctx, total: Counters from training
        prev2: Token two positions back
        prev1: Token one position back
        w: Token to predict
        V: Vocabulary size
        alpha: Smoothing parameter
        lambdas: Interpolation weights (l3, l2, l1) that sum to 1.0
                 Default: (0.7, 0.25, 0.05) means 70% trigram, 25% bigram, 5% unigram
    
    Returns:
        Interpolated probability: weighted combination of all three models
    
    Formula:
        P(w | prev2, prev1) = l3 * P3(w | prev2, prev1) 
                           + l2 * P2(w | prev1)
                           + l1 * P1(w)
    
    Why interpolation:
        - Trigrams are specific but sparse (many contexts never seen)
        - Bigrams are less specific but more reliable
        - Unigrams are general but always have data
        - Weighted combination gives best of all worlds
    
    How:
        We weight each model's probability and sum them. The weights (lambdas) should sum to 1.0
        to ensure the result is a valid probability distribution.
    """
    # Unpack interpolation weights: l3 for trigram, l2 for bigram, l1 for unigram
    l3, l2, l1 = lambdas
    # Return weighted sum of all three probability estimates
    # l3 * trigram_prob: most specific, used most (70%)
    # l2 * bigram_prob: medium specificity, used moderately (25%)
    # l1 * unigram_prob: most general, used least (5%) but provides fallback
    return (
        l3 * p3(trigram, trigram_ctx, prev2, prev1, w, V, alpha)
        + l2 * p2(bigram, bigram_ctx, prev1, w, V, alpha)
        + l1 * p1(unigram, total, w, V, alpha)
    )


def perplexity(valid_tokens: np.ndarray, unigram, bigram, bigram_ctx, trigram, trigram_ctx, total,
               V: int, alpha: float, lambdas) -> tuple[float, float]:
    """
    Evaluate trigram interpolated perplexity on validation tokens.
    
    Perplexity measures how 'surprised' the model is on validation text.
    Lower perplexity = model is less surprised = better predictions.
    
    For trigrams: we predict token x_t using context (x_{t-2}, x_{t-1})
    We use interpolation to combine trigram, bigram, and unigram models.
    
    Args:
        valid_tokens: Array of token IDs from validation set
        unigram, bigram, bigram_ctx, trigram, trigram_ctx, total: Counters from training
        V: Vocabulary size
        alpha: Smoothing parameter
        lambdas: Interpolation weights
    
    Returns:
        Tuple of (perplexity, average_negative_log_likelihood)
        Lower is better for both metrics.
    
    How:
        1. For each trigram (a, b, c) in validation data:
           - a = token at position t-2
           - b = token at position t-1
           - c = token at position t (what we're trying to predict)
        2. Calculate interpolated probability P(c | a, b)
        3. Add negative log probability to NLL accumulator
        4. Average NLL and convert to perplexity
    """
    # Initialize negative log-likelihood accumulator
    nll = 0.0
    # Initialize counter for number of trigrams processed
    N = 0

    # Iterate through all trigrams in validation sequence
    # valid_tokens[:-2] = tokens at positions 0, 1, 2, ... (context: prev2)
    # valid_tokens[1:-1] = tokens at positions 1, 2, 3, ... (context: prev1)
    # valid_tokens[2:] = tokens at positions 2, 3, 4, ... (token to predict: w)
    # zip creates triplets: (token[0], token[1], token[2]), (token[1], token[2], token[3]), etc.
    for a, b, c in zip(valid_tokens[:-2], valid_tokens[1:-1], valid_tokens[2:]):
        # Convert numpy types to Python ints
        a = int(a); b = int(b); c = int(c)
        # Calculate interpolated probability of token c given context (a, b)
        p = p_interp(unigram, bigram, bigram_ctx, trigram, trigram_ctx, total, a, b, c, V, alpha, lambdas)
        # Add negative log probability to NLL accumulator
        # Lower probability = higher negative log = more surprised
        nll += -math.log(p)
        # Increment trigram counter
        N += 1

    # Calculate average negative log-likelihood (avoid division by zero)
    avg_nll = nll / max(N, 1)
    # Perplexity = e^(average NLL)
    # This measures the "effective vocabulary size" the model thinks it's choosing from
    return math.exp(avg_nll), avg_nll


def build_trigram_followers(trigram: Counter):
    """
    Build a lookup table for fast sampling during text generation.
    
    For each context (prev2, prev1), pre-compute which tokens can follow and their counts.
    This avoids scanning the entire trigram table during generation.
    
    Args:
        trigram: Counter of trigram frequencies
    
    Returns:
        Dictionary mapping (prev2, prev1) -> (list of follower tokens, list of counts)
        followers[(a,b)] = ([c1, c2, ...], [count1, count2, ...])
    
    Why:
        During generation, we need to sample the next token given context (prev2, prev1).
        Without this lookup, we'd need to scan all trigrams each time (slow).
        With this lookup, we can quickly find all possible followers for a given context.
    
    How:
        Iterate through all trigrams, group by context (prev2, prev1),
        and collect all followers (third token) and their counts.
    """
    # Initialize dictionary to store followers for each context
    followers = {}
    # Iterate through all trigrams in the counter
    for (a, b, c), cnt in trigram.items():
        # Create key from context (first two tokens)
        key = (a, b)
        # If this context hasn't been seen before, initialize lists for followers and counts
        if key not in followers:
            # Tuple of (list of follower tokens, list of counts)
            followers[key] = ([], [])
        # Add follower token c to the list
        followers[key][0].append(c)
        # Add count for this trigram to the list
        followers[key][1].append(cnt)
    # Return the lookup table
    return followers


def sample_next(prev2, prev1, followers, unigram, total, V, alpha, rng):
    """
    Sample the next token given context (prev2, prev1).
    
    For speed, we only sample among tokens that were observed following (prev2, prev1) in training.
    If context was never seen, fall back to sampling from frequent unigrams.
    
    Args:
        prev2: Token two positions back
        prev1: Token one position back
        followers: Lookup table from build_trigram_followers()
        unigram: Counter of unigram frequencies (for fallback)
        total: Total number of tokens (for unigram probability)
        V: Vocabulary size
        alpha: Smoothing parameter
        rng: Random number generator for sampling
    
    Returns:
        Sampled next token ID
    
    Why this approach:
        - Fast: only considers observed followers, not all V tokens
        - Fallback: if context never seen, use unigram distribution (better than random)
        - Approximation: we ignore unseen followers for speed, but they have tiny probability anyway
    
    How:
        1. Check if context (prev2, prev1) exists in followers lookup
        2. If not, fall back to top 2000 most frequent unigrams
        3. If yes, sample from observed followers using smoothed probabilities
    """
    # Create key from context
    key = (prev2, prev1)
    # Check if we've seen this context before
    if key not in followers:
        # Fallback: context never seen in training
        # Sample from top 2000 most frequent unigrams (fast approximation)
        # Why 2000? Balance between quality (more is better) and speed (fewer is faster)
        top = [w for w, _ in unigram.most_common(2000)]
        # Calculate smoothed unigram probabilities for these top tokens
        probs = np.array([(unigram[w] + alpha) for w in top], dtype=np.float64)
        # Normalize probabilities so they sum to 1
        probs /= probs.sum()
        # Sample one token according to probabilities
        return int(rng.choice(top, p=probs))

    # Context was seen: get list of observed followers and their counts
    cands, counts = followers[key]
    # Convert counts to numpy array for efficient computation
    counts = np.array(counts, dtype=np.float64)

    # Calculate smoothed probabilities over observed followers only
    # Add alpha for smoothing (ensures no zero probabilities)
    probs = (counts + alpha)
    # Normalize probabilities so they sum to 1
    # Note: This is an approximation - we're ignoring unseen followers for speed
    probs /= probs.sum()
    # Sample one follower according to probabilities
    return int(rng.choice(cands, p=probs))


def generate(tokenizer: Tokenizer, trigram: Counter, unigram: Counter, total: int, V: int,
             max_tokens=250, seed=42, alpha=0.1):
    """
    Generate text using the trigram interpolated model.
    
    Starts with beginning-of-sequence tokens (or random tokens) and samples
    next tokens one by one using trigram context until end-of-sequence or max_tokens.
    
    Args:
        tokenizer: Tokenizer for encoding/decoding text
        trigram: Counter of trigram frequencies
        unigram: Counter of unigram frequencies (for fallback sampling)
        total: Total number of tokens (for unigram probability)
        V: Vocabulary size
        max_tokens: Maximum number of tokens to generate
        seed: Random seed for reproducibility
        alpha: Smoothing parameter
    
    Returns:
        Generated text string
    
    How:
        1. Build followers lookup table for fast sampling
        2. Initialize with BOS tokens (or random if BOS doesn't exist)
        3. For each step: use (prev2, prev1) to sample next token
        4. Update context: shift prev2, prev1 forward
        5. Stop if EOS token generated or max_tokens reached
    """
    # Initialize random number generator with seed for reproducibility
    rng = np.random.default_rng(seed)
    # Build lookup table for fast sampling (only done once)
    followers = build_trigram_followers(trigram)

    # Get token IDs for special tokens (if they exist)
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")

    # Initialize context: need two previous tokens for trigram model
    if bos is None:
        # If no BOS token, start with two random tokens
        prev2 = int(rng.integers(0, V))
        prev1 = int(rng.integers(0, V))
        # Initialize output with these two starting tokens
        out = [prev2, prev1]
    else:
        # Start with BOS token (use it for both prev2 and prev1)
        prev2 = prev1 = int(bos)
        # Initialize output with BOS tokens
        out = [prev2, prev1]

    # Generate tokens one by one (max_tokens - 2 because we already have 2 starting tokens)
    for _ in range(max_tokens - 2):
        # Sample next token given context (prev2, prev1)
        nxt = sample_next(prev2, prev1, followers, unigram, total, V, alpha, rng)
        # Add sampled token to output
        out.append(nxt)
        # Update context: shift forward
        # prev2 becomes old prev1, prev1 becomes new token
        prev2, prev1 = prev1, nxt
        # Stop early if we generate the end-of-sequence token
        if eos is not None and nxt == eos:
            break

    # Decode the list of token IDs back into text
    return tokenizer.decode(out)


def main():
    """
    Main function: train trigram interpolated model, evaluate it, save it, and generate sample text.
    
    This script trains a more sophisticated language model than bigrams by:
    1. Using trigram context (2 previous tokens instead of 1)
    2. Interpolating trigram, bigram, and unigram models for robustness
    3. Handling data sparsity through smoothing and fallbacks
    """
    # Path to training token binary file (created by tokenize_to_bin.py)
    train_bin = "data/processed/train.bin"
    # Path to validation token binary file (created by tokenize_to_bin.py)
    valid_bin = "data/processed/valid.bin"
    # Path to trained tokenizer JSON file (created by train_tokenizer.py)
    tok_path = "tokenizer/bpe_tokenizer.json"

    # CPU safety cap: limit training tokens to avoid memory issues
    # Trigrams require more memory than bigrams (more combinations to count)
    # Increase this later if you have more RAM and want better model quality
    MAX_TRAIN_TOKENS = 5_000_000

    # Smoothing parameter: small positive number added to all counts
    # Prevents zero probabilities which break log calculations
    alpha = 0.1
    # Interpolation weights: (trigram_weight, bigram_weight, unigram_weight)
    # These should sum to 1.0 for valid probability distribution
    # Default: 70% trigram (most specific), 25% bigram (medium), 5% unigram (fallback)
    lambdas = (0.7, 0.25, 0.05)

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

    # Print statistics about the data
    print(f"Train tokens (full): {len(train_tokens):,}")
    print(f"Valid tokens:        {len(valid_tokens):,}")

    # Cap training tokens to avoid memory issues with trigram counting
    # Trigrams grow as O(V^3) in worst case, so limiting tokens helps
    if len(train_tokens) > MAX_TRAIN_TOKENS:
        # Take only the first MAX_TRAIN_TOKENS tokens
        train_tokens = train_tokens[:MAX_TRAIN_TOKENS]
        # Inform user about the limitation
        print(f"Using first {MAX_TRAIN_TOKENS:,} train tokens for trigram counting (CPU-safe).")

    # Build all n-gram counts from training data
    print("\nCounting unigrams, bigrams, trigrams...")
    unigram, bigram, bigram_ctx, trigram, trigram_ctx, total = build_counts(train_tokens)

    # Print statistics about what was learned
    print(f"Unique unigrams: {len(unigram):,}")
    print(f"Unique bigrams:  {len(bigram):,}")
    print(f"Unique trigrams: {len(trigram):,}")

    # Evaluate the model on validation data using perplexity
    print("\nEvaluating trigram interpolated model...")
    ppl, avg_nll = perplexity(valid_tokens, unigram, bigram, bigram_ctx, trigram, trigram_ctx, total, V, alpha, lambdas)
    # Print average negative log-likelihood (lower is better)
    print(f"Valid avg NLL: {avg_nll:.4f}")
    # Print perplexity (lower is better, measures model's "surprise")
    print(f"Valid perplexity: {ppl:.2f}")

    # Create models directory if it doesn't exist
    pathlib.Path("models").mkdir(exist_ok=True)
    # Save the trained model to a pickle file
    with open("models/trigram_interpolated.pkl", "wb") as f:
        # Save dictionary containing all model components needed for generation
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
    # Confirm model was saved
    print("\nSaved: models/trigram_interpolated.pkl")

    # Generate and print a sample text to demonstrate the model
    print("\n--- Trigram sample ---\n")
    print(generate(tokenizer, trigram, unigram, total, V, max_tokens=250, seed=42, alpha=alpha))


# Only run main() if this script is executed directly (not imported as a module)
if __name__ == "__main__":
    main()
