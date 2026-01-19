# Import numpy for efficient array operations and binary file I/O
import numpy as np
# Import Tokenizer to load and use the trained tokenizer
from tokenizers import Tokenizer
# Import pathlib for cross-platform path handling
import pathlib

def encode_to_bin(tok, txt_path, bin_path):
    """
    Converts a text file to a binary file of token IDs.
    
    Args:
        tok: The tokenizer to use for encoding
        txt_path: Path to the input text file
        bin_path: Path to the output binary file
    """
    # Read the entire text file as a string
    # errors="ignore" skips any problematic characters instead of crashing
    text = pathlib.Path(txt_path).read_text(encoding="utf-8", errors="ignore")
    # Encode the text into token IDs (integers representing each token)
    # .ids gives us just the list of token IDs (not the full encoding object)
    ids = tok.encode(text).ids
    # Convert the list of token IDs to a numpy array
    # dtype=np.uint16 means each token ID is stored as a 16-bit unsigned integer (0-65535)
    # This works because our vocab_size is 8000, which fits in uint16
    arr = np.array(ids, dtype=np.uint16)
    # Write the array to a binary file (efficient storage format)
    arr.tofile(bin_path)
    # Print progress: show which files were converted and how many tokens
    print(f"{txt_path} -> {bin_path} | tokens: {len(arr):,}")

def main():
    # Load the trained tokenizer from the JSON file (created by train_tokenizer.py)
    tok = Tokenizer.from_file("tokenizer/bpe_tokenizer.json")

    # Convert the training text file to binary format
    encode_to_bin(tok, "data/processed/train.txt", "data/processed/train.bin")
    # Convert the validation text file to binary format
    encode_to_bin(tok, "data/processed/valid.txt", "data/processed/valid.bin")

    # Sanity check: decode first 200 tokens to verify the encoding worked correctly
    # Read the first 200 token IDs from the binary file
    sample = np.fromfile("data/processed/train.bin", dtype=np.uint16)[:200].tolist()
    # Print a header for the sanity check
    print("\n--- sanity decode ---\n")
    # Decode the token IDs back to text and print first 500 characters
    # This verifies that the tokenization process is reversible
    print(tok.decode(sample)[:500])

# Only run main() if this script is executed directly (not imported as a module)
if __name__ == "__main__":
    main()
