# Import the main Tokenizer class from the HuggingFace tokenizers library
from tokenizers import Tokenizer
# Import BPE (Byte Pair Encoding) model - a subword tokenization algorithm
from tokenizers.models import BPE
# Import BpeTrainer to train the BPE tokenizer on our data
from tokenizers.trainers import BpeTrainer
# Import ByteLevel pre-tokenizer that splits text into bytes/characters before BPE
from tokenizers.pre_tokenizers import ByteLevel
# Import pathlib for cross-platform path handling
import pathlib

def main():
    # Vocabulary size: number of unique tokens in the tokenizer
    # 8000 is CPU-friendly (smaller = faster, but less expressive)
    vocab_size = 8000
    # Path to the training text file (output from split_train_valid.py)
    train_file = "data/processed/train.txt"

    # Create a new tokenizer using BPE (Byte Pair Encoding) algorithm
    # unk_token="<unk>" means unknown tokens (not in vocabulary) will be represented as <unk>
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    # Set the pre-tokenizer to ByteLevel, which splits text into bytes/characters
    # add_prefix_space=False means we don't add a space at the beginning of text
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    # Create a trainer that will learn the BPE merges from our training data
    trainer = BpeTrainer(
        vocab_size=vocab_size,  # Target vocabulary size
        # Special tokens that have special meaning:
        # <pad> = padding token (for batching sequences of different lengths)
        # <unk> = unknown token (for words not in vocabulary)
        # <bos> = beginning of sequence token
        # <eos> = end of sequence token
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        # Minimum frequency: only include subwords that appear at least 2 times
        min_frequency=2,
    )

    # Train the tokenizer on the training file
    # This learns which byte pairs to merge to create the vocabulary
    tokenizer.train(files=[train_file], trainer=trainer)

    # Create the tokenizer directory if it doesn't exist
    pathlib.Path("tokenizer").mkdir(exist_ok=True)
    # Save the trained tokenizer to a JSON file for later use
    tokenizer.save("tokenizer/bpe_tokenizer.json")

    # Confirm that the tokenizer was saved successfully
    print("Saved tokenizer/tokenizer.json")

# Only run main() if this script is executed directly (not imported as a module)
if __name__ == "__main__":
    main()
