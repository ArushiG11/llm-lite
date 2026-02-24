# Import pathlib for cross-platform path handling
import pathlib

# Path to the extracted Wikipedia text file (output from extract_wiki_text.py)
IN_PATH = pathlib.Path("data/wiki/extracted/wiki_text.txt")
# Path where we'll save the training dataset
TRAIN_PATH = pathlib.Path("data/processed/train.txt")
# Path where we'll save the validation dataset
VALID_PATH = pathlib.Path("data/processed/valid.txt")

# Fraction of text used for training; the rest is validation (e.g. 0.9 = 90% train, 10% valid)
TRAIN_FRAC = 0.9

# Create the output directory (data/processed/) if it doesn't exist
TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

def main():
    # Read the entire extracted text file as a single string
    # errors="ignore" skips any problematic characters instead of crashing
    text = IN_PATH.read_text(encoding="utf-8", errors="ignore")

    # Split by character position: first TRAIN_FRAC for training, rest for validation
    cut = int(len(text) * TRAIN_FRAC)

    TRAIN_PATH.write_text(text[:cut], encoding="utf-8")
    VALID_PATH.write_text(text[cut:], encoding="utf-8")

    # Print confirmation and statistics
    print("Split complete!")
    print(f"train chars: {cut:,} ({100 * TRAIN_FRAC:.0f}%)")
    print(f"valid chars: {len(text) - cut:,} ({100 * (1 - TRAIN_FRAC):.0f}%)")

# Only run main() if this script is executed directly (not imported as a module)
if __name__ == "__main__":
    main()
