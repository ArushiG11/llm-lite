# Import pathlib for cross-platform path handling
import pathlib

# Path to the extracted Wikipedia text file (output from extract_wiki_text.py)
IN_PATH = pathlib.Path("data/wiki/extracted/wiki_text.txt")
# Path where we'll save the training dataset (99% of the data)
TRAIN_PATH = pathlib.Path("data/processed/train.txt")
# Path where we'll save the validation dataset (1% of the data)
VALID_PATH = pathlib.Path("data/processed/valid.txt")

# Create the output directory (data/processed/) if it doesn't exist
TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

def main():
    # Read the entire extracted text file as a single string
    # errors="ignore" skips any problematic characters instead of crashing
    text = IN_PATH.read_text(encoding="utf-8", errors="ignore")

    # Calculate the split point: 99% for training, 1% for validation
    # This is a common split ratio for language model training
    cut = int(len(text) * 0.99)

    # Write the first 99% of characters to the training file
    TRAIN_PATH.write_text(text[:cut], encoding="utf-8")
    # Write the remaining 1% of characters to the validation file
    VALID_PATH.write_text(text[cut:], encoding="utf-8")

    # Print confirmation and statistics
    print("Split complete!")
    print("train chars:", cut)
    print("valid chars:", len(text) - cut)

# Only run main() if this script is executed directly (not imported as a module)
if __name__ == "__main__":
    main()
