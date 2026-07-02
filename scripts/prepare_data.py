"""
Data preparation pipeline for llm-lite. Three stages:

  Stage 1 — data:      Download TinyStories, save train.txt / valid.txt
  Stage 2 — tokenizer: Train a BPE tokenizer on train.txt
  Stage 3 — encode:    Encode text to uint16 binary (train.bin / valid.bin)

Usage:
  python scripts/prepare_data.py                 # run all three stages
  python scripts/prepare_data.py --stage data
  python scripts/prepare_data.py --stage tokenizer
  python scripts/prepare_data.py --stage encode
  python scripts/prepare_data.py --vocab-size 4000
"""

import argparse
import json
import pathlib

import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
PROCESSED_DIR = pathlib.Path("data/processed")
TOKENIZER_DIR = pathlib.Path("tokenizer")

# ── dataset config ─────────────────────────────────────────────────────────────
# TinyStories: short, simple English stories designed for small LMs.
# Unlike Wikipedia, a tiny model can actually learn to generate coherent text
# from this dataset because the vocabulary and sentence structure are simpler.
DATASET_NAME = "roneneldan/TinyStories"
MAX_TRAIN_DOCS = 500_000   # ~500k stories is plenty; caps memory and train time
MIN_CHARS = 50             # skip stories shorter than this (usually bad data)

# ── tokenizer config ───────────────────────────────────────────────────────────
DEFAULT_VOCAB_SIZE = 8000  # fits in uint16 (max 65535); good for a small corpus


# ── helpers ────────────────────────────────────────────────────────────────────

def basic_clean(text: str) -> str:
    """Strip blank lines and trailing whitespace. Keep everything else."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


# ── Stage 1: download data ─────────────────────────────────────────────────────

def load_data() -> None:
    """
    Stream TinyStories from HuggingFace and save to data/processed/.

    TinyStories already provides train and validation splits, so we use them
    directly instead of doing our own random split.
    """
    from datasets import load_dataset  # imported here so the script can be read without datasets installed

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Streaming {DATASET_NAME} ...")

    ds = load_dataset(DATASET_NAME, streaming=True)

    def collect(split_name: str, max_docs: int) -> list[str]:
        docs = []
        for ex in ds[split_name]:
            txt = basic_clean(ex.get("text", ""))
            if len(txt) >= MIN_CHARS:
                docs.append(txt)
            if len(docs) >= max_docs:
                break
        return docs

    train_docs = collect("train", MAX_TRAIN_DOCS)
    # TinyStories has a "validation" split — use it directly
    valid_docs = collect("validation", 10_000)

    (PROCESSED_DIR / "train.txt").write_text("\n\n".join(train_docs), encoding="utf-8")
    (PROCESSED_DIR / "valid.txt").write_text("\n\n".join(valid_docs), encoding="utf-8")

    print(f"train : {len(train_docs):>7,} docs")
    print(f"valid : {len(valid_docs):>7,} docs")
    print(f"saved → {PROCESSED_DIR}/train.txt  &  valid.txt")


# ── Stage 2: train tokenizer ───────────────────────────────────────────────────

def train_tokenizer(vocab_size: int = DEFAULT_VOCAB_SIZE) -> None:
    """
    Train a BPE tokenizer on train.txt and save to tokenizer/.

    BPE (Byte Pair Encoding) works by:
      1. Start with every character as its own token.
      2. Find the most frequent adjacent pair (e.g. "th").
      3. Merge it into one token ("th" → token #42).
      4. Repeat until vocab_size is reached.

    So common words become single tokens, rare words get split into subwords.

    Special tokens:
      <pad>  padding (batching sequences of different lengths)
      <unk>  unknown token (word not in vocab)
      <bos>  beginning of sequence — prepended to every prompt at inference
      <eos>  end of sequence — model should learn to predict this when done
    """
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    TOKENIZER_DIR.mkdir(exist_ok=True)
    train_file = str(PROCESSED_DIR / "train.txt")

    tok = Tokenizer(BPE(unk_token="<unk>"))
    # ByteLevel pre-tokenizer splits on whitespace/punctuation at the byte level
    # so the tokenizer can represent any Unicode without out-of-vocabulary bytes.
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=2,  # ignore subwords that appear fewer than 2 times
    )
    tok.train(files=[train_file], trainer=trainer)

    tok_path = TOKENIZER_DIR / "bpe_tokenizer.json"
    tok.save(str(tok_path))

    # metadata.json is a reproducibility record — not needed by the model,
    # but useful to remember exactly how this tokenizer was trained.
    (TOKENIZER_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "tokenizer_file": str(tok_path),
                "model_type": "BPE",
                "vocab_size": vocab_size,
                "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"],
                "min_frequency": 2,
                "dataset": DATASET_NAME,
                "train_file": train_file,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved → {tok_path}  (vocab_size={vocab_size})")
    print(f"saved → {TOKENIZER_DIR}/metadata.json")


# ── Stage 3: encode to binary ──────────────────────────────────────────────────

def tokenize_to_bin() -> None:
    """
    Encode train.txt and valid.txt into flat binary files of uint16 token IDs.

    Why uint16?  Our vocab is 8000 tokens, which fits in uint16 (max 65535).
    Each token costs 2 bytes — so 1 million tokens = 2 MB. Efficient for
    reading random windows during training without loading everything into RAM.

    The training loop reads a random window of 256 token IDs from train.bin
    and treats them as one training example.
    """
    from tokenizers import Tokenizer
    from tqdm import tqdm

    tok = Tokenizer.from_file(str(TOKENIZER_DIR / "bpe_tokenizer.json"))

    for split in ("train", "valid"):
        txt_path = PROCESSED_DIR / f"{split}.txt"
        bin_path = PROCESSED_DIR / f"{split}.bin"

        # Read all documents (separated by double newline)
        # Encode one document at a time and stream to disk — avoids loading
        # hundreds of millions of token IDs into RAM at once.
        docs = txt_path.read_text(encoding="utf-8", errors="ignore").split("\n\n")
        docs = [d for d in docs if d.strip()]

        total = 0
        with open(bin_path, "wb") as f:
            for doc in tqdm(docs, desc=f"Encoding {split}", unit="doc"):
                ids = tok.encode(doc).ids
                if ids:
                    np.array(ids, dtype=np.uint16).tofile(f)
                    total += len(ids)

        print(f"{split:5s}: {total:>10,} tokens → {bin_path}")

    # Sanity check: decode the first 100 tokens to verify round-trip works
    sample_ids = np.fromfile(PROCESSED_DIR / "train.bin", dtype=np.uint16)[:100].tolist()
    print("\n--- sanity decode (first 100 tokens) ---")
    print(tok.decode(sample_ids))


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="llm-lite data preparation pipeline")
    ap.add_argument(
        "--stage",
        choices=["data", "tokenizer", "encode", "all"],
        default="all",
        help="Which stage to run (default: all)",
    )
    ap.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help=f"BPE vocabulary size (default: {DEFAULT_VOCAB_SIZE})",
    )
    args = ap.parse_args()

    if args.stage in ("data", "all"):
        load_data()
    if args.stage in ("tokenizer", "all"):
        train_tokenizer(args.vocab_size)
    if args.stage in ("encode", "all"):
        tokenize_to_bin()


if __name__ == "__main__":
    main()
