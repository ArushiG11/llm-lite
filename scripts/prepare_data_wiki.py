"""
Flow 2: Wikipedia dump pipeline in one script.

Runs in order:
  1. download_wiki_dump.py   → data/wiki/dump/simplewiki-...xml.bz2
  2. extract_wiki_text.py   → data/wiki/extracted/wiki_text.txt
  3. split_train_valid.py   → data/processed/train.txt, valid.txt
  4. train_tokenizer.py     → tokenizer/bpe_tokenizer.json
  5. tokenize_to_bin.py     → data/processed/train.bin, valid.bin

Run from repo root:  python scripts/prepare_data_wiki.py

Requires Flow 2 deps:  pip install -r requirements.txt
"""
import subprocess
import sys
from pathlib import Path


def _check_wiki_deps():
    try:
        import mwxml  # noqa: F401
        import mwparserfromhell  # noqa: F401
        from tokenizers import Tokenizer  # noqa: F401  # steps 4–5
        import numpy  # noqa: F401  # step 5 (tokenize_to_bin)
    except ImportError as e:
        print("Missing dependency:", e)
        print("Install all deps:  pip install -r requirements.txt")
        sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent
STEPS = [
    "download_wiki_dump.py",
    "extract_wiki_text.py",
    "split_train_valid.py",
    "train_tokenizer.py",
    "tokenize_to_bin.py",
]


def main():
    _check_wiki_deps()
    print("Flow 2: Wikipedia dump → train/valid .bin + tokenizer\n")
    for i, name in enumerate(STEPS, 1):
        path = REPO_ROOT / "scripts" / name
        print(f"\n--- Step {i}/{len(STEPS)}: {name} ---\n")
        subprocess.run([sys.executable, str(path)], cwd=REPO_ROOT, check=True)
    print("\nDone. data/processed/train.bin, valid.bin and tokenizer/bpe_tokenizer.json are ready.\n")


if __name__ == "__main__":
    main()
