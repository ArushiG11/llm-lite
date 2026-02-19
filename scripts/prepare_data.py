import os
import pathlib
import numpy as np

def write_wikitext():
    # Uses HuggingFace datasets
    from datasets import load_dataset

    out_dir = pathlib.Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    train_text = "\n".join(ds["train"]["text"])
    valid_text = "\n".join(ds["validation"]["text"])

    (out_dir / "train.txt").write_text(train_text, encoding="utf-8")
    (out_dir / "valid.txt").write_text(valid_text, encoding="utf-8")

    print("Saved train.txt and valid.txt (WikiText-2).")
    print("train chars:", len(train_text))
    print("valid chars:", len(valid_text))

def train_tokenizer(vocab_size=8000):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=2,
    )

    tok.train(files=["data/processed/train.txt"], trainer=trainer)

    pathlib.Path("tokenizer").mkdir(exist_ok=True)
    tok.save("tokenizer/bpe_tokenizer.json")
    print("Saved tokenizer/bpe_tokenizer.json")

def tokenize_to_bin():
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file("tokenizer/bpe_tokenizer.json")

    def encode_file(txt_path, bin_path):
        text = pathlib.Path(txt_path).read_text(encoding="utf-8", errors="ignore")
        ids = tok.encode(text).ids
        arr = np.array(ids, dtype=np.uint16)
        arr.tofile(bin_path)
        print(f"{txt_path} -> {bin_path} | tokens: {len(arr):,}")

    encode_file("data/processed/train.txt", "data/processed/train.bin")
    encode_file("data/processed/valid.txt", "data/processed/valid.bin")

    # sanity decode
    sample = np.fromfile("data/processed/train.bin", dtype=np.uint16)[:200].tolist()
    print("\n--- decoded sample ---\n")
    print(tok.decode(sample)[:500])

def main():
    write_wikitext()
    train_tokenizer(vocab_size=8000)
    tokenize_to_bin()

if __name__ == "__main__":
    main()
