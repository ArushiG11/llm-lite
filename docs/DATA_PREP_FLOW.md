# Data preparation flow: splitting and tokenizing

You have **two ways** to get from raw data to token IDs ready for training. Both end at the same place: `data/processed/train.bin` and `data/processed/valid.bin`.

---

## Dataset: Tiny Shakespeare vs WikiText / Wikipedia

| | Tiny Shakespeare | WikiText / Wikipedia (Flow 1 & 2) |
|--|------------------|-----------------------------------|
| **Size** | ~1 MB, ~300k tokens | WikiText-2: ~2M tokens; Wikipedia: much more |
| **Content** | One author, plays/poetry | Diverse articles, standard LM benchmark |
| **Best for** | Quick demos, overfitting in minutes | Actually training an LLM, comparing perplexity |
| **Recommendation** | Fun for “can my model learn this?” | **Better for a real LLM** — more data, more variety, standard eval |

Use **WikiText or Wikipedia** (Flow 1 or Flow 2) when you care about a real language model. Use Tiny Shakespeare only for fast, small experiments.

---

## Flow 1: WikiText-2 (simplest, one script)

**Use this when you want to start quickly** with a small, pre-split dataset.

| Step | What happens | Input | Output |
|------|----------------|-------|--------|
| 1 | Download WikiText-2 and **split** (handled by HuggingFace) | — | `data/processed/train.txt`, `data/processed/valid.txt` |
| 2 | **Train BPE tokenizer** on train text | `train.txt` | `tokenizer/bpe_tokenizer.json` |
| 3 | **Tokenize** train and valid text to binary token IDs | `train.txt`, `valid.txt` | `train.bin`, `valid.bin` |

**Run:**
```bash
python scripts/prepare_data.py
```
This single script does all three: writes train/valid text, trains the tokenizer, then tokenizes to `.bin`.

---

## Flow 2: Wikipedia dump (recommended: one script)

**Use this when you want** your own split and raw Wikipedia text (e.g. Simple English Wikipedia).

| Step | What it does | Output |
|------|----------------|--------|
| 1 | Download Simple English Wikipedia dump | `data/wiki/dump/simplewiki-latest-pages-articles.xml.bz2` |
| 2 | Decompress, parse XML, strip wiki markup, one article per line | `data/wiki/extracted/wiki_text.txt` |
| 3 | **Split** text into train vs validation (90% / 10%) | `data/processed/train.txt`, `data/processed/valid.txt` |
| 4 | **Train BPE** on `train.txt` only | `tokenizer/bpe_tokenizer.json` |
| 5 | **Tokenize** train and valid to binary IDs | `data/processed/train.bin`, `data/processed/valid.bin` |

**Install Flow 2 deps first** (use the same Python you’ll run the script with):
```bash
pip install -r requirements.txt
# or, without a venv:  pip3 install -r requirements.txt
```

**Run (single command from repo root):**
```bash
python scripts/prepare_data_wiki.py
# or, without a venv:  python3 scripts/prepare_data_wiki.py
```
This runs all five steps in order. You can still run the individual scripts under `scripts/` if you need to re-run only part of the pipeline.

**Without a venv:** Deactivate the venv (`deactivate`), then install deps for your system Python and run:
```bash
pip3 install -r requirements.txt
python3 scripts/prepare_data_wiki.py
```

---

## Splitting (when it happens)

- **Flow 1:** No explicit split script. The dataset (`wikitext-2-raw-v1`) already has `train` and `validation` splits; `prepare_data.py` just writes them to `train.txt` and `valid.txt`.
- **Flow 2:** Splitting is done by `split_train_valid.py`: it reads the single extracted file and splits by **character position** (e.g. first 90% → train, last 10% → valid). So train and valid are contiguous chunks of the same corpus.

---

## Tokenizing (same in both flows)

1. **Train the tokenizer** on **training text only** (so the vocabulary is learned from train, not from valid).
2. **Encode** both train and valid text with that tokenizer → sequences of integer token IDs.
3. **Save as binary** (e.g. `uint16`) so training scripts can memory-map or load quickly: `train.bin`, `valid.bin`.

All training scripts (bigram, n-gram, transformer) expect:
- `data/processed/train.bin`
- `data/processed/valid.bin`
- `tokenizer/bpe_tokenizer.json`

---

## Summary

| Goal | Flow | Command |
|------|------|--------|
| Quick start, small data | Flow 1 | `python scripts/prepare_data.py` |
| Your own split + Wikipedia | Flow 2 | `python scripts/prepare_data_wiki.py` |

After either flow, you have **split data** (train vs valid) and **tokenized** data (`.bin` + tokenizer). You can then run e.g. `train_bigram.py`, `train_ngram.py`, or `train_transformer_causal.py`.
