# llm-lite

Build a language model from scratch: **bigram → n-gram → transformer**, on a single machine with minimal dependencies.

## What’s in this repo

| Step | Model | Script | Idea |
|------|--------|--------|------|
| 1 | **Bigram** | `scripts/train_bigram.py` | P(next \| prev); count-based, add-α smoothing. |
| 2 | **N-gram** | `scripts/train_ngram.py` | Configurable N (2,3,4,5,…) with interpolated backoff. See [docs/NGRAM.md](docs/NGRAM.md). |
| 3 | **Trigram (interpolated)** | `scripts/train_trigram_interpolated.py` | Trigram + bigram + unigram with λ weights. |
| 4 | **Transformer (encoder)** | `scripts/train_transformer.py` | PyTorch `TransformerEncoder` (bidirectional). |
| 5 | **Transformer (causal)** | `scripts/train_transformer_causal.py` | GPT-style: causal self-attention, MLP blocks, top-k sampling. |

You also have: sparse bigram (`train_bigram_sparse.py`), tokenizer training, and data prep (WikiText-2, BPE, tokenize-to-bin).

## Quick start

```bash
# 1. Data + tokenizer (Flow 2: Wikipedia dump → train/valid .bin + BPE)
python scripts/prepare_data_wiki.py

# 2. Train models (run from repo root)
python scripts/train_bigram.py
python scripts/train_ngram.py --n 4    # 4-gram with interpolation
python scripts/train_trigram_interpolated.py
python scripts/train_transformer.py     # small encoder (CPU-friendly)
python scripts/train_transformer_causal.py   # causal GPT-style
```

Generated artifacts: `data/processed/*.bin`, `tokenizer/bpe_tokenizer.json`, `models/*.npy`, `models/*.pkl`.

## Data preparation (splitting and tokenizing)

Two options; both produce `data/processed/train.bin`, `valid.bin`, and `tokenizer/bpe_tokenizer.json`:

- **Flow 1 — WikiText-2:** `python scripts/prepare_data.py`
- **Flow 2 — Wikipedia dump (one script):**  
  `pip install -r requirements.txt` then `python scripts/prepare_data_wiki.py`  
  (Use the same venv so steps 4–5 see `tokenizers` and `numpy`.)  
  See **[docs/DATA_PREP_FLOW.md](docs/DATA_PREP_FLOW.md)** for details.

## Pipeline

1. **Data**: Either Flow 1 or Flow 2 → `data/processed/train.txt`, `valid.txt`.
2. **Tokenizer**: BPE (e.g. 8k vocab) on `train.txt` only → `tokenizer/bpe_tokenizer.json`.
3. **Tokenize to binary**: Encode train and valid → `train.bin`, `valid.bin` (uint16 token IDs).
4. **Train**: bigram → n-gram → transformer (see table above).

## What could we do better?

- **More context**: N-grams are limited to a fixed window; transformers give long-range context with attention.
- **Causal vs bidirectional**: For autoregressive text generation use the **causal** transformer; the encoder script is for understanding the API / bidirectional setups.
- **Scaling**: Larger `BLOCK_SIZE`, more layers/heads, more data, LR schedule (cosine, warmup).
- **Training**: Checkpointing, validation-based early stopping, gradient clipping.
- **Sampling**: You already have top-k; add top-p (nucleus), temperature tuning, repetition penalty.
- **Tokenization**: Larger BPE vocab, sentencepiece, or train on your target domain.
- **Architecture**: Try a small **RNN/LSTM** as a middle step between n-grams and transformers; then **attention variants** (e.g. grouped-query, RoPE).
- **Evaluation**: Track perplexity and a few fixed prompts; add a simple downstream task (e.g. next-sentence or classification) if you want.

See `ROADMAP.md` for a concrete ordered list of next steps.
