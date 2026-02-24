# N-gram model

The n-gram script is the next step after **bigram**: same count-based idea, but with **longer context** (N−1 previous tokens) and **interpolation** so you don’t need to run separate bigram/trigram scripts for each order.

## What it does

- **Trains** counts for 1-gram, 2-gram, … N-gram from `data/processed/train.bin`.
- **Combines** them with **interpolation**:  
  `P(w | context) = λ_N · P_N(w|context) + … + λ_1 · P_1(w)`  
  with add-α smoothing at each level.
- **Evaluates** validation perplexity and **generates** a sample.
- **Saves** the model as `models/ngram_n{N}_interpolated.pkl`.

## How to run

From repo root (after [data prep](DATA_PREP_FLOW.md)):

```bash
# Trigram (N=3), similar to train_trigram_interpolated.py
python scripts/train_ngram.py --n 3

# 4-gram (default)
python scripts/train_ngram.py --n 4

# 5-gram (needs more data; use more train tokens or increase --max-tokens)
python scripts/train_ngram.py --n 5
```

Options:

| Option | Default | Meaning |
|--------|--------|--------|
| `--n` | 4 | N-gram order (2=bigram, 3=trigram, …). |
| `--alpha` | 0.1 | Add-α smoothing. |
| `--max-tokens` | 5_000_000 | Cap training tokens (memory/speed). |
| `--train-bin` | data/processed/train.bin | Training token IDs. |
| `--valid-bin` | data/processed/valid.bin | Validation token IDs. |
| `--tokenizer` | tokenizer/bpe_tokenizer.json | Tokenizer for decode. |
| `--out-dir` | models | Where to save the `.pkl` model. |

## Bigram vs N-gram vs trigram script

| Script | Context | Storage | Use case |
|--------|--------|--------|----------|
| `train_bigram.py` | 1 token (prev) | Dense V×V array | Fast, small vocab. |
| `train_ngram.py --n 3` | 2 tokens | Sparse (Counter) | Same idea as trigram, one script. |
| `train_trigram_interpolated.py` | 2 tokens | Sparse (Counter) | Fixed trigram + interpolation. |
| `train_ngram.py --n 4` | 3 tokens | Sparse | 4-gram with interpolation. |

For N=2, `train_ngram.py` is an interpolated bigram+unigram; for N=3 it matches the idea of `train_trigram_interpolated.py`. Use `train_ngram.py` when you want one script for any N and to try 4-gram, 5-gram, etc.

## Output

- **Console:** Vocab size, token counts, per-level unique n-gram counts, lambdas, validation NLL and perplexity, and a short generated sample.
- **File:** `models/ngram_n{N}_interpolated.pkl` (counts, ctx, total, V, alpha, lambdas, followers for generation).

## Next step

After n-gram, move on to the **transformer** (causal LM for generation):

```bash
python scripts/train_transformer_causal.py
```
