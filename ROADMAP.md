# llm-lite roadmap

Ordered ideas to go from “working” to “better” without changing the core bigram → n-gram → transformer path.

## Done in this repo

- [x] Bigram (count-based, smoothed)
- [x] Trigram with interpolation
- [x] Generic N-gram (any N, interpolated)
- [x] Transformer (encoder, for learning)
- [x] Causal transformer (GPT-style, for generation)
- [x] BPE tokenizer, WikiText-2, tokenize-to-bin pipeline

## Next steps (in order)

1. **Run and compare**  
   Train bigram, n-gram (e.g. `--n 4`), and causal transformer on the same data; compare validation perplexity and sample quality.

2. **Checkpointing**  
   Save best model by validation loss; resume from checkpoint (optional).

3. **LR schedule**  
   Add warmup + cosine (or linear) decay in the transformer training loop.

4. **Small RNN/LSTM**  
   Add a script that trains a 1–2 layer LSTM LM on the same `.bin` data as a bridge between n-grams and transformers.

5. **Sampling**  
   Add top-p (nucleus) and optional repetition penalty to the causal transformer’s `generate()`.

6. **Larger runs**  
   Increase `BLOCK_SIZE`, layers, heads, and data (e.g. more of WikiText or another corpus) when you have the compute.

7. **Tokenizer**  
   Experiment with vocab size (e.g. 16k) or train on domain-specific data.

8. **Position encoding**  
   Try learned vs sinusoidal; later, RoPE or ALiBi if you go beyond toy size.

9. **Eval**  
    Log 1–2 fixed prompts every N steps to see qualitative progress.
