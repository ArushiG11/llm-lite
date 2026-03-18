import argparse
import os
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

# Repo root for shared model
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from model_causal import GPTMini

DEVICE = "cpu"
@torch.no_grad()
def sample_next(logits, rng, temperature=0.9, top_k=50, top_p=0.9, repetition_penalty=1.0, recent_ids=None):
    logits = logits / max(temperature, 1e-6)

    # repetition penalty: downweight recently used tokens
    if repetition_penalty > 1.0 and recent_ids:
        logits = logits.clone()
        for tid in set(recent_ids):
            logits[tid] /= repetition_penalty

    # top-k
    if top_k and top_k > 0:
        v, ix = torch.topk(logits, k=top_k)
        logits = v
        idxs = ix
    else:
        idxs = torch.arange(logits.numel(), device=logits.device)

    probs = F.softmax(logits, dim=-1).cpu().numpy()

    # top-p (nucleus)
    if top_p and 0 < top_p < 1.0:
        order = np.argsort(-probs)
        sorted_probs = probs[order]
        cdf = np.cumsum(sorted_probs)
        keep = cdf <= top_p
        keep[0] = True
        order = order[keep]
        probs2 = probs[order]
        probs2 = probs2 / probs2.sum()
        chosen = int(rng.choice(order, p=probs2))
        return int(idxs[chosen].item()) if top_k else chosen

    chosen = int(rng.choice(len(probs), p=probs))
    return int(idxs[chosen].item()) if top_k else chosen

@torch.no_grad()
def generate(model, start_ids, max_new_tokens, seed, temperature, top_k, top_p, repetition_penalty, repeat_window):
    rng = np.random.default_rng(seed)
    ids = start_ids[:]
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-model.block_size:]], dtype=torch.long, device=DEVICE)
        logits = model(x)[0, -1, :]
        recent = ids[-repeat_window:] if repeat_window > 0 else []
        nxt = sample_next(logits, rng, temperature, top_k, top_p, repetition_penalty, recent_ids=recent)
        ids.append(nxt)
    return ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/transformer/ckpt.pt", help="Checkpoint from train_transformer_causal.py")
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    ap.add_argument("--repeat_window", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.isfile(args.ckpt):
        print(f"Checkpoint not found: {args.ckpt}")
        print("Train first: python scripts/train_transformer_causal.py")
        raise SystemExit(1)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]
    tok_path = ckpt.get("tokenizer_path", "tokenizer/bpe_tokenizer.json")
    tok = Tokenizer.from_file(tok_path)

    model = GPTMini.from_config(cfg, dropout_inference=0.0).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # encode prompt
    enc = tok.encode(args.prompt)
    ids = enc.ids
    if not ids:
        bos = tok.token_to_id("<bos>")
        ids = [bos if bos is not None else 0]

    out_ids = generate(
        model, ids, args.max_new_tokens, args.seed,
        args.temperature, args.top_k, args.top_p,
        args.repetition_penalty, args.repeat_window
    )
    text = tok.decode(out_ids)
    print(text)

if __name__ == "__main__":
    main()