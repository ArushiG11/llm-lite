import argparse
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import math

DEVICE = "cpu"

# ---- Model code must match training ----
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.nh = num_heads
        self.hd = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )
        self.block_size = block_size

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)
        k = k.view(B, T, self.nh, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nh, self.hd).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.drop(out)
        return out

class MLP(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTMini(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, block_size, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

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

    model = GPTMini(
        vocab_size=cfg["VOCAB_SIZE"],
        embed_dim=cfg["EMBED_DIM"],
        num_heads=cfg["NUM_HEADS"],
        num_layers=cfg["NUM_LAYERS"],
        block_size=cfg["BLOCK_SIZE"],
        dropout=0.0,  # no dropout at inference
    ).to(DEVICE)
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