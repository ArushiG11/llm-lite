"""
Shared GPT-style causal transformer used by train_transformer_causal.py,
generate_transformer.py, evaluate_models.py, and the API.
"""
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_default_config() -> Dict[str, Any]:
    """Default training/inference config. Override via CLI or checkpoint."""
    return {
        "BATCH_SIZE": 8,
        "BLOCK_SIZE": 256,
        "EMBED_DIM": 192,
        "NUM_HEADS": 6,
        "NUM_LAYERS": 3,
        "DROPOUT": 0.1,
        "VOCAB_SIZE": None,  # set from tokenizer
    }


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, block_size: int, dropout: float):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.nh = num_heads
        self.hd = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTMini(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        block_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, block_size, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        seed: int = 42,
        temperature: float = 0.9,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive sample; returns token indices (B, T+max_new_tokens)."""
        rng = np.random.default_rng(seed)
        vocab_size = self.head.out_features
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond, targets=None)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            logits = logits.squeeze(0)

            if top_k and top_k > 0:
                v, ix = torch.topk(logits, k=min(top_k, logits.size(-1)))
                probs = F.softmax(v, dim=-1).cpu().numpy()
                next_id = int(rng.choice(ix.cpu().numpy(), p=probs))
            else:
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                next_id = int(rng.choice(vocab_size, p=probs))

            idx = torch.cat(
                [idx, torch.tensor([[next_id]], dtype=torch.long, device=idx.device)],
                dim=1,
            )
        return idx

    @staticmethod
    def from_config(config: Dict[str, Any], dropout_inference: float = 0.0) -> "GPTMini":
        return GPTMini(
            vocab_size=config["VOCAB_SIZE"],
            embed_dim=config["EMBED_DIM"],
            num_heads=config["NUM_HEADS"],
            num_layers=config["NUM_LAYERS"],
            block_size=config["BLOCK_SIZE"],
            dropout=dropout_inference,
        )
