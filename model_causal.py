"""
Shared GPT-style causal transformer used by train_transformer_causal.py,
generate_transformer.py, evaluate_models.py, and the API.
"""
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Per-layer KV cache: (key, value) both shaped (B, num_heads, seq_len, head_dim)
KVCache = Tuple[torch.Tensor, torch.Tensor]


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
        # Kept for checkpoint compatibility — not read in forward().
        # F.scaled_dot_product_attention handles the causal mask via is_causal=True.
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
        # Flash Attention: fused CUDA kernel, O(√T) memory vs O(T²) naive.
        dropout_p = self.drop.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.drop(out)
        return out

    def forward_cached(
        self, x: torch.Tensor, past_kv: Optional[KVCache]
    ) -> Tuple[torch.Tensor, KVCache]:
        """KV-cache variant for autoregressive decoding.

        x contains only the NEW token(s). Cached K/V from previous steps are
        prepended so each query can attend to the full context without
        recomputing K and V for every past token.

        Returns (attn_output, (updated_k, updated_v)).
        """
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)
        k = k.view(B, T, self.nh, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nh, self.hd).transpose(1, 2)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        # Prefill (no past): T_query == T_key → is_causal=True applies the
        # triangular mask correctly.
        # Decode (with past): T_query=1 < T_key — every cached key precedes
        # the current query by definition, so no mask is needed.
        is_causal = past_kv is None
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        return out, (k, v)


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

    def forward_cached(
        self, x: torch.Tensor, past_kv: Optional[KVCache]
    ) -> Tuple[torch.Tensor, KVCache]:
        attn_out, new_kv = self.attn.forward_cached(self.ln1(x), past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv


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
        _, T = idx.shape
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

    def _forward_cached(
        self,
        idx: torch.Tensor,
        past_kvs: Optional[List[KVCache]] = None,
    ) -> Tuple[torch.Tensor, List[KVCache]]:
        """Forward pass with per-layer KV cache.

        Position offsets are inferred from the cached key length so callers
        don't need to track step counts separately.

        Returns (logits, new_kvs) — pass new_kvs as past_kvs on the next step.
        """
        _, T = idx.shape
        past_len = past_kvs[0][0].shape[2] if past_kvs is not None else 0
        pos = torch.arange(past_len, past_len + T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        # Embedding dropout skipped at inference time (model already in eval()).

        new_kvs: List[KVCache] = []
        for i, blk in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, new_kv = blk.forward_cached(x, past_kv=past_kv)
            new_kvs.append(new_kv)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, new_kvs

    @staticmethod
    def _nucleus_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
        """Zero out tokens outside the top_p probability mass and renormalize.

        Works on any 1-D probability tensor (full vocab or a top-k subset).
        """
        sorted_probs, sort_order = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        # Remove tokens whose cumulative mass (excluding themselves) >= top_p.
        # This keeps the smallest set of tokens whose total probability >= top_p.
        remove = (cumsum - sorted_probs) >= top_p
        sorted_probs[remove] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum().clamp(min=1e-8)
        return sorted_probs[torch.argsort(sort_order)]

    def _sample_next(
        self,
        logits: torch.Tensor,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        repeat_window: int,
        generator: torch.Generator,
        idx: torch.Tensor,
    ) -> int:
        """Apply repetition penalty + top-k/top-p filtering, then sample one token."""
        # Repetition penalty: divide logits of recently-seen tokens by penalty.
        # This makes the model less likely to repeat the same words.
        # For positive logits → dividing reduces them.
        # For negative logits → dividing makes them more negative (less likely).
        if repetition_penalty != 1.0 and idx.shape[1] > 0:
            window = idx[0, -repeat_window:] if repeat_window > 0 else idx[0]
            for prev_id in window.unique().tolist():
                if logits[prev_id] > 0:
                    logits[prev_id] /= repetition_penalty
                else:
                    logits[prev_id] *= repetition_penalty

        if top_k and top_k > 0:
            # Restrict to top-k highest-scoring tokens, then optionally apply top-p
            # within that subset so we're sampling from a nucleus of the top-k.
            k = min(top_k, logits.size(-1))
            top_vals, top_idx = torch.topk(logits, k)
            probs = F.softmax(top_vals, dim=-1)
            if 0.0 < top_p < 1.0:
                probs = self._nucleus_filter(probs, top_p)
            chosen = torch.multinomial(probs, num_samples=1, generator=generator)
            return int(top_idx[chosen].item())
        else:
            # No top-k: sample from full vocab with optional nucleus filtering.
            probs = F.softmax(logits, dim=-1)
            if 0.0 < top_p < 1.0:
                probs = self._nucleus_filter(probs, top_p)
            return int(torch.multinomial(probs, num_samples=1, generator=generator).item())

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        seed: int = 42,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        repeat_window: int = 64,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive sample with KV-cache; returns token indices (1, T+new_tokens).

        KV-cache eliminates redundant K/V recomputation: after the prompt is
        processed once (prefill), each decoding step only runs one token through
        the network instead of the entire growing context.

        temperature: scales logits before softmax (higher = more random).
        top_k:       sample only from the k highest-probability tokens.
        top_p:       nucleus sampling — keep the smallest set of tokens whose
                     cumulative probability >= top_p, then sample from them.
        repetition_penalty: reduce probability of tokens already in the context.
        repeat_window: how many past tokens to check for repetition (0 = all).
        eos_id:      stop early when this token is generated.
        """
        assert idx.shape[0] == 1, "generate() only supports batch size 1"
        generator = torch.Generator(device=idx.device).manual_seed(seed)

        # Prefill: process entire prompt in one pass, populate KV cache.
        logits, past_kvs = self._forward_cached(idx)

        for _ in range(max_new_tokens):
            # Cast to float32 so softmax/multinomial have full precision even
            # when the model runs in float16.
            last_logit = logits[0, -1, :].float() / max(temperature, 1e-6)
            next_id = self._sample_next(
                last_logit, top_k, top_p, repetition_penalty, repeat_window, generator, idx
            )
            next_tok = torch.tensor([[next_id]], dtype=torch.long, device=idx.device)
            idx = torch.cat([idx, next_tok], dim=1)

            if eos_id is not None and next_id == eos_id:
                break
            if idx.shape[1] >= self.block_size:
                break

            # Decode: single-token forward, extend KV cache by one step.
            logits, past_kvs = self._forward_cached(next_tok, past_kvs=past_kvs)

        return idx

    @torch.no_grad()
    def generate_iter(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        seed: int = 42,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        repeat_window: int = 64,
        eos_id: Optional[int] = None,
    ) -> Iterator[int]:
        """KV-cache streaming variant — yields one token ID per step.

        Used by the streaming API endpoint so each decoded chunk is sent to
        the client as soon as it is produced.
        """
        assert idx.shape[0] == 1, "generate_iter() only supports batch size 1"
        generator = torch.Generator(device=idx.device).manual_seed(seed)

        logits, past_kvs = self._forward_cached(idx)

        for _ in range(max_new_tokens):
            last_logit = logits[0, -1, :].float() / max(temperature, 1e-6)
            next_id = self._sample_next(
                last_logit, top_k, top_p, repetition_penalty, repeat_window, generator, idx
            )
            next_tok = torch.tensor([[next_id]], dtype=torch.long, device=idx.device)
            idx = torch.cat([idx, next_tok], dim=1)
            yield next_id

            if eos_id is not None and next_id == eos_id:
                break
            if idx.shape[1] >= self.block_size:
                break

            logits, past_kvs = self._forward_cached(next_tok, past_kvs=past_kvs)

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
