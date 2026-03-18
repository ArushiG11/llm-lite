import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from typing import List, Optional, Tuple

# Repo root for shared model (api/app/model.py -> api -> repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from model_causal import GPTMini

DEVICE = "cpu"
def _apply_repetition_penalty(logits: torch.Tensor, recent_ids: List[int], penalty: float):
    if penalty <= 1.0 or not recent_ids:
        return logits
    logits = logits.clone()
    for tid in set(recent_ids):
        logits[tid] = logits[tid] / penalty
    return logits

def _top_k_top_p_filter(logits: torch.Tensor, top_k: int, top_p: float) -> Tuple[torch.Tensor, torch.Tensor]:
    # returns (filtered_logits, corresponding_token_ids)
    V = logits.numel()
    token_ids = torch.arange(V, device=logits.device)

    if top_k and top_k > 0 and top_k < V:
        v, ix = torch.topk(logits, k=top_k)
        logits = v
        token_ids = token_ids[ix]

    probs = F.softmax(logits, dim=-1)

    if top_p and 0.0 < top_p < 1.0:
        sorted_probs, sorted_ix = torch.sort(probs, descending=True)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        keep = cdf <= top_p
        keep[0] = True
        keep_ix = sorted_ix[keep]
        logits = logits[keep_ix]
        token_ids = token_ids[keep_ix]

    return logits, token_ids


def _normalize_decoded(s: str) -> str:
    """Normalize tokenizer output so Ġ, âĢĵ, and similar don't appear in the UI."""
    # BPE space (U+2581) → normal space (both escape and literal in case of encoding quirks)
    s = s.replace("\u2581", " ")
    s = s.replace("Ġ", " ")
    s = s.replace("\N{LOWER ONE EIGHTH BLOCK}", " ")
    # En/em dash and similar (often show as âĢĵ when UTF-8 is mis-displayed) → hyphen
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    # Literal three-char sequence some tokenizers emit
    s = s.replace("âĢĵ", "")
    return s


class InferenceEngine:
    def __init__(self, ckpt_path: str):
        self.ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = self.ckpt["config"]
        self.tokenizer = Tokenizer.from_file(self.ckpt["tokenizer_path"])
        self.vocab_size = cfg["VOCAB_SIZE"]

        self.model = GPTMini.from_config(cfg, dropout_inference=0.0).to(DEVICE)
        self.model.load_state_dict(self.ckpt["state_dict"])
        self.model.eval()

        self.block_size = cfg["BLOCK_SIZE"]

    def encode(self, text: str) -> List[int]:
        ids = self.tokenizer.encode(text).ids
        if not ids:
            bos = self.tokenizer.token_to_id("<bos>")
            ids = [bos if bos is not None else 0]
        return ids

    def decode(self, ids: List[int]) -> str:
        s = self.tokenizer.decode(ids)
        # Normalize BPE/tokenizer output for clean display
        s = _normalize_decoded(s)
        return s

    @torch.no_grad()
    def generate_ids(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        repeat_window: int,
        seed: Optional[int],
    ):
        rng = np.random.default_rng(seed if seed is not None else 42)
        ids = prompt_ids[:]

        for _ in range(max_new_tokens):
            x = torch.tensor([ids[-self.block_size:]], dtype=torch.long, device=DEVICE)
            logits = self.model(x)[0, -1, :]  # (V,)

            logits = logits / max(temperature, 1e-6)
            recent = ids[-repeat_window:] if repeat_window > 0 else []
            logits = _apply_repetition_penalty(logits, recent, repetition_penalty)

            fl, token_ids = _top_k_top_p_filter(logits, top_k, top_p)
            probs = F.softmax(fl, dim=-1).cpu().numpy()
            chosen = int(rng.choice(len(probs), p=probs))
            next_id = int(token_ids[chosen].item())

            ids.append(next_id)

        return ids