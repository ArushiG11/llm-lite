"""
Load a local PyTorch checkpoint produced by train_transformer_causal.py and build GPTMini.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch

from ..core.config import Settings, repo_root

# Repo root on sys.path for `model_causal` (lives at project root).
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model_causal import GPTMini  # noqa: E402


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Model + metadata from disk."""

    model: GPTMini
    config: Dict[str, Any]
    checkpoint_path: Path
    tokenizer_path: Path
    training_step: int | None
    best_val_loss: float | None


def load_checkpoint(settings: Settings) -> LoadedCheckpoint:
    ckpt_path = settings.resolved_checkpoint_path()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=settings.device)
    if "state_dict" not in ckpt or "config" not in ckpt:
        raise ValueError(f"Invalid checkpoint keys in {ckpt_path}; expected state_dict and config.")

    cfg = ckpt["config"]
    tok_raw = ckpt.get("tokenizer_path") or "tokenizer/bpe_tokenizer.json"
    tokenizer_path = Path(tok_raw)
    if tokenizer_path.is_file():
        tokenizer_path = tokenizer_path.resolve()
    elif not tokenizer_path.is_absolute():
        for base in (repo_root(), Path.cwd()):
            candidate = (base / tok_raw).resolve()
            if candidate.is_file():
                tokenizer_path = candidate
                break
        else:
            tokenizer_path = (repo_root() / tok_raw).resolve()

    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {tok_raw} (resolved: {tokenizer_path})")

    model = GPTMini.from_config(cfg, dropout_inference=0.0).to(settings.device)
    model.load_state_dict(ckpt["state_dict"])
    # Half-precision halves VRAM and speeds up compute on CUDA/MPS.
    # Stay in float32 on CPU — float16 ops are unoptimized there.
    if settings.device not in ("cpu",):
        model = model.half()
    model.eval()

    return LoadedCheckpoint(
        model=model,
        config=dict(cfg),
        checkpoint_path=ckpt_path.resolve(),
        tokenizer_path=tokenizer_path.resolve(),
        training_step=ckpt.get("step"),
        best_val_loss=ckpt.get("best_val_loss"),
    )
