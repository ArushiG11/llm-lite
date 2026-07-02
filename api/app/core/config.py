"""
Application settings (pydantic-settings). Override via environment variables with prefix LLMLITE_.
Example: LLMLITE_CHECKPOINT_PATH=models/transformer/ckpt.pt
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def repo_root() -> Path:
    """api/app/core/config.py -> repo root (llm-lite)."""
    return Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LLMLITE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Logical version string for API responses (not the checkpoint file).
    model_version: str = "1.0.0"

    # If unset, resolves to repo_root()/models/transformer/ckpt.pt with optional ckpt_best.pt preference.
    checkpoint_path: Optional[Path] = Field(
        default=None,
        description="Explicit checkpoint file. If None, use default path resolution.",
    )
    prefer_best_checkpoint: bool = Field(
        default=True,
        description="When checkpoint_path is None, prefer ckpt_best.pt if it exists next to ckpt.pt.",
    )

    device: str = Field(default="cpu", description="torch device for inference")

    max_prompt_tokens: int = Field(default=2048, ge=1, le=100_000)
    max_new_tokens_upper: int = Field(
        default=512,
        ge=1,
        le=8192,
        description="Hard upper bound for max_new_tokens in requests.",
    )
    min_temperature: float = Field(default=0.0, ge=0.0)
    max_temperature: float = Field(default=2.0, le=10.0)

    # Rough guardrail for raw string length on tokenize/detokenize (characters / ids).
    max_tokenize_chars: int = Field(default=200_000, ge=1)
    max_detokenize_ids: int = Field(default=50_000, ge=1)

    def resolved_checkpoint_path(self) -> Path:
        """Absolute path to the checkpoint file to load."""
        if self.checkpoint_path is not None:
            p = self.checkpoint_path
            return p if p.is_absolute() else repo_root() / p

        base = repo_root() / "models" / "transformer" / "ckpt.pt"
        if self.prefer_best_checkpoint:
            best = base.parent / "ckpt_best.pt"
            if best.is_file():
                return best.resolve()
        return base.resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
