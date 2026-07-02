"""
Orchestrates tokenizer + model.generate() with validation and timing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generator, List, Optional

import torch

from ..core.config import Settings
from .model_loader import LoadedCheckpoint, load_checkpoint
from .tokenizer_service import TokenizerService


class InferenceError(Exception):
    """User-facing inference errors (mapped to 4xx)."""

    def __init__(self, message: str, code: str = "inference_error"):
        super().__init__(message)
        self.code = code


class InferenceNotReadyError(Exception):
    """Raised when the model was not loaded at startup."""

    def __init__(self, message: str = "Model and tokenizer are not loaded"):
        super().__init__(message)
        self.message = message


@dataclass
class GenerateResult:
    text: str
    latency_ms: float
    prompt_token_count: int
    context_token_count: int
    new_tokens_generated: int
    model_version: str
    tokenizer_path: str
    tokenizer_version: str


@dataclass
class TokenizeResult:
    ids: List[int]
    latency_ms: float
    token_count: int
    tokenizer_path: str
    tokenizer_version: str


@dataclass
class DetokenizeResult:
    text: str
    latency_ms: float
    token_count: int
    tokenizer_path: str
    tokenizer_version: str


class InferenceService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._loaded: Optional[LoadedCheckpoint] = None
        self._tokenizer: Optional[TokenizerService] = None
        self._load_error: Optional[str] = None

        try:
            loaded = load_checkpoint(settings)
            self._loaded = loaded
            self._tokenizer = TokenizerService(loaded.tokenizer_path)
        except Exception as e:  # noqa: BLE001 — startup must not crash the process
            self._load_error = f"{type(e).__name__}: {e}"

    @property
    def is_ready(self) -> bool:
        return self._loaded is not None and self._tokenizer is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _require_ready(self) -> tuple[LoadedCheckpoint, TokenizerService]:
        if not self.is_ready or self._loaded is None or self._tokenizer is None:
            raise InferenceNotReadyError(self._load_error or "Model and tokenizer are not loaded.")
        return self._loaded, self._tokenizer

    def model_info(self) -> dict:
        if not self.is_ready or self._loaded is None or self._tokenizer is None:
            raise InferenceNotReadyError(self._load_error or "Not loaded.")
        lc = self._loaded
        cfg = lc.config
        return {
            "model_version": self.settings.model_version,
            "checkpoint_path": str(lc.checkpoint_path),
            "tokenizer_path": str(lc.tokenizer_path),
            "tokenizer_version": self._tokenizer.tokenizer_version,
            "device": self.settings.device,
            "vocab_size": cfg.get("VOCAB_SIZE"),
            "block_size": cfg.get("BLOCK_SIZE"),
            "embed_dim": cfg.get("EMBED_DIM"),
            "num_layers": cfg.get("NUM_LAYERS"),
            "num_heads": cfg.get("NUM_HEADS"),
            "training_step": lc.training_step,
            "best_val_loss": lc.best_val_loss,
            "limits": {
                "max_prompt_tokens": self.settings.max_prompt_tokens,
                "max_new_tokens_upper": self.settings.max_new_tokens_upper,
                "min_temperature": self.settings.min_temperature,
                "max_temperature": self.settings.max_temperature,
            },
        }

    def validate_generation_params(self, max_new_tokens: int, temperature: float) -> None:
        if max_new_tokens < 1:
            raise InferenceError("max_new_tokens must be >= 1", "invalid_max_new_tokens")
        if max_new_tokens > self.settings.max_new_tokens_upper:
            raise InferenceError(
                f"max_new_tokens must be <= {self.settings.max_new_tokens_upper}",
                "max_new_tokens_exceeded",
            )
        if temperature < self.settings.min_temperature or temperature > self.settings.max_temperature:
            raise InferenceError(
                f"temperature must be between {self.settings.min_temperature} and {self.settings.max_temperature}",
                "invalid_temperature",
            )

    def _prepare_context(
        self, prompt: str, loaded: LoadedCheckpoint, tok: TokenizerService
    ) -> tuple[List[int], torch.Tensor]:
        """Encode prompt and clip to block_size; raise if over token limit."""
        prompt_ids = tok.encode(prompt)
        if len(prompt_ids) > self.settings.max_prompt_tokens:
            raise InferenceError(
                f"Prompt encodes to {len(prompt_ids)} tokens; limit is {self.settings.max_prompt_tokens}",
                "prompt_too_long",
            )
        block = int(loaded.config["BLOCK_SIZE"])
        context = prompt_ids[-block:]
        idx = torch.tensor([context], dtype=torch.long, device=self.settings.device)
        return prompt_ids, idx

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        repeat_window: int,
        seed: Optional[int],
    ) -> GenerateResult:
        self.validate_generation_params(max_new_tokens, temperature)
        loaded, tok = self._require_ready()

        t0 = time.perf_counter()
        prompt_ids, idx = self._prepare_context(prompt, loaded, tok)
        context_token_count = idx.shape[1]

        eos_id = tok._tok.token_to_id("<eos>")
        gen_seed = seed if seed is not None else 42

        out = loaded.model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            seed=gen_seed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            repeat_window=repeat_window,
            eos_id=eos_id,
        )
        full_ids = out[0].tolist()
        text = tok.decode(full_ids)
        new_tokens_generated = int(out.shape[1] - idx.shape[1])
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return GenerateResult(
            text=text,
            latency_ms=latency_ms,
            prompt_token_count=len(prompt_ids),
            context_token_count=context_token_count,
            new_tokens_generated=new_tokens_generated,
            model_version=self.settings.model_version,
            tokenizer_path=str(tok.path),
            tokenizer_version=tok.tokenizer_version,
        )

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        repeat_window: int,
        seed: Optional[int],
    ) -> Generator[str, None, None]:
        """Yield decoded text chunks one token at a time for SSE streaming."""
        self.validate_generation_params(max_new_tokens, temperature)
        loaded, tok = self._require_ready()

        _, idx = self._prepare_context(prompt, loaded, tok)
        eos_id = tok._tok.token_to_id("<eos>")
        gen_seed = seed if seed is not None else 42

        for next_id in loaded.model.generate_iter(
            idx,
            max_new_tokens=max_new_tokens,
            seed=gen_seed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            repeat_window=repeat_window,
            eos_id=eos_id,
        ):
            # Decode each token individually. ByteLevelDecoder converts Ġ→space
            # at the single-token level, so streaming produces clean text.
            chunk = tok.decode([next_id])
            if chunk:
                yield chunk

    def tokenize(self, text: str) -> TokenizeResult:
        _, tok = self._require_ready()
        if len(text) > self.settings.max_tokenize_chars:
            raise InferenceError(
                f"Text length {len(text)} exceeds max_tokenize_chars={self.settings.max_tokenize_chars}",
                "text_too_long",
            )
        t0 = time.perf_counter()
        ids = tok.encode(text)
        if len(ids) > self.settings.max_prompt_tokens:
            raise InferenceError(
                f"Encoded length {len(ids)} exceeds max_prompt_tokens={self.settings.max_prompt_tokens}",
                "too_many_tokens",
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return TokenizeResult(
            ids=ids,
            latency_ms=latency_ms,
            token_count=len(ids),
            tokenizer_path=str(tok.path),
            tokenizer_version=tok.tokenizer_version,
        )

    def detokenize(self, ids: List[int]) -> DetokenizeResult:
        _, tok = self._require_ready()
        if not ids:
            raise InferenceError("ids must be non-empty", "empty_ids")
        if len(ids) > self.settings.max_detokenize_ids:
            raise InferenceError(
                f"ids length {len(ids)} exceeds max_detokenize_ids={self.settings.max_detokenize_ids}",
                "too_many_ids",
            )
        t0 = time.perf_counter()
        text = tok.decode(ids)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return DetokenizeResult(
            text=text,
            latency_ms=latency_ms,
            token_count=len(ids),
            tokenizer_path=str(tok.path),
            tokenizer_version=tok.tokenizer_version,
        )
