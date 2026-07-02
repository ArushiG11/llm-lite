"""Pydantic request/response models for the inference API."""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


# --- Health / readiness ---


class HealthResponse(BaseModel):
    status: str = Field(description="Liveness indicator")


class ReadyResponse(BaseModel):
    ready: bool
    detail: Optional[str] = None


# --- Model info ---


class ModelLimits(BaseModel):
    max_prompt_tokens: int
    max_new_tokens_upper: int
    min_temperature: float
    max_temperature: float


class ModelInfoResponse(BaseModel):
    model_version: str
    checkpoint_path: str
    tokenizer_path: str
    tokenizer_version: str
    device: str
    vocab_size: Optional[int] = None
    block_size: Optional[int] = None
    embed_dim: Optional[int] = None
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    training_step: Optional[int] = None
    best_val_loss: Optional[float] = None
    limits: ModelLimits


# --- Generate ---


class GenerateRequest(BaseModel):
    prompt: str = Field(default="", description="Input text (encoded with the trained BPE tokenizer)")
    max_new_tokens: int = Field(default=64, ge=1, description="Number of new tokens to sample")
    temperature: float = Field(
        default=0.9,
        ge=0.0,
        description="Sampling temperature (validated against server limits at runtime)",
    )
    top_k: int = Field(default=50, ge=0, description="Top-k filtering (0 disables)")
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling: keep smallest set of tokens with cumulative prob >= top_p (1.0 disables)",
    )
    repetition_penalty: float = Field(
        default=1.0,
        ge=1.0,
        description="Divide logits of recently-seen tokens by this factor to reduce repetition (1.0 disables)",
    )
    repeat_window: int = Field(
        default=64,
        ge=0,
        description="Number of recent tokens to check for repetition penalty (0 = full context)",
    )
    seed: Optional[int] = Field(default=None, description="RNG seed for reproducibility")


class GenerateMetadata(BaseModel):
    model_version: str
    tokenizer_path: str
    tokenizer_version: str
    latency_ms: float
    prompt_token_count: int
    context_token_count: int
    new_tokens_generated: int


class GenerateResponse(BaseModel):
    text: str
    metadata: GenerateMetadata


# --- Tokenize / detokenize ---


class TokenizeRequest(BaseModel):
    text: str = Field(default="", description="Raw text to encode")


class TokenizeResponse(BaseModel):
    ids: List[int]
    token_count: int
    latency_ms: float
    tokenizer_path: str
    tokenizer_version: str


class DetokenizeRequest(BaseModel):
    ids: List[int] = Field(description="Token ids to decode")


class DetokenizeResponse(BaseModel):
    text: str
    token_count: int
    latency_ms: float
    tokenizer_path: str
    tokenizer_version: str


class ErrorResponse(BaseModel):
    """Consistent error body for HTTP errors."""

    error: str
    code: str
    detail: Optional[Any] = None
