"""
Wrap the Hugging Face `tokenizers` JSON tokenizer used during training.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


class TokenizerService:
    def __init__(self, tokenizer_path: Path):
        self.path = tokenizer_path.resolve()
        self._tok = Tokenizer.from_file(str(self.path))
        # ByteLevelDecoder is the inverse of ByteLevel pre-tokenizer:
        # it converts Ġ (U+0120, space prefix) back to a real space.
        # Without this, decoded text contains Ġhello instead of " hello".
        self._tok.decoder = ByteLevelDecoder()

    @property
    def tokenizer_version(self) -> str:
        """Stable-ish fingerprint from file mtime (no extra deps)."""
        try:
            mt = self.path.stat().st_mtime_ns
            return f"local:{self.path.name}@{mt:x}"
        except OSError:
            return f"local:{self.path.name}"

    def encode(self, text: str) -> List[int]:
        enc = self._tok.encode(text)
        ids = enc.ids
        if not ids:
            bos = self._tok.token_to_id("<bos>")
            ids = [bos if bos is not None else 0]
        return ids

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids)
