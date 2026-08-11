"""Frozen tokenizer — GPT-2 BPE with a locked SHA-256 hash."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

from transformers import GPT2TokenizerFast  # type: ignore[import-untyped]


class TokenizerMismatch(RuntimeError):
    pass


class FrozenTokenizer:
    """Wraps GPT-2 tokenizer and computes a stable hash for integrity checks."""

    PAD_ID: int = 0
    EOS_ID: int = 50256  # GPT-2 <|endoftext|>

    def __init__(self, tok: GPT2TokenizerFast, tok_hash: str) -> None:
        self._tok = tok
        self.hash = tok_hash
        self.vocab_size: int = tok.vocab_size

    @classmethod
    def load(cls) -> "FrozenTokenizer":
        tok: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        tok_hash = cls._compute_hash(tok)
        return cls(tok, tok_hash)

    @staticmethod
    def _compute_hash(tok: GPT2TokenizerFast) -> str:
        h = hashlib.sha256()
        # Hash the vocab (sorted for stability)
        vocab = tok.get_vocab()
        for token, idx in sorted(vocab.items()):
            h.update(f"{token}:{idx}".encode())
        return h.hexdigest()

    def encode(self, text: str) -> List[int]:
        ids: List[int] = self._tok.encode(text, add_special_tokens=False)
        return ids

    def verify_hash(self, expected_hash: str) -> None:
        if self.hash != expected_hash:
            raise TokenizerMismatch(
                f"Tokenizer hash mismatch: expected {expected_hash}, got {self.hash}"
            )
