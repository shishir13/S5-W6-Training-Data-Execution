"""Sequence packing — TEXT, CODE, INSTRUCTION, EVAL policies with correct masks."""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch

from tdes.tokenizer import FrozenTokenizer


class PackingPolicy(str, Enum):
    TEXT = "text"
    CODE = "code"
    INSTRUCTION = "instruction"
    EVAL = "eval"


POLICY_MAP: Dict[str, PackingPolicy] = {
    "english_web": PackingPolicy.TEXT,
    "math": PackingPolicy.TEXT,
    "code": PackingPolicy.CODE,
    "instruction": PackingPolicy.INSTRUCTION,
    "eval": PackingPolicy.EVAL,
}


@dataclass
class PackedSequence:
    input_ids: torch.Tensor       # [seq_len]
    labels: torch.Tensor          # [seq_len]  (-100 for ignored positions)
    attention_mask: torch.Tensor  # [seq_len]
    position_ids: torch.Tensor    # [seq_len]
    loss_mask: torch.Tensor       # [seq_len]  (1 = contributes to loss)
    lane: str
    shard_id: str
    token_span_start: int
    token_span_end: int
    policy: PackingPolicy


def _instruction_split(tokens: List[int], response_start_token: int = 14974) -> int:
    """Find the index where '### Response' begins.

    Falls back to 50% split if marker not found.
    The token 14974 is '###' in GPT-2 — we look for the second occurrence.
    """
    count = 0
    for i, t in enumerate(tokens):
        if t == response_start_token:
            count += 1
            if count == 2:
                return i
    return len(tokens) // 2


def pack_sequence(
    tokens: List[int],
    shard_id: str,
    lane: str,
    token_span_start: int,
    seq_len: int,
    tokenizer: FrozenTokenizer,
    is_eval: bool = False,
) -> PackedSequence:
    """Pack tokens into a fixed-length sequence with correct masks."""
    policy = PackingPolicy.EVAL if is_eval else POLICY_MAP.get(lane, PackingPolicy.TEXT)

    # Truncate to seq_len
    tokens = tokens[:seq_len]
    token_span_end = token_span_start + len(tokens)
    pad_len = seq_len - len(tokens)

    # Build input_ids with padding
    padded = tokens + [tokenizer.PAD_ID] * pad_len
    input_ids = torch.tensor(padded, dtype=torch.long)

    # Labels: shift left by 1 (next-token prediction), -100 for pad and last position
    label_tokens = tokens[1:] + [tokenizer.EOS_ID]
    label_padded = label_tokens + [-100] * pad_len
    labels = torch.tensor(label_padded, dtype=torch.long)

    # Attention mask: 1 for real tokens
    attention_mask = torch.zeros(seq_len, dtype=torch.long)
    attention_mask[: len(tokens)] = 1

    # Position IDs: reset to 0 after each EOS token
    position_ids = _build_position_ids(tokens, tokenizer.EOS_ID, seq_len)

    # Loss mask
    loss_mask = _build_loss_mask(tokens, policy, seq_len, tokenizer.EOS_ID)

    return PackedSequence(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        position_ids=position_ids,
        loss_mask=loss_mask,
        lane=lane,
        shard_id=shard_id,
        token_span_start=token_span_start,
        token_span_end=token_span_end,
        policy=policy,
    )


def _build_position_ids(
    tokens: List[int], eos_id: int, seq_len: int
) -> torch.Tensor:
    """Position IDs that reset to 0 at each EOS (document boundary), then pad with 0."""
    pos = 0
    result: List[int] = []
    for tok in tokens:
        result.append(pos)
        if tok == eos_id:
            pos = 0
        else:
            pos += 1
    # pad remainder with 0
    result += [0] * (seq_len - len(tokens))
    return torch.tensor(result, dtype=torch.long)


def _build_loss_mask(
    tokens: List[int], policy: PackingPolicy, seq_len: int, eos_id: int
) -> torch.Tensor:
    mask = torch.zeros(seq_len, dtype=torch.float)

    if policy == PackingPolicy.EVAL:
        # Firewall: no loss ever flows from eval tokens
        return mask

    if policy in (PackingPolicy.TEXT, PackingPolicy.CODE):
        mask[: len(tokens)] = 1.0
        return mask

    if policy == PackingPolicy.INSTRUCTION:
        # Prompt tokens get loss_mask=0; response tokens get 1
        split = _instruction_split(tokens)
        mask[split: len(tokens)] = 1.0
        # Ensure at least the last token has loss if split is at the very end
        if mask[: len(tokens)].sum() == 0 and len(tokens) > 0:
            mask[max(0, len(tokens) - 1)] = 1.0
        return mask

    return mask


@dataclass
class BatchRecord:
    sequences: List[PackedSequence]
    batch_id: str
    step: int

    # Stacked tensors (shape [B, seq_len])
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    loss_mask: torch.Tensor


def assemble_batch(
    sequences: List[PackedSequence],
    step: int,
) -> BatchRecord:
    """Stack sequences and compute deterministic batch_id."""
    input_ids = torch.stack([s.input_ids for s in sequences])
    labels = torch.stack([s.labels for s in sequences])
    attention_mask = torch.stack([s.attention_mask for s in sequences])
    position_ids = torch.stack([s.position_ids for s in sequences])
    loss_mask = torch.stack([s.loss_mask for s in sequences])

    # batch_id = sha256(step_bytes || token_bytes) — pure function of content
    token_bytes = input_ids.numpy().tobytes()
    payload = step.to_bytes(8, "big") + token_bytes
    batch_id = hashlib.sha256(payload).hexdigest()[:16]

    return BatchRecord(
        sequences=sequences,
        batch_id=batch_id,
        step=step,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        position_ids=position_ids,
        loss_mask=loss_mask,
    )


def packing_utilization(sequences: List[PackedSequence], seq_len: int) -> float:
    """Fraction of non-pad tokens across all sequences."""
    total = len(sequences) * seq_len
    if total == 0:
        return 0.0
    used = sum(int(s.attention_mask.sum().item()) for s in sequences)
    return used / total
