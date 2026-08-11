"""Tests for packing policies, masks, and position IDs."""
from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import pytest

from tdes.packing import (
    pack_sequence, assemble_batch, packing_utilization,
    PackingPolicy, POLICY_MAP,
)
from tdes.tokenizer import FrozenTokenizer


SEQ_LEN = 64  # small for tests


@pytest.fixture(scope="module")
def tokenizer() -> FrozenTokenizer:
    return FrozenTokenizer.load()


def _make_tokens(n: int, eos_id: int, with_eos: bool = True) -> List[int]:
    tokens = list(range(100, 100 + n))
    if with_eos:
        tokens.append(eos_id)
    return tokens


def test_attention_mask_matches_real_tokens(tokenizer: FrozenTokenizer) -> None:
    tokens = _make_tokens(20, tokenizer.EOS_ID)
    seq = pack_sequence(tokens, "s0", "english_web", 0, SEQ_LEN, tokenizer)
    # Padding positions must have input_ids == PAD_ID
    for i in range(SEQ_LEN):
        if seq.attention_mask[i] == 0:
            assert seq.input_ids[i] == tokenizer.PAD_ID


def test_position_ids_reset_at_eos(tokenizer: FrozenTokenizer) -> None:
    # Build two short docs separated by EOS
    doc1 = list(range(200, 210)) + [tokenizer.EOS_ID]
    doc2 = list(range(300, 308))
    tokens = doc1 + doc2
    seq = pack_sequence(tokens, "s0", "english_web", 0, SEQ_LEN, tokenizer)
    pos = seq.position_ids.tolist()
    # After EOS (position len(doc1)-1), next position must be 0
    eos_pos = len(doc1) - 1
    if eos_pos + 1 < SEQ_LEN:
        assert pos[eos_pos + 1] == 0, f"Expected 0 after EOS, got {pos[eos_pos + 1]}"


def test_loss_mask_text_all_real_tokens(tokenizer: FrozenTokenizer) -> None:
    tokens = _make_tokens(30, tokenizer.EOS_ID)
    seq = pack_sequence(tokens, "s0", "english_web", 0, SEQ_LEN, tokenizer)
    assert seq.policy == PackingPolicy.TEXT
    # All real (non-pad) tokens should have loss_mask=1
    for i in range(len(tokens)):
        assert seq.loss_mask[i].item() == 1.0


def test_loss_mask_eval_all_zero(tokenizer: FrozenTokenizer) -> None:
    tokens = _make_tokens(20, tokenizer.EOS_ID)
    seq = pack_sequence(tokens, "s0", "eval", 0, SEQ_LEN, tokenizer, is_eval=True)
    assert seq.policy == PackingPolicy.EVAL
    assert seq.loss_mask.sum().item() == 0.0


def test_loss_mask_instruction_prompt_excluded(tokenizer: FrozenTokenizer) -> None:
    # Use actual instruction text so split fires correctly
    text = (
        "### Instruction\nWhat is 2+2?\n"
        "### Response\nThe answer is 4."
    )
    tokens = tokenizer.encode(text)
    seq = pack_sequence(tokens, "s0", "instruction", 0, SEQ_LEN, tokenizer)
    assert seq.policy == PackingPolicy.INSTRUCTION
    # At least some tokens should have loss_mask=0 (prompt side)
    assert seq.loss_mask[: len(tokens)].sum().item() < len(tokens)


def test_labels_shifted_correctly(tokenizer: FrozenTokenizer) -> None:
    tokens = _make_tokens(20, tokenizer.EOS_ID, with_eos=False)
    seq = pack_sequence(tokens, "s0", "english_web", 0, SEQ_LEN, tokenizer)
    # labels[i] == input_ids[i+1] for i < len(tokens)-1
    for i in range(len(tokens) - 1):
        assert seq.labels[i].item() == seq.input_ids[i + 1].item()


def test_padding_labels_are_minus_100(tokenizer: FrozenTokenizer) -> None:
    tokens = _make_tokens(10, tokenizer.EOS_ID)
    seq = pack_sequence(tokens, "s0", "english_web", 0, SEQ_LEN, tokenizer)
    for i in range(len(tokens), SEQ_LEN):
        assert seq.labels[i].item() == -100


def test_batch_id_deterministic(tokenizer: FrozenTokenizer) -> None:
    tokens = _make_tokens(20, tokenizer.EOS_ID)
    seq1 = pack_sequence(tokens, "s0", "english_web", 0, SEQ_LEN, tokenizer)
    seq2 = pack_sequence(tokens, "s0", "english_web", 0, SEQ_LEN, tokenizer)
    b1 = assemble_batch([seq1], step=5)
    b2 = assemble_batch([seq2], step=5)
    assert b1.batch_id == b2.batch_id


def test_batch_id_changes_with_step(tokenizer: FrozenTokenizer) -> None:
    tokens = _make_tokens(20, tokenizer.EOS_ID)
    seq = pack_sequence(tokens, "s0", "english_web", 0, SEQ_LEN, tokenizer)
    b1 = assemble_batch([seq], step=1)
    b2 = assemble_batch([seq], step=2)
    assert b1.batch_id != b2.batch_id


def test_packing_utilization_nonzero(tokenizer: FrozenTokenizer) -> None:
    tokens = _make_tokens(40, tokenizer.EOS_ID)
    seq = pack_sequence(tokens, "s0", "english_web", 0, SEQ_LEN, tokenizer)
    util = packing_utilization([seq], SEQ_LEN)
    assert util > 0.0
    assert util <= 1.0
