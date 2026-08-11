"""Tests for checkpoint save/load and tokenizer hash verification."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from tdes.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from tdes.model import TinyGPT
from tdes.tokenizer import FrozenTokenizer, TokenizerMismatch


SEQ_LEN = 64


@pytest.fixture(scope="module")
def tokenizer() -> FrozenTokenizer:
    return FrozenTokenizer.load()


def _make_model_and_opt() -> tuple:
    model = TinyGPT(seq_len=SEQ_LEN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def test_checkpoint_round_trip(tmp_path: Path, tokenizer: FrozenTokenizer) -> None:
    model, optimizer = _make_model_and_opt()
    ckpt_path = save_checkpoint(
        step=5,
        batch_id="aabbccdd",
        next_expected_batch_id="11223344",
        model=model,
        optimizer=optimizer,
        consumption_offset=128,
        learning_offset=256,
        mixture_state={"step": 5},
        opus_state={"G": [0.0] * 64, "v_hat": [0.01] * 64,
                    "consumed_counts": {}, "total_consumed": 0, "deferred_lanes": []},
        tokenizer_hash=tokenizer.hash,
        checkpoints_dir=tmp_path,
    )
    assert ckpt_path.exists()
    payload = load_checkpoint(ckpt_path, tokenizer.hash)
    assert payload["step"] == 5
    assert payload["batch_id_at_checkpoint"] == "aabbccdd"
    assert payload["next_expected_batch_id"] == "11223344"
    assert payload["consumption_ledger_offset"] == 128
    assert payload["learning_ledger_offset"] == 256


def test_tokenizer_hash_verified_on_load(tmp_path: Path, tokenizer: FrozenTokenizer) -> None:
    model, optimizer = _make_model_and_opt()
    ckpt_path = save_checkpoint(
        step=1,
        batch_id="x",
        next_expected_batch_id="y",
        model=model,
        optimizer=optimizer,
        consumption_offset=0,
        learning_offset=0,
        mixture_state={},
        opus_state={"G": [0.0]*64, "v_hat": [0.01]*64,
                    "consumed_counts": {}, "total_consumed": 0, "deferred_lanes": []},
        tokenizer_hash=tokenizer.hash,
        checkpoints_dir=tmp_path,
    )
    with pytest.raises(TokenizerMismatch):
        load_checkpoint(ckpt_path, "wrong_hash" * 4)


def test_find_latest_checkpoint(tmp_path: Path, tokenizer: FrozenTokenizer) -> None:
    model, optimizer = _make_model_and_opt()
    for step in [10, 20, 30]:
        save_checkpoint(
            step=step,
            batch_id=f"id{step}",
            next_expected_batch_id=f"next{step}",
            model=model,
            optimizer=optimizer,
            consumption_offset=step * 100,
            learning_offset=step * 50,
            mixture_state={},
            opus_state={"G": [0.0]*64, "v_hat": [0.01]*64,
                        "consumed_counts": {}, "total_consumed": 0, "deferred_lanes": []},
            tokenizer_hash=tokenizer.hash,
            checkpoints_dir=tmp_path,
        )
    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert "00030" in latest.name


def test_model_state_restored_after_load(tmp_path: Path, tokenizer: FrozenTokenizer) -> None:
    model, optimizer = _make_model_and_opt()
    # Do one optimizer step to change weights
    dummy_input = torch.randint(0, 100, (1, SEQ_LEN))
    dummy_pos = torch.arange(SEQ_LEN).unsqueeze(0)
    logits = model(dummy_input, dummy_pos)
    loss = logits.mean()
    loss.backward()
    optimizer.step()
    original_param = next(model.parameters()).data.clone()

    ckpt_path = save_checkpoint(
        step=1,
        batch_id="id1",
        next_expected_batch_id="id2",
        model=model,
        optimizer=optimizer,
        consumption_offset=0,
        learning_offset=0,
        mixture_state={},
        opus_state={"G": [0.0]*64, "v_hat": [0.01]*64,
                    "consumed_counts": {}, "total_consumed": 0, "deferred_lanes": []},
        tokenizer_hash=tokenizer.hash,
        checkpoints_dir=tmp_path,
    )

    new_model, new_opt = _make_model_and_opt()
    payload = load_checkpoint(ckpt_path, tokenizer.hash)
    new_model.load_state_dict(payload["model_state_dict"])
    restored_param = next(new_model.parameters()).data

    assert torch.allclose(original_param, restored_param)
