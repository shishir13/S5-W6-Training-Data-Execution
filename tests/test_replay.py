"""Tests for replay correctness — batch_ids must match and tampered shards must fail."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest

from tdes.corpus import generate_corpus
from tdes.ledger import ConsumptionLedger, ConsumptionEntry
from tdes.manifest import ShardManifest
from tdes.packing import assemble_batch, pack_sequence
from tdes.replay import replay_from_ledger
from tdes.sharding import build_shards, load_shard_tokens
from tdes.tokenizer import FrozenTokenizer


SEQ_LEN = 256  # must match replay.py and packing.py


@pytest.fixture(scope="module")
def tokenizer() -> FrozenTokenizer:
    return FrozenTokenizer.load()


@pytest.fixture()
def setup(tmp_path: Path, tokenizer: FrozenTokenizer):
    docs = generate_corpus()
    manifests = build_shards(docs, tokenizer, tmp_path)
    return tmp_path, manifests, tokenizer


def _write_consumption_entry(
    ledger: ConsumptionLedger,
    shard_id: str,
    shard_hash: str,
    step: int,
    span_start: int,
    span_end: int,
    batch_id: str,
) -> None:
    offset = ledger.current_offset()
    entry = ConsumptionEntry(
        batch_id=batch_id,
        step=step,
        lane="english_web",
        shard_id=shard_id,
        shard_hash=shard_hash,
        token_span_start=span_start,
        token_span_end=span_end,
        token_count=span_end - span_start,
        opus_decision="accept",
        opus_utility=0.5,
        is_eval=False,
        ledger_offset=offset,
    )
    ledger.append_entry(entry)


def _compute_expected_batch_id(step: int, tokens: list) -> str:
    import hashlib
    import numpy as np
    arr = tokens[:SEQ_LEN] + [0] * max(0, SEQ_LEN - len(tokens[:SEQ_LEN]))
    row = np.array(arr, dtype=np.int64)
    batch = row[np.newaxis, :]  # [1, SEQ_LEN]
    payload = step.to_bytes(8, "big") + batch.tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def test_replay_batch_ids_match_original(setup) -> None:
    root, manifests, tokenizer = setup
    ledger_path = root / "c.jsonl"
    ledger = ConsumptionLedger(ledger_path)

    # Pick first non-eval shard
    shard_id = next(sid for sid, m in manifests.items() if not m.is_eval)
    m = manifests[shard_id]
    shard_path = root / m.shard_file
    tokens = load_shard_tokens(shard_path)
    span = tokens[:min(SEQ_LEN, len(tokens))]
    expected_id = _compute_expected_batch_id(0, span)

    _write_consumption_entry(
        ledger, shard_id, m.sha256_content,
        step=0, span_start=0, span_end=len(span), batch_id=expected_id,
    )

    records = replay_from_ledger(ledger_path, root / "manifests", root / "shards")
    assert len(records) == 1
    assert records[0].batch_id_match, (
        f"batch_id mismatch: original={records[0].batch_id_original} "
        f"replayed={records[0].batch_id_replayed}"
    )


def test_replay_fails_on_modified_shard(setup) -> None:
    root, manifests, tokenizer = setup
    ledger_path = root / "c_tamper.jsonl"
    ledger = ConsumptionLedger(ledger_path)

    shard_id = next(sid for sid, m in manifests.items() if not m.is_eval)
    m = manifests[shard_id]
    shard_path = root / m.shard_file
    tokens = load_shard_tokens(shard_path)
    span = tokens[:min(SEQ_LEN, len(tokens))]
    expected_id = _compute_expected_batch_id(0, span)

    _write_consumption_entry(
        ledger, shard_id, m.sha256_content,
        step=0, span_start=0, span_end=len(span), batch_id=expected_id,
    )

    # Tamper with the shard
    data = bytearray(shard_path.read_bytes())
    data[0] ^= 0xFF
    shard_path.write_bytes(bytes(data))

    records = replay_from_ledger(ledger_path, root / "manifests", root / "shards")
    assert len(records) == 1
    assert not records[0].shard_hash_ok
    assert not records[0].batch_id_match
