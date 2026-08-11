"""Tests for ledger append, offset tracking, and truncation."""
from __future__ import annotations

from pathlib import Path

import pytest

from tdes.ledger import (
    ConsumptionLedger, LearningLedger,
    ConsumptionEntry, LearningEntry,
)


def _make_consumption(batch_id: str = "abc", step: int = 0) -> ConsumptionEntry:
    return ConsumptionEntry(
        batch_id=batch_id,
        step=step,
        lane="english_web",
        shard_id="shard_english_web_0000",
        shard_hash="a" * 64,
        token_span_start=0,
        token_span_end=100,
        token_count=100,
        opus_decision="accept",
        opus_utility=0.42,
        is_eval=False,
        ledger_offset=0,
    )


def _make_learning(batch_id: str = "abc", step: int = 0) -> LearningEntry:
    return LearningEntry(
        batch_id=batch_id,
        step=step,
        loss=2.5,
        perplexity=12.2,
        loss_bearing_tokens=90,
        total_tokens=100,
        lane_breakdown={"english_web": 100},
        source_shard_ids=["shard_english_web_0000"],
        linked_consumption_offset=0,
        ledger_offset=0,
    )


def test_consumption_append_increases_size(tmp_path: Path) -> None:
    ledger = ConsumptionLedger(tmp_path / "c.jsonl")
    assert len(ledger) == 0
    ledger.append_entry(_make_consumption("id1", 0))
    assert len(ledger) == 1
    ledger.append_entry(_make_consumption("id2", 1))
    assert len(ledger) == 2


def test_ledger_offset_equals_file_position(tmp_path: Path) -> None:
    ledger = ConsumptionLedger(tmp_path / "c.jsonl")
    offset_before = ledger.current_offset()
    ledger.append_entry(_make_consumption("id1", 0))
    entries = ledger.all_entries()
    assert entries[0].ledger_offset == offset_before


def test_truncation_restores_state(tmp_path: Path) -> None:
    ledger = ConsumptionLedger(tmp_path / "c.jsonl")
    ledger.append_entry(_make_consumption("id1", 0))
    offset_after_first = ledger.current_offset()
    ledger.append_entry(_make_consumption("id2", 1))
    assert len(ledger) == 2
    ledger.truncate_to(offset_after_first)
    assert len(ledger) == 1
    entries = ledger.all_entries()
    assert entries[0].batch_id == "id1"


def test_learning_ledger_roundtrip(tmp_path: Path) -> None:
    ledger = LearningLedger(tmp_path / "l.jsonl")
    entry = _make_learning("xyz", 5)
    ledger.append_entry(entry)
    loaded = ledger.all_entries()
    assert len(loaded) == 1
    assert loaded[0].batch_id == "xyz"
    assert loaded[0].loss == 2.5


def test_ledger_links_consumption_to_learning(tmp_path: Path) -> None:
    c_ledger = ConsumptionLedger(tmp_path / "c.jsonl")
    l_ledger = LearningLedger(tmp_path / "l.jsonl")
    c_offset = c_ledger.current_offset()
    c_ledger.append_entry(_make_consumption("batch1", 0))
    l_entry = _make_learning("batch1", 0)
    l_entry.linked_consumption_offset = c_offset
    l_ledger.append_entry(l_entry)

    c_entries = c_ledger.all_entries()
    l_entries = l_ledger.all_entries()
    assert l_entries[0].linked_consumption_offset == c_entries[0].ledger_offset
