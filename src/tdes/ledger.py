"""Append-only JSONL ledger with byte-level offset tracking."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional


@dataclass
class ConsumptionEntry:
    batch_id: str
    step: int
    lane: str
    shard_id: str
    shard_hash: str
    token_span_start: int
    token_span_end: int
    token_count: int
    opus_decision: str
    opus_utility: float
    is_eval: bool
    ledger_offset: int      # byte offset of THIS line in the file
    batch_input_hash: str = ""   # sha256(input_ids.numpy().tobytes()) — for replay
    all_shard_ids: str = ""      # comma-separated list of all shard_ids in this batch
    all_span_starts: str = ""    # comma-separated token_span_start for each sequence
    all_span_ends: str = ""      # comma-separated token_span_end for each sequence


@dataclass
class LearningEntry:
    batch_id: str
    step: int
    loss: float
    perplexity: float
    loss_bearing_tokens: int
    total_tokens: int
    lane_breakdown: Dict[str, int]
    source_shard_ids: List[str]
    linked_consumption_offset: int   # byte offset of matching ConsumptionEntry
    ledger_offset: int


class Ledger:
    """Append-only JSONL ledger with byte-offset tracking for crash-safe truncation."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_bytes(b"")

    def current_offset(self) -> int:
        return self.path.stat().st_size

    def append(self, entry: object) -> int:
        """Append entry as JSON line. Returns the byte offset where the line starts."""
        offset = self.current_offset()
        line = json.dumps(asdict(entry)) + "\n"  # type: ignore[call-overload]
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)
        return offset

    def truncate_to(self, offset: int) -> None:
        """Truncate file to exactly `offset` bytes (roll back partial writes)."""
        with self.path.open("r+b") as f:
            f.truncate(offset)

    def read_all(self) -> List[dict]:  # type: ignore[type-arg]
        lines = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(json.loads(line))
        return lines

    def __len__(self) -> int:
        return len(self.read_all())


class ConsumptionLedger(Ledger):
    def append_entry(self, entry: ConsumptionEntry) -> int:
        return self.append(entry)

    def all_entries(self) -> List[ConsumptionEntry]:
        return [ConsumptionEntry(**d) for d in self.read_all()]


class LearningLedger(Ledger):
    def append_entry(self, entry: LearningEntry) -> int:
        return self.append(entry)

    def all_entries(self) -> List[LearningEntry]:
        return [LearningEntry(**d) for d in self.read_all()]
