"""Replay — reconstruct batches from ledger and verify batch_ids match exactly."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from tdes.ledger import ConsumptionEntry, ConsumptionLedger
from tdes.manifest import ShardManifest
from tdes.sharding import load_shard_tokens


SEQ_LEN = 256  # must match run_demo.py


@dataclass
class ReplayRecord:
    step: int
    batch_id_original: str
    batch_id_replayed: str
    shard_hash_ok: bool
    batch_id_match: bool


def _pad_tokens(tokens: List[int], seq_len: int) -> np.ndarray:
    arr = tokens[:seq_len] + [0] * max(0, seq_len - len(tokens[:seq_len]))
    return np.array(arr, dtype=np.int64)


def _recompute_batch_id(step: int, input_ids_bytes: bytes) -> str:
    payload = step.to_bytes(8, "big") + input_ids_bytes
    return hashlib.sha256(payload).hexdigest()[:16]


def replay_from_ledger(
    consumption_ledger_path: Path,
    manifests_dir: Path,
    shards_dir: Path,
) -> List[ReplayRecord]:
    """For each consumption entry, reload all shards in the batch, repack to
    get input_ids, recompute batch_id, and assert it matches the stored value."""
    ledger = ConsumptionLedger(consumption_ledger_path)
    entries = ledger.all_entries()

    records: List[ReplayRecord] = []
    for entry in entries:
        # Verify primary shard hash
        shard_path = shards_dir / f"{entry.shard_id}.bin"
        if not shard_path.exists():
            records.append(ReplayRecord(
                step=entry.step,
                batch_id_original=entry.batch_id,
                batch_id_replayed="MISSING_SHARD",
                shard_hash_ok=False,
                batch_id_match=False,
            ))
            continue

        actual_hash = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        shard_hash_ok = actual_hash == entry.shard_hash

        if not shard_hash_ok:
            records.append(ReplayRecord(
                step=entry.step,
                batch_id_original=entry.batch_id,
                batch_id_replayed="SHARD_HASH_MISMATCH",
                shard_hash_ok=False,
                batch_id_match=False,
            ))
            continue

        # Reconstruct the batch input_ids from stored shard IDs and spans
        all_shard_ids_str = getattr(entry, "all_shard_ids", "")
        all_span_starts_str = getattr(entry, "all_span_starts", "")
        all_span_ends_str = getattr(entry, "all_span_ends", "")

        if all_shard_ids_str:
            shard_ids = all_shard_ids_str.split(",")
            span_starts = [int(x) for x in all_span_starts_str.split(",")]
            span_ends = [int(x) for x in all_span_ends_str.split(",")]
        else:
            # Legacy: only primary shard stored — reconstruct with just that
            shard_ids = [entry.shard_id]
            span_starts = [entry.token_span_start]
            span_ends = [entry.token_span_end]

        rows: List[np.ndarray] = []
        shard_ok = True
        for sid, s_start, s_end in zip(shard_ids, span_starts, span_ends):
            sp = shards_dir / f"{sid}.bin"
            if not sp.exists():
                shard_ok = False
                break
            toks = load_shard_tokens(sp)
            span = toks[s_start:s_end]
            rows.append(_pad_tokens(span, SEQ_LEN))

        if not shard_ok or not rows:
            records.append(ReplayRecord(
                step=entry.step,
                batch_id_original=entry.batch_id,
                batch_id_replayed="MISSING_BATCH_SHARD",
                shard_hash_ok=shard_hash_ok,
                batch_id_match=False,
            ))
            continue

        batch_arr = np.stack(rows)  # [B, SEQ_LEN] int64
        replayed_id = _recompute_batch_id(entry.step, batch_arr.tobytes())

        records.append(ReplayRecord(
            step=entry.step,
            batch_id_original=entry.batch_id,
            batch_id_replayed=replayed_id,
            shard_hash_ok=True,
            batch_id_match=replayed_id == entry.batch_id,
        ))

    return records
