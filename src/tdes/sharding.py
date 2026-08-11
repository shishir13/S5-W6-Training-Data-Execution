"""Shard creation — tokenize documents and write immutable binary shards."""
from __future__ import annotations

import array
import hashlib
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from tdes.corpus import Document
from tdes.manifest import ShardManifest
from tdes.tokenizer import FrozenTokenizer


def _tokens_to_bytes(tokens: List[int]) -> bytes:
    """Pack token IDs as big-endian uint32."""
    buf = array.array("I", tokens)
    # ensure big-endian on all platforms
    if struct.pack("H", 1) == b"\x00\x01":  # already big-endian
        return buf.tobytes()
    buf.byteswap()
    return buf.tobytes()


def _bytes_to_tokens(data: bytes) -> List[int]:
    buf = array.array("I", data)
    if struct.pack("H", 1) != b"\x00\x01":  # little-endian machine
        buf.byteswap()
    return list(buf)


def build_shards(
    docs: List[Document],
    tokenizer: FrozenTokenizer,
    artifacts_root: Path,
) -> Dict[str, ShardManifest]:
    """Tokenize each document, write a .bin shard, and produce a manifest.

    One shard per document (small corpus — one doc = one shard).
    Returns mapping shard_id -> ShardManifest.
    """
    manifests_dir = artifacts_root / "manifests"
    shards_dir = artifacts_root / "shards"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, ShardManifest] = {}
    lane_counters: Dict[str, int] = {}

    for doc in docs:
        idx = lane_counters.get(doc.lane, 0)
        lane_counters[doc.lane] = idx + 1
        shard_id = f"shard_{doc.lane}_{idx:04d}"

        tokens = tokenizer.encode(doc.text)
        if not tokens:
            tokens = [tokenizer.EOS_ID]

        raw_bytes = _tokens_to_bytes(tokens)
        content_hash = hashlib.sha256(raw_bytes).hexdigest()

        shard_rel = f"shards/{shard_id}.bin"
        shard_abs = artifacts_root / shard_rel
        shard_abs.write_bytes(raw_bytes)

        manifest = ShardManifest(
            shard_id=shard_id,
            lane=doc.lane,
            shard_file=shard_rel,
            token_count=len(tokens),
            sha256_content=content_hash,
            tokenizer_id="gpt2",
            tokenizer_hash=tokenizer.hash,
            created_at=datetime.now(timezone.utc).isoformat(),
            is_eval=doc.is_eval,
            doc_count=1,
            source_doc_ids=[doc.doc_id],
        )
        manifest.save(manifests_dir / f"{shard_id}.json")
        results[shard_id] = manifest

    return results


def load_shard_tokens(shard_file: Path) -> List[int]:
    return _bytes_to_tokens(shard_file.read_bytes())


def verify_all_manifests(
    manifests: Dict[str, ShardManifest],
    artifacts_root: Path,
) -> List[str]:
    """Return list of shard_ids that FAIL hash verification (empty = all good)."""
    failures = []
    for shard_id, manifest in manifests.items():
        if not manifest.validate_file(artifacts_root):
            failures.append(shard_id)
    return failures
