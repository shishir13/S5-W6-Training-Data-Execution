"""Tests for shard creation and manifest integrity."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tdes.corpus import generate_corpus
from tdes.manifest import ShardManifest
from tdes.sharding import build_shards, load_shard_tokens, verify_all_manifests
from tdes.tokenizer import FrozenTokenizer


@pytest.fixture(scope="module")
def tokenizer() -> FrozenTokenizer:
    return FrozenTokenizer.load()


@pytest.fixture()
def built_shards(tmp_path: Path, tokenizer: FrozenTokenizer):
    docs = generate_corpus()
    manifests = build_shards(docs, tokenizer, tmp_path)
    return tmp_path, manifests, tokenizer


def test_shard_sha256_reproducible(tmp_path: Path, tokenizer: FrozenTokenizer) -> None:
    docs = generate_corpus()
    m1 = build_shards(docs, tokenizer, tmp_path / "run1")
    m2 = build_shards(docs, tokenizer, tmp_path / "run2")
    for shard_id in m1:
        assert m1[shard_id].sha256_content == m2[shard_id].sha256_content


def test_tokenizer_hash_stable(tokenizer: FrozenTokenizer) -> None:
    tok2 = FrozenTokenizer.load()
    assert tokenizer.hash == tok2.hash


def test_shard_manifest_schema(built_shards) -> None:
    _, manifests, _ = built_shards
    required = [
        "shard_id", "lane", "shard_file", "token_count",
        "sha256_content", "tokenizer_id", "tokenizer_hash",
        "created_at", "is_eval", "doc_count", "source_doc_ids",
    ]
    for m in manifests.values():
        for field in required:
            assert hasattr(m, field), f"Missing field: {field}"
        assert m.token_count > 0
        assert len(m.sha256_content) == 64  # hex SHA-256


def test_eval_shard_flagged(built_shards) -> None:
    _, manifests, _ = built_shards
    eval_shards = [m for m in manifests.values() if m.is_eval]
    assert len(eval_shards) > 0, "No eval shards found"
    for m in eval_shards:
        assert m.lane == "eval"


def test_verify_all_manifests_passes(built_shards) -> None:
    root, manifests, _ = built_shards
    failures = verify_all_manifests(manifests, root)
    assert failures == [], f"Hash failures: {failures}"


def test_modified_shard_fails_verification(built_shards) -> None:
    root, manifests, _ = built_shards
    shard_id = next(iter(manifests))
    shard_path = root / manifests[shard_id].shard_file
    data = bytearray(shard_path.read_bytes())
    data[0] ^= 0xFF  # flip first byte
    shard_path.write_bytes(bytes(data))
    failures = verify_all_manifests(manifests, root)
    assert shard_id in failures


def test_tokenizer_hash_mismatch_raises(built_shards) -> None:
    _, manifests, tokenizer = built_shards
    from tdes.tokenizer import TokenizerMismatch
    with pytest.raises(TokenizerMismatch):
        tokenizer.verify_hash("deadbeef" * 8)


def test_load_shard_tokens_roundtrip(built_shards) -> None:
    root, manifests, tokenizer = built_shards
    for shard_id, m in list(manifests.items())[:3]:
        shard_path = root / m.shard_file
        tokens = load_shard_tokens(shard_path)
        assert len(tokens) == m.token_count
        assert all(0 <= t < tokenizer.vocab_size for t in tokens)
