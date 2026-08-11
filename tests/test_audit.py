"""Tests for the evidence bundle schema and audit correctness."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tdes.audit import run_audit, write_evidence_json, write_evidence_md
from tdes.corpus import generate_corpus
from tdes.ledger import ConsumptionLedger, LearningLedger, ConsumptionEntry, LearningEntry
from tdes.manifest import ShardManifest
from tdes.replay import ReplayRecord
from tdes.sharding import build_shards
from tdes.tokenizer import FrozenTokenizer


@pytest.fixture(scope="module")
def tokenizer() -> FrozenTokenizer:
    return FrozenTokenizer.load()


@pytest.fixture()
def minimal_setup(tmp_path: Path, tokenizer: FrozenTokenizer):
    docs = generate_corpus()
    manifests = build_shards(docs, tokenizer, tmp_path)

    # Write a minimal consumption + learning entry (non-eval only)
    c_ledger = ConsumptionLedger(tmp_path / "ledgers" / "consumption_ledger.jsonl")
    l_ledger = LearningLedger(tmp_path / "ledgers" / "learning_ledger.jsonl")

    shard_id = next(sid for sid, m in manifests.items() if not m.is_eval)
    m = manifests[shard_id]
    c_offset = c_ledger.current_offset()
    c_entry = ConsumptionEntry(
        batch_id="testbatch",
        step=0,
        lane=m.lane,
        shard_id=shard_id,
        shard_hash=m.sha256_content,
        token_span_start=0,
        token_span_end=50,
        token_count=50,
        opus_decision="accept",
        opus_utility=0.5,
        is_eval=False,
        ledger_offset=c_offset,
    )
    c_ledger.append_entry(c_entry)

    l_offset = l_ledger.current_offset()
    l_entry = LearningEntry(
        batch_id="testbatch",
        step=0,
        loss=3.0,
        perplexity=20.1,
        loss_bearing_tokens=40,
        total_tokens=50,
        lane_breakdown={m.lane: 50},
        source_shard_ids=[shard_id],
        linked_consumption_offset=c_offset,
        ledger_offset=l_offset,
    )
    l_ledger.append_entry(l_entry)

    # Move ledgers into the expected path structure
    (tmp_path / "ledgers").mkdir(exist_ok=True)

    replay = [ReplayRecord(
        step=0,
        batch_id_original="testbatch",
        batch_id_replayed="testbatch",
        shard_hash_ok=True,
        batch_id_match=True,
    )]

    return tmp_path, manifests, tokenizer, replay


def test_evidence_json_has_all_required_keys(minimal_setup) -> None:
    root, manifests, tokenizer, replay = minimal_setup
    evidence = run_audit(
        artifacts_root=root,
        manifests=manifests,
        tokenizer_hash=tokenizer.hash,
        resume_expected_id="abc",
        resume_produced_id="abc",
        replay_records=replay,
        fork_original_id="orig",
        fork_forked_id="fork",
        fork_step=10,
        perf_summary={"mean_packing_utilization_pct": 80.0},
    )
    required_keys = [
        "shard_hashes_frozen", "tokenizer_hash_match", "eval_firewall",
        "floors_never_violated", "opus_decisions", "ledger_completeness",
        "crash_resume_batch_id_match", "replay_all_match",
        "fork_diverges_from_original", "packing_utilization",
    ]
    for k in required_keys:
        assert k in evidence["criteria"], f"Missing key: {k}"


def test_evidence_pass_values_are_booleans(minimal_setup) -> None:
    root, manifests, tokenizer, replay = minimal_setup
    evidence = run_audit(
        artifacts_root=root,
        manifests=manifests,
        tokenizer_hash=tokenizer.hash,
        resume_expected_id="abc",
        resume_produced_id="abc",
        replay_records=replay,
        fork_original_id="orig",
        fork_forked_id="fork",
        fork_step=10,
        perf_summary={"mean_packing_utilization_pct": 80.0},
    )
    for k, v in evidence["criteria"].items():
        assert isinstance(v["pass"], bool), f"Key {k}: 'pass' should be bool, got {type(v['pass'])}"


def test_evidence_json_is_valid_json(minimal_setup, tmp_path: Path) -> None:
    root, manifests, tokenizer, replay = minimal_setup
    evidence = run_audit(
        artifacts_root=root,
        manifests=manifests,
        tokenizer_hash=tokenizer.hash,
        resume_expected_id="x",
        resume_produced_id="x",
        replay_records=replay,
        fork_original_id="a",
        fork_forked_id="b",
        fork_step=10,
        perf_summary={"mean_packing_utilization_pct": 75.0},
    )
    write_evidence_json(evidence, root)
    written = json.loads((root / "evidence.json").read_text())
    assert "criteria" in written
    assert "ledger_summary" in written


def test_evidence_md_contains_pass_fail(minimal_setup) -> None:
    root, manifests, tokenizer, replay = minimal_setup
    evidence = run_audit(
        artifacts_root=root,
        manifests=manifests,
        tokenizer_hash=tokenizer.hash,
        resume_expected_id="match",
        resume_produced_id="match",
        replay_records=replay,
        fork_original_id="a",
        fork_forked_id="b",
        fork_step=10,
        perf_summary={"mean_packing_utilization_pct": 85.0},
    )
    write_evidence_md(evidence, root)
    md = (root / "evidence.md").read_text()
    assert "PASS" in md or "FAIL" in md
    assert "Evidence Summary" in md
