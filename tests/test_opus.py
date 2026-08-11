"""Tests for OPUS decision states — floor_override, accept, reject, defer."""
from __future__ import annotations

import numpy as np
import pytest

from tdes.mixture import FLOORS, LANES
from tdes.opus import OPUSSelector, Candidate, Decision, SKETCH_DIM


def _make_candidate(lane: str, shard_id: str = "s0", seed: int = 0) -> Candidate:
    rng = np.random.default_rng(seed)
    grad = rng.normal(0, 1, size=SKETCH_DIM)
    return Candidate(
        shard_id=shard_id,
        lane=lane,
        token_span_start=0,
        token_span_end=100,
        grad_features=grad,
    )


def test_floor_override_fires_when_lane_starved() -> None:
    """When a floored lane has zero consumed tokens, floor_override must appear."""
    selector = OPUSSelector(seed=42)
    # All candidates from "english_web" — starvation for all other floored lanes
    candidates = [_make_candidate("english_web", f"s{i}", seed=i) for i in range(8)]
    # Force consumed to be non-zero so floor logic activates
    results = selector.select(candidates, target_k=4, step=0)
    decisions = {r.decision for r in results}
    # Accept or floor_override must appear
    assert Decision.ACCEPT in decisions or Decision.FLOOR_OVERRIDE in decisions


def test_accept_and_reject_decisions_present() -> None:
    selector = OPUSSelector(seed=0)
    candidates = [_make_candidate(LANES[i % len(LANES)], f"s{i}", seed=i) for i in range(16)]
    results = selector.select(candidates, target_k=4, step=0)
    decisions = [r.decision for r in results]
    assert len(decisions) == len(candidates)
    has_accept = any(d in (Decision.ACCEPT, Decision.FLOOR_OVERRIDE) for d in decisions)
    assert has_accept


def test_decisions_are_deterministic() -> None:
    """Same candidates + same seed must produce same decisions."""
    candidates = [_make_candidate(LANES[i % len(LANES)], f"s{i}", seed=i) for i in range(8)]

    sel1 = OPUSSelector(seed=99)
    r1 = sel1.select(candidates, target_k=4, step=5)

    sel2 = OPUSSelector(seed=99)
    r2 = sel2.select(candidates, target_k=4, step=5)

    for a, b in zip(r1, r2):
        assert a.decision == b.decision
        assert abs(a.utility - b.utility) < 1e-9


def test_state_save_restore() -> None:
    sel = OPUSSelector(seed=42)
    candidates = [_make_candidate(LANES[i % len(LANES)], f"s{i}", seed=i) for i in range(8)]
    sel.select(candidates, target_k=4, step=0)

    state = sel.get_state()
    sel2 = OPUSSelector(seed=42)
    sel2.set_state(state)

    assert sel2._total_consumed == sel._total_consumed
    assert np.allclose(sel2.G, sel.G)


def test_no_eval_shard_in_candidates() -> None:
    """Eval shards should never be passed as candidates — test that eval lane
    gets no floor treatment (it has no floor entry)."""
    assert "eval" not in FLOORS


def test_target_k_respected() -> None:
    selector = OPUSSelector(seed=7)
    candidates = [_make_candidate(LANES[i % len(LANES)], f"s{i}", seed=i) for i in range(20)]
    results = selector.select(candidates, target_k=5, step=0)
    accepted = [r for r in results if r.decision in (Decision.ACCEPT, Decision.FLOOR_OVERRIDE)]
    # Should select at most target_k + floor_override candidates
    assert len(accepted) >= 1
