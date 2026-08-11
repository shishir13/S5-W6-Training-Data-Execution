"""OPUS selection — accept / reject / defer / floor_override decisions."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from tdes.manifest import ShardManifest
from tdes.mixture import FLOORS, LANES


class Decision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    DEFER = "defer"
    FLOOR_OVERRIDE = "floor_override"


@dataclass
class Candidate:
    shard_id: str
    lane: str
    token_span_start: int
    token_span_end: int
    grad_features: np.ndarray  # CountSketch projected gradient


@dataclass
class OPUSResult:
    candidate: Candidate
    decision: Decision
    utility: float


# Bench-Proxy direction weights (multi-domain, from S5 plan)
PROXY_WEIGHTS: Dict[str, float] = {
    "english_web": 0.22,
    "code": 0.30,
    "instruction": 0.24,
    "math": 0.24,
}

SKETCH_DIM = 64  # small for CPU demo (real = 8192)


def _countsketch(grad: np.ndarray, m: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = len(grad)
    h = rng.integers(0, m, size=d)
    s = rng.choice([-1, 1], size=d)
    sketch = np.zeros(m)
    np.add.at(sketch, h, s * grad)
    return sketch / math.sqrt(max(d / m, 1))


def _make_lane_basis(m: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    basis: Dict[str, np.ndarray] = {}
    for i, lane in enumerate(LANES):
        base = np.zeros(m)
        base[i % m] = 1.0
        noise = rng.normal(0, 0.3, size=m)
        basis[lane] = base + noise
    return basis


class OPUSSelector:
    """Simplified OPUS selector with explicit decision states."""

    def __init__(self, seed: int = 42) -> None:
        self.m = SKETCH_DIM
        self._rng = np.random.default_rng(seed)
        self._lane_basis = _make_lane_basis(self.m, self._rng)
        self.G = np.zeros(self.m)
        self.v_hat = np.ones(self.m) * 0.01
        self.eta = 0.01
        self.tau = 0.9
        self._deferred: List[Candidate] = []   # deferred queue (depth 1 per lane)
        self._consumed_counts: Dict[str, int] = {lane: 0 for lane in LANES}
        self._total_consumed: int = 0

    def _build_proxy(self) -> np.ndarray:
        g = np.zeros(self.m)
        for lane, w in PROXY_WEIGHTS.items():
            if lane in self._lane_basis:
                g += w * self._lane_basis[lane]
        norm = np.linalg.norm(g)
        return g / norm if norm > 1e-9 else g

    def _precondition(self, grad: np.ndarray) -> np.ndarray:
        return grad / (np.sqrt(self.v_hat) + 1e-8)

    def _utility(self, grad: np.ndarray, g_proxy: np.ndarray) -> float:
        u = self._precondition(grad)
        alignment = self.eta * float(np.dot(u, g_proxy))
        redundancy = (self.eta ** 2) * float(np.dot(u, self.G))
        return alignment - redundancy

    def select(
        self,
        candidates: List[Candidate],
        target_k: int,
        step: int,
    ) -> List[OPUSResult]:
        g_proxy = self._build_proxy()

        # Inject deferred candidates from previous step
        all_candidates = self._deferred + candidates
        self._deferred = []

        utilities = [self._utility(c.grad_features, g_proxy) for c in all_candidates]
        results: List[OPUSResult] = []
        selected_indices: List[int] = []

        # Phase 1: floor protection
        realized = self._realized_shares()
        for lane, floor in FLOORS.items():
            if realized.get(lane, 0.0) < floor:
                lane_idxs = [i for i, c in enumerate(all_candidates) if c.lane == lane
                             and i not in selected_indices]
                if lane_idxs:
                    # pick highest utility from this lane
                    best = max(lane_idxs, key=lambda i: utilities[i])
                    selected_indices.append(best)
                    results.append(OPUSResult(
                        candidate=all_candidates[best],
                        decision=Decision.FLOOR_OVERRIDE,
                        utility=utilities[best],
                    ))

        # Phase 2: Boltzmann sampling for remaining slots
        slots_left = target_k - len(selected_indices)
        remaining = [i for i in range(len(all_candidates)) if i not in selected_indices]
        if slots_left > 0 and remaining:
            rem_utils = np.array([utilities[i] for i in remaining])
            rem_utils = rem_utils - rem_utils.max()
            probs = np.exp(rem_utils / self.tau)
            probs /= probs.sum()
            n_pick = min(slots_left, len(remaining))
            chosen = np.random.choice(remaining, size=n_pick, replace=False, p=probs).tolist()
            for i in chosen:
                selected_indices.append(i)
                results.append(OPUSResult(
                    candidate=all_candidates[i],
                    decision=Decision.ACCEPT,
                    utility=utilities[i],
                ))

        # Phase 3: reject / defer remaining
        selected_set = set(selected_indices)
        for i, cand in enumerate(all_candidates):
            if i in selected_set:
                continue
            # Defer if lane is approaching its floor
            lane_floor = FLOORS.get(cand.lane, 0.0)
            lane_share = realized.get(cand.lane, 0.0)
            if lane_floor > 0 and lane_share < lane_floor * 1.5 and len(self._deferred) < 8:
                self._deferred.append(cand)
                results.append(OPUSResult(
                    candidate=cand, decision=Decision.DEFER, utility=utilities[i]
                ))
            else:
                results.append(OPUSResult(
                    candidate=cand, decision=Decision.REJECT, utility=utilities[i]
                ))

        # Update running state for accepted + floor_override
        for r in results:
            if r.decision in (Decision.ACCEPT, Decision.FLOOR_OVERRIDE):
                u = self._precondition(r.candidate.grad_features)
                self.G += u
                self._consumed_counts[r.candidate.lane] = (
                    self._consumed_counts.get(r.candidate.lane, 0) + 1
                )
                self._total_consumed += 1

        # Decay v_hat
        self.v_hat = self.v_hat * 0.99 + 0.001 * self._rng.random(self.m)

        return results

    def _realized_shares(self) -> Dict[str, float]:
        if self._total_consumed == 0:
            return {}
        return {lane: cnt / self._total_consumed
                for lane, cnt in self._consumed_counts.items()}

    def get_state(self) -> dict:  # type: ignore[type-arg]
        return {
            "G": self.G.tolist(),
            "v_hat": self.v_hat.tolist(),
            "consumed_counts": dict(self._consumed_counts),
            "total_consumed": self._total_consumed,
            "deferred_lanes": [c.lane for c in self._deferred],
        }

    def set_state(self, state: dict) -> None:  # type: ignore[type-arg]
        self.G = np.array(state["G"])
        self.v_hat = np.array(state["v_hat"])
        self._consumed_counts = dict(state["consumed_counts"])
        self._total_consumed = state["total_consumed"]
        self._deferred = []  # deferred queue not persisted (by design — it's ephemeral)
