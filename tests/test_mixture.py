"""Tests for mixture schedule and curriculum phases."""
from __future__ import annotations

import pytest

from tdes.mixture import build_curriculum, get_phase, get_lane_weights, FLOORS, LANES


TOTAL_STEPS = 60


def test_phases_cover_all_steps() -> None:
    phases = build_curriculum(TOTAL_STEPS)
    covered = set()
    for p in phases:
        for s in range(p.start_step, p.end_step):
            covered.add(s)
    assert covered == set(range(TOTAL_STEPS))


def test_phase_names_in_order() -> None:
    phases = build_curriculum(TOTAL_STEPS)
    names = [p.name for p in phases]
    assert names == ["warmup", "foundation", "skill_ramp", "anneal"]


def test_lane_weights_sum_to_one() -> None:
    phases = build_curriculum(TOTAL_STEPS)
    for p in phases:
        total = sum(p.lane_weights.values())
        assert abs(total - 1.0) < 1e-6, f"Phase {p.name} weights sum to {total}"


def test_get_phase_returns_correct_phase() -> None:
    phases = build_curriculum(TOTAL_STEPS)
    assert get_phase(0, phases).name == "warmup"
    assert get_phase(TOTAL_STEPS - 1, phases).name == "anneal"


def test_floors_are_positive() -> None:
    for lane, floor in FLOORS.items():
        assert floor > 0.0, f"Floor for {lane} should be positive"


def test_floors_sum_below_one() -> None:
    assert sum(FLOORS.values()) < 1.0, "Floors must leave room for OPUS free allocation"


def test_eval_lane_not_in_training_lanes() -> None:
    assert "eval" not in LANES


def test_lane_weights_all_lanes_present() -> None:
    phases = build_curriculum(TOTAL_STEPS)
    for p in phases:
        for lane in LANES:
            assert lane in p.lane_weights, f"Lane {lane} missing in phase {p.name}"
