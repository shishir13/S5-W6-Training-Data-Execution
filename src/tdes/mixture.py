"""Mixture schedule — curriculum phases with per-step lane weights and protected floors."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


LANES = ["english_web", "code", "instruction", "math"]
EVAL_LANE = "eval"

# Protected floors: minimum realized consumed share per lane
FLOORS: Dict[str, float] = {
    "english_web": 0.16,
    "code": 0.07,
    "math": 0.05,
    "instruction": 0.05,
}

# Stream priors (candidate-stream weights, not consumed)
STREAM_PRIORS: Dict[str, float] = {
    "english_web": 0.40,
    "code": 0.25,
    "instruction": 0.20,
    "math": 0.15,
}


@dataclass
class CurriculumPhase:
    name: str
    start_step: int        # inclusive
    end_step: int          # exclusive
    lane_weights: Dict[str, float]
    context_length: int


def build_curriculum(total_steps: int = 60) -> List[CurriculumPhase]:
    """Four curriculum phases scaled to total_steps."""
    p0_end = max(2, round(0.03 * total_steps))
    p1_end = max(p0_end + 1, round(0.55 * total_steps))
    p2_end = max(p1_end + 1, round(0.85 * total_steps))
    p3_end = total_steps

    return [
        CurriculumPhase(
            name="warmup",
            start_step=0,
            end_step=p0_end,
            lane_weights={"english_web": 0.60, "code": 0.20, "instruction": 0.10, "math": 0.10},
            context_length=128,
        ),
        CurriculumPhase(
            name="foundation",
            start_step=p0_end,
            end_step=p1_end,
            lane_weights={"english_web": 0.40, "code": 0.25, "instruction": 0.20, "math": 0.15},
            context_length=256,
        ),
        CurriculumPhase(
            name="skill_ramp",
            start_step=p1_end,
            end_step=p2_end,
            lane_weights={"english_web": 0.25, "code": 0.30, "instruction": 0.25, "math": 0.20},
            context_length=256,
        ),
        CurriculumPhase(
            name="anneal",
            start_step=p2_end,
            end_step=p3_end,
            lane_weights={"english_web": 0.20, "code": 0.20, "instruction": 0.40, "math": 0.20},
            context_length=256,
        ),
    ]


def get_phase(step: int, phases: List[CurriculumPhase]) -> CurriculumPhase:
    for phase in phases:
        if phase.start_step <= step < phase.end_step:
            return phase
    return phases[-1]


def get_lane_weights(step: int, phases: List[CurriculumPhase]) -> Dict[str, float]:
    return get_phase(step, phases).lane_weights
