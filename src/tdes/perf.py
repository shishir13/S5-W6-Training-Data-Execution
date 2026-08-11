"""Performance tracking — tokens/sec, packing utilization."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class StepPerf:
    step: int
    elapsed_sec: float
    total_tokens: int
    loss_bearing_tokens: int
    packing_utilization: float


class PerfTracker:
    def __init__(self) -> None:
        self._records: List[StepPerf] = []
        self._step_start: float = 0.0

    def start_step(self) -> None:
        self._step_start = time.perf_counter()

    def record_step(
        self,
        step: int,
        total_tokens: int,
        loss_bearing_tokens: int,
        packing_utilization: float,
    ) -> None:
        elapsed = time.perf_counter() - self._step_start
        self._records.append(StepPerf(
            step=step,
            elapsed_sec=elapsed,
            total_tokens=total_tokens,
            loss_bearing_tokens=loss_bearing_tokens,
            packing_utilization=packing_utilization,
        ))

    def summary(self) -> Dict[str, float]:
        if not self._records:
            return {}
        total_tokens = sum(r.total_tokens for r in self._records)
        loss_tokens = sum(r.loss_bearing_tokens for r in self._records)
        total_time = sum(r.elapsed_sec for r in self._records)
        utils = [r.packing_utilization for r in self._records]
        return {
            "total_steps": len(self._records),
            "total_tokens": total_tokens,
            "total_loss_bearing_tokens": loss_tokens,
            "total_time_sec": round(total_time, 3),
            "tokens_per_sec": round(total_tokens / total_time, 1) if total_time > 0 else 0.0,
            "loss_tokens_per_sec": round(loss_tokens / total_time, 1) if total_time > 0 else 0.0,
            "mean_packing_utilization_pct": round(100.0 * sum(utils) / len(utils), 2),
            "min_packing_utilization_pct": round(100.0 * min(utils), 2),
            "max_packing_utilization_pct": round(100.0 * max(utils), 2),
        }
