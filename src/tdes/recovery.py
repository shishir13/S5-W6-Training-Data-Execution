"""Crash recovery — detect crash, restore state, verify next batch_id matches."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple


CRASH_FLAG = "crash.flag"


def write_crash_flag(artifacts_root: Path) -> None:
    (artifacts_root / CRASH_FLAG).write_text("crashed")


def crash_flag_exists(artifacts_root: Path) -> bool:
    return (artifacts_root / CRASH_FLAG).exists()


def clear_crash_flag(artifacts_root: Path) -> None:
    flag = artifacts_root / CRASH_FLAG
    if flag.exists():
        flag.unlink()


def truncate_ledgers_to_checkpoint(
    consumption_ledger_path: Path,
    learning_ledger_path: Path,
    consumption_offset: int,
    learning_offset: int,
) -> None:
    """Roll back both ledgers to the exact byte offsets stored in the checkpoint."""
    with consumption_ledger_path.open("r+b") as f:
        f.truncate(consumption_offset)
    with learning_ledger_path.open("r+b") as f:
        f.truncate(learning_offset)
