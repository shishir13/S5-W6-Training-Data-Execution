"""Checkpoint — save/load model state tied to ledger offsets."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from tdes.tokenizer import FrozenTokenizer, TokenizerMismatch


@dataclass
class CheckpointMeta:
    step: int
    consumption_ledger_offset: int
    learning_ledger_offset: int
    batch_id_at_checkpoint: str
    next_expected_batch_id: str     # pre-computed at SAVE time — not on resume
    tokenizer_hash: str
    created_at: str
    # These are stored in the .pt file alongside this meta
    # (accessed via torch.load, not dataclass fields)


def save_checkpoint(
    step: int,
    batch_id: str,
    next_expected_batch_id: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    consumption_offset: int,
    learning_offset: int,
    mixture_state: Dict[str, Any],
    opus_state: Dict[str, Any],
    tokenizer_hash: str,
    checkpoints_dir: Path,
) -> Path:
    from datetime import datetime, timezone
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    fname = f"ckpt_{step:05d}_coff{consumption_offset}.pt"
    path = checkpoints_dir / fname

    payload = {
        "step": step,
        "consumption_ledger_offset": consumption_offset,
        "learning_ledger_offset": learning_offset,
        "batch_id_at_checkpoint": batch_id,
        "next_expected_batch_id": next_expected_batch_id,
        "tokenizer_hash": tokenizer_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "mixture_state": mixture_state,
        "opus_state": opus_state,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path, tokenizer_hash: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = torch.load(path, weights_only=False)
    saved_hash = payload.get("tokenizer_hash", "")
    if saved_hash != tokenizer_hash:
        raise TokenizerMismatch(
            f"Checkpoint tokenizer_hash {saved_hash!r} != current {tokenizer_hash!r}"
        )
    return payload


def restore_rng_states(payload: Dict[str, Any]) -> None:
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])


def find_latest_checkpoint(checkpoints_dir: Path) -> Optional[Path]:
    ckpts = sorted(checkpoints_dir.glob("ckpt_*.pt"))
    return ckpts[-1] if ckpts else None


def find_checkpoint_at_step(checkpoints_dir: Path, step: int) -> Optional[Path]:
    pattern = f"ckpt_{step:05d}_*.pt"
    matches = list(checkpoints_dir.glob(pattern))
    return matches[0] if matches else None
