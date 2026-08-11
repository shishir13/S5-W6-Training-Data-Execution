"""Training step — masked cross-entropy loss."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tdes.model import TinyGPT
from tdes.packing import BatchRecord


@dataclass
class StepResult:
    loss: float
    perplexity: float
    loss_bearing_tokens: int
    total_tokens: int


def train_step(
    model: TinyGPT,
    optimizer: torch.optim.Optimizer,
    batch: BatchRecord,
) -> StepResult:
    model.train()
    optimizer.zero_grad()

    logits = model(batch.input_ids, batch.position_ids)  # [B, T, V]
    B, T, V = logits.shape

    # Flatten for cross-entropy
    logits_flat = logits.reshape(B * T, V)
    labels_flat = batch.labels.reshape(B * T)
    loss_mask_flat = batch.loss_mask.reshape(B * T)

    # Raw per-token CE (reduction=none)
    per_token_loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100, reduction="none")

    # Apply loss mask — only loss-bearing tokens contribute
    masked_loss = per_token_loss * loss_mask_flat
    n_loss_tokens = int(loss_mask_flat.sum().item())

    if n_loss_tokens > 0:
        loss = masked_loss.sum() / n_loss_tokens
    else:
        loss = masked_loss.sum() * 0.0  # zero but keeps autograd graph

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    loss_val = float(loss.item())
    perplexity = float(torch.exp(torch.tensor(loss_val)).item()) if n_loss_tokens > 0 else 0.0
    total_tokens = int((batch.labels != -100).sum().item())

    return StepResult(
        loss=loss_val,
        perplexity=perplexity,
        loss_bearing_tokens=n_loss_tokens,
        total_tokens=total_tokens,
    )
