# Copyright 2026 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
WSD (Warmup-Stable-Decay) learning rate scheduler.

Three-phase schedule designed for large-scale pretraining:
  1. Warmup:  linear 0 → max_lr  (warmup_steps)
  2. Stable:  constant max_lr    (stable_steps)
  3. Decay:   cosine max_lr → min_lr  (decay_steps)

Advantages over standard cosine:
  - Stable phase makes it easy to extend training without restarting decay
  - Decay phase can be restarted from any checkpoint seamlessly
  - Works well with multi-stage training pipelines

Usage:
    from titans.scheduler import get_wsd_schedule

    scheduler = get_wsd_schedule(
        optimizer,
        warmup_steps=2000,
        stable_steps=65500,
        decay_steps=7500,
        max_lr=3e-4,
        min_lr=3e-5,
    )
"""

from __future__ import annotations

import math

import torch
import torch.optim as optim


def get_wsd_schedule(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    max_lr: float = 3e-4,
    min_lr: float = 3e-5,
    last_epoch: int = -1,
) -> optim.lr_scheduler.LambdaLR:
    """Create a WSD (Warmup-Stable-Decay) learning rate scheduler.

    The scheduler returns a multiplicative factor relative to the optimizer's
    base_lr.  Set optimizer lr = max_lr before calling this function.

    Phase boundaries:
        [0, warmup_steps)               → warmup
        [warmup_steps, warmup+stable)   → stable
        [warmup+stable, total)          → cosine decay

    Args:
        optimizer: PyTorch optimizer (should have lr=max_lr as base).
        warmup_steps: Number of linear warmup steps.
        stable_steps: Number of constant-lr steps.
        decay_steps: Number of cosine decay steps.
        max_lr: Peak learning rate.
        min_lr: Final learning rate after decay.
        last_epoch: Passed to LambdaLR for resuming.

    Returns:
        LambdaLR scheduler.
    """
    assert max_lr > 0, f"max_lr must be positive, got {max_lr}"
    min_ratio = min_lr / max_lr
    stable_end = warmup_steps + stable_steps

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            # Linear warmup: 0 → 1.0; warmup_steps=0 skips this phase entirely
            return float(step) / float(warmup_steps)
        elif step < stable_end:
            # Stable phase: constant
            return 1.0
        else:
            # Cosine decay: 1.0 → min_ratio
            decay_progress = float(step - stable_end) / float(max(1, decay_steps))
            decay_progress = min(decay_progress, 1.0)  # clamp past end of decay
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return min_ratio + (1.0 - min_ratio) * cosine_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
