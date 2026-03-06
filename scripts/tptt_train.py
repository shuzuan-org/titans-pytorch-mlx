#!/usr/bin/env python3
# Copyright 2026 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
TPTT fine-tuning: Qwen2.5-7B + NeuralLongTermMemory (MAL) + LoRA.

Trainable params:
    - NeuralLongTermMemory projection/gate params (memory structural weights)
    - LoRA adapters on Attn + FFN layers of the base Qwen model

Base model weights are frozen. MemoryMLP layer weights are updated by the
Titans mechanism during forward pass (not by the optimizer).

Usage (H800 single GPU, smoke test):
    uv run python scripts/tptt_train.py \\
        --model Qwen/Qwen2.5-7B --data /data/tokens/ \\
        --max-steps 20 --batch-size 2 --output checkpoints/tptt_test

Usage (CPU smoke test with small model):
    uv run python scripts/tptt_train.py \\
        --model Qwen/Qwen2.5-0.5B --data /data/tokens/ \\
        --max-steps 5 --batch-size 1 --seq-len 64 --device cpu
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from titans.config import TitansConfig
from titans.data import BinaryTokenDataset
from titans.tptt import (
    get_trainable_params,
    inject_memory_into_qwen,
    reset_memory_states,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model

    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    force=True,  # override any pre-existing root logger config (transformers/peft may set handlers)
    handlers=[logging.StreamHandler(sys.stdout)],  # stdout is unbuffered with PYTHONUNBUFFERED=1
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TPTT fine-tuning with LoRA + memory")

    # Model / data
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--data", required=True, help="Directory with shard_*.bin files")
    p.add_argument("--output", default="checkpoints/tptt_qwen25_7b")

    # Sequence / batch
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)

    # Training schedule
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=500)

    # LoRA
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument(
        "--lora-target",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    # Memory
    p.add_argument("--num-memory-layers", type=int, default=1)
    p.add_argument("--memory-lr", type=float, default=0.1)
    p.add_argument("--memory-momentum", type=float, default=0.9)
    p.add_argument("--memory-decay", type=float, default=0.01)

    # Hardware
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--use-8bit-adam",
        action="store_true",
        help="Use bitsandbytes 8-bit AdamW to reduce optimizer memory by ~4×",
    )

    # Resume
    p.add_argument(
        "--resume",
        default=None,
        metavar="CKPT",
        help="Path to a step_*.pt checkpoint to resume training from",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Learning rate schedule (cosine with linear warmup, no external deps)
# ---------------------------------------------------------------------------


def get_lr(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int,
) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"step_{step:07d}.pt"

    memory_state = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if ".memory." in name and ".memory.memory." not in name
    }
    lora_state = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if "lora_" in name
    }

    torch.save(
        {
            "step": step,
            "loss": loss,
            "memory_state": memory_state,
            "lora_state": lora_state,
            "optimizer_state": optimizer.state_dict(),
        },
        ckpt_path,
    )
    log.info("Saved checkpoint → %s", ckpt_path)


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def build_model(args: argparse.Namespace) -> tuple[nn.Module, list[nn.Parameter]]:
    if not HAS_PEFT:
        raise ImportError("peft is required. Run: pip install peft>=0.9.0")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    device = args.device

    log.info("Loading base model: %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map={"": device},
    )

    # Infer dim from model config
    model_dim: int = model.config.hidden_size
    log.info("Model dim: %d", model_dim)

    # Build memory config matching base model dimensions
    mem_cfg = TitansConfig(
        dim=model_dim,
        num_memory_layers=args.num_memory_layers,
        memory_lr=args.memory_lr,
        memory_momentum=args.memory_momentum,
        memory_decay=args.memory_decay,
        # Disable conv for simpler injection (can be enabled later)
        use_conv=False,
    )

    log.info("Injecting NeuralLongTermMemory (num_memory_layers=%d)", args.num_memory_layers)
    inject_memory_into_qwen(model, mem_cfg)

    # Apply LoRA to Attn + FFN projections of the base Qwen layers
    target_modules = [m.strip() for m in args.lora_target.split(",")]
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)

    # Enable memory structural params in addition to LoRA
    trainable_params = get_trainable_params(model)

    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    log.info(
        "Trainable params: %s / %s (%.2f%%)",
        f"{n_trainable:,}",
        f"{n_total:,}",
        100.0 * n_trainable / n_total,
    )

    return model, trainable_params


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def build_dataset(args: argparse.Namespace) -> DataLoader:
    dataset = BinaryTokenDataset.from_directory(
        args.data,
        seq_len=args.seq_len,
        seed=args.seed,
    )
    log.info(
        "Dataset: %d sequences, %s tokens total",
        len(dataset),
        f"{dataset.total_tokens:,}",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device != "cpu"),
        drop_last=True,
    )
    return loader


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    model: nn.Module,
    trainable_params: list[nn.Parameter],
    loader: DataLoader,
    args: argparse.Namespace,
) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    min_lr = args.lr * 0.1

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                trainable_params,
                lr=args.lr,
                weight_decay=0.1,
                betas=(0.9, 0.95),
            )
            log.info("Using 8-bit AdamW (bitsandbytes)")
        except ImportError:
            log.warning("bitsandbytes not found, falling back to fp32 AdamW")
            optimizer = torch.optim.AdamW(
                trainable_params, lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )

    micro_step = 0      # total micro-steps processed
    opt_step = 0        # optimizer updates applied
    # log_loss: sum of true (unscaled) losses for the current 10-step window
    log_loss = 0.0
    # step_loss: average true loss over the last completed optimizer step
    step_loss = float("nan")
    # micro_loss: accumulator within the current grad_accum window
    micro_loss = 0.0

    # --- Resume from checkpoint ---
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        log.info("Resuming from %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

        # Restore model params by name
        all_named = dict(model.named_parameters())
        for state_key in ("memory_state", "lora_state"):
            for name, tensor in ckpt.get(state_key, {}).items():
                if name in all_named:
                    all_named[name].data.copy_(tensor.to(all_named[name].device))

        # Restore optimizer state (AdamW momentum buffers)
        optimizer.load_state_dict(ckpt["optimizer_state"])

        opt_step = ckpt["step"]
        micro_step = opt_step * args.grad_accum
        step_loss = ckpt.get("loss", float("nan"))
        log.info("Resumed: opt_step=%d, last_loss=%.4f", opt_step, step_loss)

    model.train()
    optimizer.zero_grad()
    t0 = time.time()

    for batch in loader:
        if opt_step >= args.max_steps:
            break

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        # Each micro-batch is an independent sequence: reset memory state
        reset_memory_states(model)

        # Forward pass — Qwen2ForCausalLM computes cross-entropy when labels given
        outputs = model(input_ids=input_ids, labels=labels)
        true_loss = outputs.loss.item()       # unscaled, for logging
        scaled_loss = outputs.loss / args.grad_accum

        scaled_loss.backward()
        micro_loss += true_loss
        micro_step += 1

        if micro_step % args.grad_accum != 0:
            continue

        # --- Optimizer step ---
        nn.utils.clip_grad_norm_(trainable_params, 1.0)

        lr = get_lr(opt_step, args.lr, min_lr, args.warmup_steps, args.max_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.step()
        optimizer.zero_grad()

        step_loss = micro_loss / args.grad_accum   # avg true loss this opt step
        log_loss += step_loss
        micro_loss = 0.0
        opt_step += 1

        # Logging every 10 opt steps
        if opt_step % 10 == 0:
            elapsed = time.time() - t0
            log.info(
                "step %d / %d | loss %.4f | lr %.2e | %.1f s/10steps",
                opt_step,
                args.max_steps,
                log_loss / 10,    # true average loss over the last 10 steps
                lr,
                elapsed,
            )
            log_loss = 0.0
            t0 = time.time()

        # Checkpoint
        if opt_step % args.save_steps == 0:
            save_checkpoint(model, optimizer, opt_step, step_loss, output_dir)

    # Final save — step_loss holds the last opt-step's true average loss
    log.info("Training complete at step %d (last loss %.4f)", opt_step, step_loss)
    save_checkpoint(model, optimizer, opt_step, step_loss, output_dir)

    # Save peft model (LoRA adapters in HF format)
    model.save_pretrained(str(output_dir / "final"))
    log.info("Saved final peft model → %s/final", output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device != "cpu":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    model, trainable_params = build_model(args)
    loader = build_dataset(args)
    train(model, trainable_params, loader, args)


if __name__ == "__main__":
    main()
