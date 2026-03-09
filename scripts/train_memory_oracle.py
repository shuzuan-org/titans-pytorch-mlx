#!/usr/bin/env python3
# Copyright 2026 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Memory Oracle 三阶段训练脚本。

模型：Qwen3.5-0.8B + NLM(num_memory_layers=1) + LoRA r=16

训练数据格式（JSONL，由 build_memory_data_v2.py 生成）：
  {"history": [{"role": "write", "content": "...", "importance": 0|1}, ...],
   "query": "...",
   "target_memory": "..."}

  importance 字段内嵌于每个 history turn（1=重要信息，0=噪音），Stage 3 importance loss 直接使用。

训练策略：
  每个样本格式化为一条对话序列：
    [WRITE]msg1 [WRITE]msg2 ... [READ]query [TARGET]target_memory
  NLM 在处理 WRITE tokens 时积累状态，LM loss 只作用于 TARGET tokens。
  每个样本开始前 reset_memory_states()。

三阶段参数差异：
  Stage 1：lr=2e-4，steps=5000，batch=32（单 session 热身，直接事实，无噪音）
  Stage 2：lr=5e-5，steps=10000，batch=16（多 session，混合噪音，跨轮综合）
  Stage 3：lr=1e-5，steps=3000，batch=16（同实体多次更新，+importance loss weight=0.3）

用法（Stage 1，单 H800）：
    uv run python scripts/train_memory_oracle.py \\
        --stage 1 \\
        --data data/oracle_stage1.jsonl \\
        --model Qwen/Qwen3.5-0.8B \\
        --output checkpoints/oracle_stage1

用法（Stage 2，接续）：
    uv run python scripts/train_memory_oracle.py \\
        --stage 2 \\
        --data data/oracle_stage2.jsonl \\
        --resume checkpoints/oracle_stage1/final \\
        --output checkpoints/oracle_stage2

用法（Stage 3）：
    uv run python scripts/train_memory_oracle.py \\
        --stage 3 \\
        --data data/oracle_stage3.jsonl \\
        --resume checkpoints/oracle_stage2/final \\
        --output checkpoints/oracle_stage3
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from titans.config import TitansConfig
from titans.qwen35_injection import (
    disable_memory_write,
    freeze_memory_updates,
    get_trainable_params,
    inject_memory_into_qwen35,
    reset_memory_states,
    unfreeze_memory_updates,
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
    force=True,
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 特殊 token
# ---------------------------------------------------------------------------

# 从 memory_oracle 导入，确保训练/推理格式严格一致。
from titans.memory_oracle import (
    MEMORY_SYS_PROMPT,
    MEMORY_SYS_PROMPT_EN,
    _WRITE_PREFIX,
    _WRITE_PREFIX_EN,
    _READ_PREFIX,
    _READ_PREFIX_EN,
    _TARGET_PREFIX,
    _TARGET_PREFIX_EN,
)

# 按语言索引 markers，键为 --lang 值。
_MARKERS: dict[str, tuple[str, str, str, str]] = {
    "zh": (MEMORY_SYS_PROMPT,    _WRITE_PREFIX,    _READ_PREFIX,    _TARGET_PREFIX),
    "en": (MEMORY_SYS_PROMPT_EN, _WRITE_PREFIX_EN, _READ_PREFIX_EN, _TARGET_PREFIX_EN),
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class OracleDataset(Dataset):
    """Memory Oracle 训练集。

    每个样本包含：
        input_ids:     完整对话序列（write messages + read query + target_memory）
        labels:        Stage 1/2：仅 target_memory token 有真实 ID，其余为 -100。
                       Stage 3：write tokens 也有真实 ID（配合 token_weights 使用）。
        token_weights: （Stage 3 专用）per-token 权重：重要 write entry 为 2.0，
                       非重要为 0.5，target token 及 padding 为 1.0。
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: Any,
        max_seq_len: int = 2048,
        stage: int = 1,
        markers: tuple[str, str, str, str] | None = None,
        split_write_read: bool = False,
    ) -> None:
        """
        Args:
            markers: (sys_prompt, write_prefix, read_prefix, target_prefix).
                     如果为 None，使用中文默认（向后兼容）。
            split_write_read: 若 True，__getitem__ 返回 write_ids / read_ids / read_labels
                     两段张量，供 --no-kv 两阶段训练使用。
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stage = stage
        self.split_write_read = split_write_read
        self.samples = []

        if markers is None:
            markers = _MARKERS["zh"]
        sys_prompt, self._write_prefix, self._read_prefix, self._target_prefix = markers

        # 缓存 sys_prompt 前缀 token IDs，避免在每次 __getitem__ 中重复 encode
        self._sys_ids: list[int] = tokenizer.encode(sys_prompt + "\n", add_special_tokens=True)

        with Path(jsonl_path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        log.info("Loaded %d samples from %s", len(self.samples), jsonl_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        history = sample["history"]
        query = sample["query"]
        target = sample["target_memory"]

        target_ids = self.tokenizer.encode(target, add_special_tokens=False)

        # read_ids 必须始终在 prefix 末尾，单独计算以便截断时保留
        read_ids = self.tokenizer.encode(
            self._read_prefix + query + "\n" + self._target_prefix, add_special_tokens=False
        )

        # 各部分 token 预算：sys + writes + read + target <= max_seq_len
        # target 最多占 1/4；writes 在剩余空间内，超出则丢弃最老的（保留最近）
        keep_target = min(len(target_ids), self.max_seq_len // 4)
        if self.split_write_read:
            # Split 模式：write 和 read 是独立序列，不共享 token 空间。
            # Write pass budget = max_seq_len - sys（无需给 read/target 留空间）。
            write_budget = max(0, self.max_seq_len - len(self._sys_ids))
        else:
            write_budget = max(0, self.max_seq_len - len(self._sys_ids) - len(read_ids) - keep_target)
        target_ids = target_ids[:keep_target]

        if self.stage == 3:
            # Stage 3：逐段 tokenize，精确追踪每个 write entry 的 token 边界。
            write_entries: list[tuple[list[int], int]] = []  # (token_ids, importance 0|1)
            for turn in history:
                ids = self.tokenizer.encode(
                    self._write_prefix + turn["content"] + "\n", add_special_tokens=False
                )
                write_entries.append((ids, int(turn.get("importance", 0))))

            # 丢弃最老的 entry 直到满足 write_budget（保留最近的）
            total_write = sum(len(ids) for ids, _ in write_entries)
            while total_write > write_budget and write_entries:
                removed_ids, _ = write_entries.pop(0)
                total_write -= len(removed_ids)

            write_token_ids = [t for ids, _ in write_entries for t in ids]
        else:
            # Stage 1/2：整体 tokenize，保留跨段 BPE 合并，token 效率更高
            write_text = "".join(self._write_prefix + turn["content"] + "\n" for turn in history)
            all_write_ids = self.tokenizer.encode(write_text, add_special_tokens=False)
            # write_budget 可能为 0（Python `-0` 陷阱：all[-0:] = all），必须显式处理
            if write_budget <= 0:
                write_token_ids = []
            elif len(all_write_ids) > write_budget:
                write_token_ids = all_write_ids[-write_budget:]  # 保留最近的
            else:
                write_token_ids = all_write_ids
            write_entries = None  # Stage 1/2 不需要 per-entry 追踪

        # --no-kv split mode：write 和 read 分成两个独立序列
        if self.split_write_read:
            if self.stage == 3:
                # Stage 3 的 importance-weighted loss 作用在 write tokens 上，
                # 但 split 模式不对 write pass 计算 loss（仅 NLM 写入），
                # 所以 token_weights 被静默丢弃，Stage 3 curriculum 退化为纯 target CE。
                log.warning(
                    "Stage 3 + --no-kv: importance-weighted write loss is disabled; "
                    "only target CE is computed (write tokens go to NLM only)."
                )
            # Write pass: sys + write_tokens（NLM 在 no_grad 中更新 state）
            write_seq = self._sys_ids + write_token_ids
            write_ids_t = torch.tensor(write_seq, dtype=torch.long)

            # Read pass: sys + read_query + target（KV cache 中无 write tokens）
            read_seq = self._sys_ids + read_ids + target_ids
            read_prefix_len = len(self._sys_ids) + len(read_ids)
            read_ids_t = torch.tensor(read_seq, dtype=torch.long)
            read_labels_t = torch.full_like(read_ids_t, -100)
            read_labels_t[read_prefix_len:] = read_ids_t[read_prefix_len:]

            return {"write_ids": write_ids_t, "read_ids": read_ids_t, "read_labels": read_labels_t}

        # 拼装：sys + writes + read（read_ids 始终在末尾，不会被截断）
        prefix_ids = self._sys_ids + write_token_ids + read_ids
        full_ids = prefix_ids + target_ids
        prefix_len = len(prefix_ids)

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = torch.full_like(input_ids, -100)
        result: dict[str, Any] = {"input_ids": input_ids}

        if self.stage == 3 and write_entries is not None:
            # write tokens 纳入 loss，重要 entry 权重 2.0，非重要 0.5
            token_weights = torch.ones(len(full_ids), dtype=torch.float)
            cur = len(self._sys_ids)
            for entry_ids, importance in write_entries:
                actual_end = min(cur + len(entry_ids), prefix_len)
                if actual_end > cur:
                    labels[cur:actual_end] = input_ids[cur:actual_end]
                    token_weights[cur:actual_end] = 2.0 if importance else 0.5
                cur += len(entry_ids)
                if cur >= prefix_len:
                    break
            labels[prefix_len:] = input_ids[prefix_len:]
            result["token_weights"] = token_weights
        else:
            labels[prefix_len:] = input_ids[prefix_len:]

        result["labels"] = labels
        return result


class WeightedConcatDataset(Dataset):
    """多文件加权混合数据集。

    根据 weights 对各子数据集按比例采样，构造一个固定大小的预混洗索引列表。
    总大小 = 所有子数据集样本数之和（自然大小），各语言按 weight 决定占比。
    不满足目标数量时重复采样（小语种），超出时截断（大语种）。

    与 DistributedSampler 完全兼容（本身就是 Dataset）。

    Args:
        datasets: 子 Dataset 列表
        weights:  各子数据集的采样权重（未归一化）；None 表示等权
        seed:     用于复现的随机种子
    """

    def __init__(
        self,
        datasets: list[Dataset],
        weights: list[float] | None = None,
        seed: int = 42,
    ) -> None:
        assert datasets, "datasets must be non-empty"
        if weights is None:
            weights = [1.0] * len(datasets)
        assert len(weights) == len(datasets), "--data-weights count must match --data count"

        total_w = sum(weights)
        norm_w = [w / total_w for w in weights]
        natural_total = sum(len(d) for d in datasets)

        rng = random.Random(seed)
        flat: list[tuple[int, int]] = []  # (dataset_idx, sample_idx)
        for di, (ds, w) in enumerate(zip(datasets, norm_w)):
            # round() 导致各语言目标之和与 natural_total 可能差 ±len(datasets) 个样本，
            # 这对训练无影响，属于故意的近似。
            target = round(natural_total * w)
            n = len(ds)
            if target == 0:
                continue
            if target <= n:
                idxs = rng.sample(range(n), target)
            else:
                full = target // n
                rem  = target % n
                idxs = list(range(n)) * full + rng.sample(range(n), rem)
            flat.extend((di, i) for i in idxs)

        rng.shuffle(flat)
        self._flat = flat
        self._datasets = datasets

        counts = {i: 0 for i in range(len(datasets))}
        for di, _ in flat:
            counts[di] += 1
        log.info(
            "WeightedConcatDataset: total=%d  per-dataset=%s",
            len(flat), [counts[i] for i in range(len(datasets))],
        )

    def __len__(self) -> int:
        return len(self._flat)

    def __getitem__(self, idx: int) -> dict:
        di, si = self._flat[idx]
        return self._datasets[di][si]


def make_collate_fn(pad_id: int):
    """返回一个 collate 函数，使用正确的 pad_id 并生成 attention_mask。

    必须在拿到 tokenizer 后调用以固定 pad_id；不能用全局变量，否则 pad_id
    在 build_model 之前始终是 0（Qwen3.5 的 pad = EOS = 151643）。
    """
    def collate(batch: list[dict]) -> dict[str, Any]:
        B = len(batch)

        # --no-kv split mode
        if "write_ids" in batch[0]:
            max_write = max(item["write_ids"].shape[0] for item in batch)
            max_read  = max(item["read_ids"].shape[0]  for item in batch)

            write_input  = torch.full((B, max_write), pad_id, dtype=torch.long)
            write_mask   = torch.zeros((B, max_write),         dtype=torch.long)
            read_input   = torch.full((B, max_read),  pad_id, dtype=torch.long)
            read_labels  = torch.full((B, max_read),  -100,   dtype=torch.long)
            read_mask    = torch.zeros((B, max_read),          dtype=torch.long)

            for i, item in enumerate(batch):
                wlen = item["write_ids"].shape[0]
                write_input[i, :wlen] = item["write_ids"]
                write_mask[i, :wlen]  = 1
                rlen = item["read_ids"].shape[0]
                read_input[i, :rlen]  = item["read_ids"]
                read_labels[i, :rlen] = item["read_labels"]
                read_mask[i, :rlen]   = 1

            return {
                "write_ids":   write_input,
                "write_mask":  write_mask,
                "read_ids":    read_input,
                "read_labels": read_labels,
                "read_mask":   read_mask,
            }

        # Normal mode
        max_len = max(item["input_ids"].shape[0] for item in batch)

        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        labels    = torch.full((B, max_len), -100,   dtype=torch.long)
        attn_mask = torch.zeros((B, max_len),          dtype=torch.long)

        for i, item in enumerate(batch):
            seq_len = item["input_ids"].shape[0]
            input_ids[i, :seq_len] = item["input_ids"]
            labels[i,    :seq_len] = item["labels"]
            attn_mask[i, :seq_len] = 1  # real tokens = 1, padding = 0

        result: dict[str, Any] = {
            "input_ids":      input_ids,
            "labels":         labels,
            "attention_mask": attn_mask,
        }

        # Stage 3 token_weights（pad 位置填 1.0，不影响 loss — padding labels=-100）
        if "token_weights" in batch[0]:
            token_weights = torch.ones((B, max_len), dtype=torch.float)
            for i, item in enumerate(batch):
                seq_len = item["token_weights"].shape[0]
                token_weights[i, :seq_len] = item["token_weights"]
            result["token_weights"] = token_weights

        return result

    return collate


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_model(
    args: argparse.Namespace,
    tokenizer: Any,
) -> tuple[nn.Module, list[nn.Parameter]]:
    if not HAS_PEFT:
        raise ImportError("peft is required: pip install peft>=0.9.0")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    device = args.device

    resume_dir = Path(args.resume) if args.resume else None

    # ------------------------------------------------------------------
    # 加载基模型（如果从 checkpoint resume，优先从 checkpoint 加载）
    # ------------------------------------------------------------------
    model_src = args.model
    if resume_dir is not None and resume_dir.is_dir() and (resume_dir / "config.json").exists():
        # resume_dir 是 peft 格式的 final/ 目录，基模型用 adapter_config 里的 base_model_name_or_path
        try:
            adapter_cfg_path = resume_dir / "adapter_config.json"
            if adapter_cfg_path.exists():
                adapter_cfg = json.loads(adapter_cfg_path.read_text())
                model_src = adapter_cfg.get("base_model_name_or_path", args.model)
                log.info("Resume: using base model %s from adapter_config", model_src)
        except Exception as e:
            log.warning("Failed to parse adapter_config.json in %s (%s); using --model", resume_dir, e)

    log.info("Loading base model: %s", model_src)
    model = AutoModelForCausalLM.from_pretrained(
        model_src,
        dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
    )

    # ------------------------------------------------------------------
    # NLM 注入（可通过 --no-nlm 跳过，用于 ablation E_A）
    # ------------------------------------------------------------------
    # Qwen3.5 multimodal: hidden_size lives in text_config
    if args.no_nlm:
        log.info("--no-nlm: NLM injection skipped (LoRA-only ablation)")
    else:
        if hasattr(model.config, "text_config"):
            model_dim: int = model.config.text_config.hidden_size
        else:
            model_dim: int = model.config.hidden_size
        mem_cfg = TitansConfig(
            dim=model_dim,
            num_memory_layers=args.num_memory_layers,
            memory_lr=args.memory_lr,
            memory_momentum=args.memory_momentum,
            memory_decay=args.memory_decay,
            use_conv=False,
        )
        log.info("Injecting NLM (dim=%d, num_memory_layers=%d)", model_dim, args.num_memory_layers)
        inject_memory_into_qwen35(model, mem_cfg)
        if args.no_memory_write:
            disable_memory_write(model)
            log.info("--no-memory-write: NLM write disabled (E_C ablation — capacity-only)")

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 可训练参数
    # ------------------------------------------------------------------
    trainable_params = get_trainable_params(model)
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    log.info(
        "Trainable: %s / %s (%.2f%%)",
        f"{n_trainable:,}", f"{n_total:,}",
        100.0 * n_trainable / n_total,
    )

    return model, trainable_params


def _load_resume(
    model: nn.Module,
    resume_dir: Path,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    """从 resume_dir 加载权重：支持 peft final/ 格式和 step_*.pt。

    Args:
        optimizer: 若提供，且 checkpoint 含 optimizer_state，则一并恢复。
                   peft 格式无 optimizer state，此参数在 peft 路径下无效。

    Returns:
        恢复的 opt_step（peft 格式无 step 信息，返回 0）。
    """
    # 先尝试 peft 格式（adapter_config.json 在 resume_dir 本身）
    if (resume_dir / "adapter_config.json").exists():
        adapter_state_path = resume_dir / "adapter_model.safetensors"
        if not adapter_state_path.exists():
            adapter_state_path = resume_dir / "adapter_model.bin"
        if not adapter_state_path.exists():
            raise RuntimeError(
                f"adapter_config.json found in {resume_dir} but no weights file "
                "(expected adapter_model.safetensors or adapter_model.bin)"
            )
        try:
            from safetensors.torch import load_file as load_safetensors
            adapter_state = load_safetensors(str(adapter_state_path), device=device)
        except ImportError:
            adapter_state = torch.load(adapter_state_path, map_location=device, weights_only=True)
        # Use peft's own API to handle naming conventions (adapter name injection, etc.)
        from peft import set_peft_model_state_dict
        result = set_peft_model_state_dict(model, adapter_state, adapter_name="default")
        unexpected = getattr(result, "unexpected_keys", [])
        missing = getattr(result, "missing_keys", [])
        if unexpected:
            log.warning("Unexpected keys when loading adapter: %s", unexpected[:5])
        if missing:
            log.warning("Missing keys when loading adapter: %s", missing[:5])
        n_loaded = len(adapter_state) - len(unexpected)
        log.info("Loaded %d params (missing=%d) from peft adapter: %s",
                 n_loaded, len(missing), resume_dir)
        # Cross-stage resume intentionally resets opt_step and optimizer state.
        # Each stage has a different target LR (Stage1=2e-4 → Stage2=5e-5 → Stage3=1e-5);
        # carrying Stage N's AdamW second-moment estimates (v) into Stage N+1 would
        # corrupt the effective step size for the entire warmup phase, since v is
        # calibrated to the previous LR scale. Fresh optimizer start is correct here.
        return 0

    # Fallback: step_*.pt (keys are saved from peft model.named_parameters() — exact match)
    step_files = sorted(resume_dir.glob("step_*.pt"))
    if step_files:
        ckpt_path = step_files[-1]
        log.info("Resuming from checkpoint: %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        all_params = dict(model.named_parameters())
        loaded = 0
        for key in ("memory_state", "lora_state"):
            for name, tensor in ckpt.get(key, {}).items():
                if name in all_params:
                    all_params[name].data.copy_(tensor.to(all_params[name].device))
                    loaded += 1
        if optimizer is not None and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            log.info("Optimizer state restored from %s", ckpt_path)
        restored_step = ckpt.get("step", 0)
        log.info("Loaded %d params from step checkpoint: %s (step=%d)",
                 loaded, ckpt_path, restored_step)
        return restored_step
    else:
        raise RuntimeError(
            f"--resume specified ({resume_dir}) but no loadable checkpoint found "
            "(no adapter_config.json, no .safetensors/.bin, and no step_*.pt)"
        )


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def get_lr(step: int, max_lr: float, min_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Stage 3：token-level importance weighted CE loss
# ---------------------------------------------------------------------------


def weighted_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_weights: torch.Tensor,
) -> torch.Tensor:
    """计算带 token-level 权重的 cross-entropy loss。

    用于 Stage 3：重要 history entry 的 token 权重 2.0，
    非重要 entry 权重 0.5，其余（query/target）权重 1.0。

    这让 NLM 的 proj_k/proj_v/gate_lr 对重要 token 看到更强的梯度信号，
    从而学会对重要内容产生更大的记忆更新（参数训练效果），是一个可微分的
    importance supervision，区别于 Stage 2 的均匀 CE。

    Args:
        logits:        (B, T, V)  模型输出 logits
        labels:        (B, T)     标签，padding / prefix 位置为 -100
        token_weights: (B, T)     per-token 权重，float

    Returns:
        标量 loss
    """
    # next-token prediction: shift by 1
    shift_logits  = logits[..., :-1, :].contiguous()   # (B, T-1, V)
    shift_labels  = labels[..., 1:].contiguous()        # (B, T-1)
    shift_weights = token_weights[..., 1:].contiguous() # (B, T-1)

    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    per_token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )  # (B*(T-1),)

    w = shift_weights.view(-1)
    valid = shift_labels.view(-1) != -100
    # weighted mean over valid positions
    weighted_sum = (per_token_loss * w)[valid].sum()
    denom = w[valid].sum().clamp(min=1e-8)
    return weighted_sum / denom


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
    """保存 checkpoint（memory_state + lora_state + optimizer_state）。"""
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
    payload: dict = {
        "step": step,
        "loss": loss,
        "memory_state": memory_state,
        "lora_state": lora_state,
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(payload, ckpt_path)
    log.info("Saved checkpoint → %s (loss=%.4f)", ckpt_path, loss)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    model: nn.Module,
    trainable_params: list[nn.Parameter],
    loader: DataLoader,
    args: argparse.Namespace,
    is_main: bool = True,
    resume_dir: Path | None = None,
) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    min_lr = args.lr * 0.1

    # Optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer: torch.optim.Optimizer = bnb.optim.AdamW8bit(
                trainable_params, lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95)
            )
            log.info("Using 8-bit AdamW")
        except ImportError:
            log.warning("bitsandbytes not found, using fp32 AdamW")
            optimizer = torch.optim.AdamW(
                trainable_params, lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95)
            )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params, lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95)
        )

    # Load weights + optimizer state; restore opt_step so LR schedule and
    # checkpoint naming continue from the correct position.
    opt_step = 0
    if resume_dir is not None:
        opt_step = _load_resume(model, resume_dir, device, optimizer=optimizer)
        log.info("Resuming training from step %d", opt_step)
        if opt_step >= args.max_steps:
            raise RuntimeError(
                f"Resumed from step {opt_step} but --max-steps={args.max_steps} "
                "is not greater than the resume point. Training would exit immediately "
                "and overwrite the existing checkpoint. Pass a larger --max-steps."
            )
    micro_step = 0
    micro_loss = 0.0
    log_loss = 0.0
    step_loss = float("nan")

    model.train()
    optimizer.zero_grad()
    t0 = time.time()

    def _infinite_loader():
        epoch = 0
        while True:
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)
            for batch in loader:
                yield batch
            epoch += 1

    for batch in _infinite_loader():
        if opt_step >= args.max_steps:
            break

        # 每个 batch 每个样本独立 reset（memory 不跨样本泄漏）
        reset_memory_states(model)

        if args.no_kv and "write_ids" in batch:
            # --no-kv 两阶段：
            # Pass 1（write）：no_grad 下全序列 forward，NLM 内部 enable_grad 仍工作，
            #   积累 memory state；KV cache 不保留（不传 past_key_values）。
            # Pass 2（read）：freeze NLM（只读），loss 仅在 target tokens，
            #   梯度流回 LoRA + NLM 结构参数（proj_q / proj_out）。
            write_ids  = batch["write_ids"].to(device,  non_blocking=True)
            write_mask = batch["write_mask"].to(device, non_blocking=True)
            read_ids   = batch["read_ids"].to(device,   non_blocking=True)
            read_lbls  = batch["read_labels"].to(device, non_blocking=True)
            read_mask  = batch["read_mask"].to(device,  non_blocking=True)

            with torch.no_grad():
                model(input_ids=write_ids, attention_mask=write_mask)

            freeze_memory_updates(model)
            outputs = model(input_ids=read_ids, attention_mask=read_mask, labels=read_lbls)
            # Unfreeze before backward: safe because autograd graph is fully recorded
            # during forward; backward replays the recorded graph and does not
            # re-inspect _nlm_frozen at backward time.
            unfreeze_memory_updates(model)
            lm_loss = outputs.loss
        else:
            input_ids   = batch["input_ids"].to(device,   non_blocking=True)
            labels      = batch["labels"].to(device,      non_blocking=True)
            attn_mask   = batch["attention_mask"].to(device, non_blocking=True)

            if args.stage == 3 and "token_weights" in batch:
                # Stage 3：token-level importance weighted CE。
                # 模型 forward 不传 labels（避免内部 mean CE），用 weighted_ce_loss 手动算。
                outputs = model(input_ids=input_ids, attention_mask=attn_mask)
                tw = batch["token_weights"].to(device, non_blocking=True)
                lm_loss = weighted_ce_loss(outputs.logits, labels, tw)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                lm_loss = outputs.loss

        true_loss  = lm_loss.item()
        total_loss = lm_loss / args.grad_accum
        total_loss.backward()
        micro_loss += true_loss
        micro_step += 1

        if micro_step % args.grad_accum != 0:
            continue

        # Multi-GPU: average gradients across ranks BEFORE clipping
        # (clip after reduce so the norm threshold applies to the averaged gradient)
        if args.multi_gpu:
            for p in trainable_params:
                if p.grad is not None:
                    torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)

        nn.utils.clip_grad_norm_(trainable_params, 1.0)

        lr = get_lr(opt_step, args.lr, min_lr, args.warmup_steps, args.max_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.step()
        optimizer.zero_grad()

        step_loss = micro_loss / args.grad_accum
        log_loss += step_loss
        micro_loss = 0.0
        opt_step += 1

        if opt_step % 10 == 0:
            elapsed = time.time() - t0
            log.info(
                "step %d/%d | loss %.4f | lr %.2e | %.1fs/10steps",
                opt_step, args.max_steps, log_loss / 10, lr, elapsed,
            )
            log_loss = 0.0
            t0 = time.time()

        if opt_step % args.save_steps == 0 and is_main:
            save_checkpoint(model, optimizer, opt_step, step_loss, output_dir)

    log.info("Training done at step %d (last loss=%.4f)", opt_step, step_loss)
    if is_main:
        save_checkpoint(model, optimizer, opt_step, step_loss, output_dir)
        model.save_pretrained(str(output_dir / "final"))
        # oracle_config.json：记录推理时必须一致的 marker 语言。
        # MemoryOracle.from_pretrained() 读取此文件自动选择正确 markers，
        # 避免 training/inference marker mismatch（曾导致 NLM 完全失效）。
        oracle_cfg = {
            "lang": args.lang,
            "stage": args.stage,
            "use_nlm": not args.no_nlm,
            "memory_write": not getattr(args, "no_memory_write", False),
            "no_kv_training": getattr(args, "no_kv", False),
        }
        with open(output_dir / "final" / "oracle_config.json", "w") as _f:
            json.dump(oracle_cfg, _f, indent=2)
        log.info("Saved peft model → %s/final", output_dir)
        log.info("oracle_config.json: lang=%s stage=%d use_nlm=%s memory_write=%s",
                 args.lang, args.stage, oracle_cfg["use_nlm"], oracle_cfg["memory_write"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Memory Oracle training (3 stages)")

    p.add_argument("--stage", type=int, choices=[1, 2, 3], default=1)
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--data", required=True, nargs="+",
                   help="JSONL training data file(s). Multiple files → mixed training.")
    p.add_argument("--data-weights", nargs="+", type=float, default=None,
                   help="Per-file sampling weights (parallel to --data). "
                        "Defaults to equal weights. Example: --data-weights 2 1 1 for 2:1:1 ratio.")
    p.add_argument("--output", required=True, help="Output checkpoint directory")
    p.add_argument("--resume", default=None,
                   help="Resume from: peft final/ dir or step_*.pt file/dir")

    # Training
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override max steps (defaults: Stage1=5000, 2=10000, 3=3000)")
    p.add_argument("--lr", type=float, default=None,
                   help="Override learning rate (defaults: Stage1=2e-4, 2=5e-5, 3=1e-5)")
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch size (defaults: Stage1=32, 2=16, 3=16)")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    # LoRA
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-target", default="q_proj,k_proj,v_proj,o_proj")

    # Memory
    p.add_argument("--no-nlm", action="store_true",
                   help="Skip NLM injection (LoRA-only ablation baseline, E_A)")
    p.add_argument("--no-memory-write", action="store_true",
                   help="Inject NLM but disable write (capacity-only ablation, E_C)")
    p.add_argument("--no-kv", action="store_true",
                   help="Two-pass training: write pass (no_grad, NLM accumulates state) "
                        "then read pass (NLM frozen, loss on target tokens only). "
                        "Forces model to rely on NLM rather than KV cache of write tokens.")
    p.add_argument("--num-memory-layers", type=int, default=1)
    p.add_argument("--memory-lr", type=float, default=0.1)
    p.add_argument("--memory-momentum", type=float, default=0.9)
    p.add_argument("--memory-decay", type=float, default=0.01)

    # Hardware
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--use-8bit-adam", action="store_true")
    p.add_argument("--multi-gpu", action="store_true",
                   help="Enable multi-GPU DDP (launch with torchrun --nproc_per_node=N)")
    p.add_argument("--lang", default=None, choices=["zh", "en"],
                   help="Marker language: en=[WRITE]/[QUERY]/[MEMORY], zh=【写入】/【查询】/【记忆】. "
                        "Required when using multiple --data files. "
                        "Single-file default: zh (backward compatible).")

    return p.parse_args()


# Stage defaults
_STAGE_DEFAULTS = {
    1: {"lr": 2e-4, "max_steps": 5000,  "batch_size": 32},
    2: {"lr": 5e-5, "max_steps": 10000, "batch_size": 16},
    3: {"lr": 1e-5, "max_steps": 3000,  "batch_size": 16},
}


def main() -> None:
    args = parse_args()

    # Apply stage defaults for any unspecified params
    defaults = _STAGE_DEFAULTS[args.stage]
    if args.lr is None:
        args.lr = defaults["lr"]
    if args.max_steps is None:
        args.max_steps = defaults["max_steps"]
    if args.batch_size is None:
        args.batch_size = defaults["batch_size"]

    # 确定 markers 语言
    if args.lang is None:
        if len(args.data) > 1:
            raise ValueError(
                "--lang is required when using multiple --data files. "
                "For multilingual mixed training use --lang en."
            )
        args.lang = "zh"  # 单文件向后兼容默认值

    markers = _MARKERS[args.lang]
    log.info(
        "Language: %s  markers=%s/%s/%s",
        args.lang.upper(), markers[1].strip(), markers[2].strip(), markers[3].strip(),
    )

    # Resume lang consistency check: warn if previous stage used a different language.
    # Training Stage N with lang=zh then Stage N+1 with lang=en causes marker mismatch
    # (NLM writes Chinese markers into memory, eval reads English markers → garbage output).
    if args.resume:
        _resume_path = Path(args.resume)
        # oracle_config.json lives in final/; also check resume path itself for flexibility
        for _cfg_dir in (_resume_path, _resume_path / "final"):
            _cfg_path = _cfg_dir / "oracle_config.json"
            if _cfg_path.exists():
                try:
                    _prev_cfg = json.loads(_cfg_path.read_text())
                    _prev_lang = _prev_cfg.get("lang")
                    if _prev_lang and _prev_lang != args.lang:
                        log.warning(
                            "LANG MISMATCH: previous checkpoint used lang=%s, "
                            "current training uses lang=%s. "
                            "NLM memory markers will be inconsistent — "
                            "this likely causes garbage eval output. "
                            "Pass --lang %s to match the previous stage, "
                            "or confirm this is intentional.",
                            _prev_lang, args.lang, _prev_lang,
                        )
                except Exception:
                    pass
                break

    log.info("=== Memory Oracle Stage %d ===", args.stage)
    log.info("  model=%s  lr=%.1e  steps=%d  batch=%d  accum=%d",
             args.model, args.lr, args.max_steps, args.batch_size, args.grad_accum)

    # Multi-GPU DDP init
    is_main = True
    if args.multi_gpu:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
        is_main = (torch.distributed.get_rank() == 0)
        if not is_main:
            # suppress verbose logs on non-main ranks
            logging.getLogger().setLevel(logging.WARNING)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Dataset（单文件或多文件加权混合）
    split_wr = args.no_kv and not args.no_nlm  # no-kv 无 NLM 无意义，静默降级
    if args.no_kv and args.no_nlm:
        log.warning("--no-kv has no effect when --no-nlm is set; ignoring --no-kv")
    if len(args.data) == 1:
        dataset: Dataset = OracleDataset(
            args.data[0], tokenizer, args.seq_len, args.stage,
            markers=markers, split_write_read=split_wr,
        )
    else:
        sub_datasets = [
            OracleDataset(p, tokenizer, args.seq_len, args.stage,
                          markers=markers, split_write_read=split_wr)
            for p in args.data
        ]
        dataset = WeightedConcatDataset(sub_datasets, args.data_weights, seed=args.seed)
        log.info("Mixed training: %d files → %d total samples", len(args.data), len(dataset))

    # 用 factory 生成 collate_fn，固定正确的 pad_id（Qwen3.5 eos = 151643）
    pad_id = tokenizer.pad_token_id
    if args.multi_gpu:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=make_collate_fn(pad_id),
            pin_memory=True,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=make_collate_fn(pad_id),
            pin_memory=(args.device != "cpu"),
            drop_last=True,
        )

    # Model
    model, trainable_params = build_model(args, tokenizer)

    # Train (pass resume_dir so optimizer state can be restored after optimizer is created)
    resume_dir = Path(args.resume) if args.resume else None
    train(model, trainable_params, loader, args, is_main=is_main, resume_dir=resume_dir)


if __name__ == "__main__":
    main()
