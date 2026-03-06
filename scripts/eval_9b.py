#!/usr/bin/env python3
"""Qwen3.5-9B Memory Oracle 评测（200 样本，F1 比较）

用法：
    python scripts/eval_9b.py

固定路径：
    BASE_MODEL  = /home/shuzuan/models/Qwen/Qwen3___5-9B
    CKPT_DIR    = /home/shuzuan/prj/titans-pytorch-mlx/checkpoints/oracle_9b/stage1
    EVAL_DATA   = /home/shuzuan/prj/titans-pytorch-mlx/data/oracle_eval.jsonl
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# 路径配置
# ────────────────────────────────────────────────────────────────────────────
BASE_MODEL = "/home/shuzuan/models/Qwen/Qwen3___5-9B"
CKPT_DIR   = "/home/shuzuan/prj/titans-pytorch-mlx/checkpoints/oracle_9b/stage1"
EVAL_DATA  = "/home/shuzuan/prj/titans-pytorch-mlx/data/oracle_eval.jsonl"
DEVICE     = "cuda:2"
MAX_SAMPLES = 200

# ────────────────────────────────────────────────────────────────────────────
# F1 计算（词级别，与 SQuAD 一致）
# ────────────────────────────────────────────────────────────────────────────

def _f1(pred: str, gold: str) -> float:
    # 字符级 F1（适用于中文），忽略空格和标点
    import re
    def _chars(s):
        return list(re.sub(r'\s+', '', s))  # 去空格，保留汉字/数字/字母
    pc = _chars(pred)
    gc = _chars(gold)
    if not pc or not gc:
        return float(pred.strip() == gold.strip())
    # 计算最长公共子序列长度作为 overlap（更准确于 set）
    from collections import Counter
    pc_cnt = Counter(pc)
    gc_cnt = Counter(gc)
    common = sum((pc_cnt & gc_cnt).values())
    if not common:
        return 0.0
    p = common / len(pc)
    r = common / len(gc)
    return 2 * p * r / (p + r)


# ────────────────────────────────────────────────────────────────────────────
# 加载数据
# ────────────────────────────────────────────────────────────────────────────

def load_eval_data(path: str, max_samples: int = MAX_SAMPLES):
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if len(samples) >= max_samples:
                break
    logger.info("Loaded %d samples from %s", len(samples), path)
    return samples


# ────────────────────────────────────────────────────────────────────────────
# 主评测逻辑
# ────────────────────────────────────────────────────────────────────────────

def main():
    import sys
    sys.path.insert(0, "/home/shuzuan/prj/titans-pytorch-mlx/src")

    from titans.memory_oracle import MemoryOracle

    samples = load_eval_data(EVAL_DATA)

    # ── 加载 Oracle（带 NLM checkpoint）────────────────────────────────────
    logger.info("Loading MemoryOracle (9B)…")
    oracle = MemoryOracle.from_pretrained(
        checkpoint_dir=CKPT_DIR,
        base_model=BASE_MODEL,
        device=DEVICE,
        num_memory_layers=1,
        memory_lr=0.1,
        memory_momentum=0.9,
        memory_decay=0.01,
        max_write_tokens=2048,
        max_read_new_tokens=200,
        read_temperature=0.3,
    )
    # Truncate from LEFT so 【记忆】 marker at end is always preserved
    oracle.tokenizer.truncation_side = "left"
    logger.info("Oracle loaded. memory_stats: %s", oracle.memory_stats())

    # Qwen3.5 thinking mode: add <think> token (id=248068) as EOS to stop early.
    # Also patch read() to prepend /no_think directive.
    _THINK_ID = 248068
    _orig_read = oracle.read.__func__  # unbound method

    import types, torch as _torch

    def _read_no_think(self, query: str) -> str:
        from titans.memory_oracle import (
            _SYS_PROMPT, MEMORY_WRITE_TEMPLATE, MEMORY_READ_TEMPLATE,
            _WRITE_PREFIX, _READ_PREFIX, _TARGET_PREFIX,
        )
        from titans.qwen35_injection import reset_memory_states, unfreeze_memory_updates
        reset_memory_states(self.model)
        unfreeze_memory_updates(self.model)

        # 不用 /no_think prefix（不稳定），改用 bad_words_ids 直接屏蔽 <think> token
        prefix = _SYS_PROMPT + "\n"
        for msg in self._write_buffer:
            prefix += MEMORY_WRITE_TEMPLATE.format(message=msg)
        prefix += MEMORY_READ_TEMPLATE.format(query=query)

        inputs = self.tokenizer(
            prefix, return_tensors="pt", add_special_tokens=True,
            truncation=True, max_length=self.max_write_tokens,
        ).to(self.device)

        with _torch.no_grad():
            output_ids = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=self.max_read_new_tokens,
                temperature=self.read_temperature,
                do_sample=(self.read_temperature > 0),
                pad_token_id=self.tokenizer.eos_token_id,
                bad_words_ids=[[_THINK_ID]],  # 禁止生成 <think>，直接生成摘要
                repetition_penalty=1.3,
            )

        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        for stop in (_WRITE_PREFIX, _READ_PREFIX, _TARGET_PREFIX, "\n\n", "<think>"):
            idx = text.find(stop)
            if idx != -1:
                text = text[:idx].strip()
        return text

    oracle.read = types.MethodType(_read_no_think, oracle)

    # ── 评测两种模式：with-write vs no-write ───────────────────────────────
    f1_write_list  = []
    f1_nowrite_list = []

    for idx, sample in enumerate(samples):
        history     = sample.get("history", sample.get("conversation", []))
        query       = sample.get("query", "")
        gold_answer = sample.get("target_memory", sample.get("answer", ""))

        # --- With NLM writes ---
        oracle.reset()
        for turn in history:
            text = turn.get("content", "")
            oracle.write(text)
        pred_write = oracle.read(query)
        f1_w = _f1(pred_write, gold_answer)
        f1_write_list.append(f1_w)

        # --- No-write baseline (empty buffer) ---
        oracle.reset()
        pred_nowrite = oracle.read(query)
        f1_nw = _f1(pred_nowrite, gold_answer)
        f1_nowrite_list.append(f1_nw)

        if (idx + 1) % 20 == 0 or idx == 0:
            avg_w  = sum(f1_write_list)  / len(f1_write_list)
            avg_nw = sum(f1_nowrite_list) / len(f1_nowrite_list)
            logger.info(
                "[%3d/%d] running F1: write=%.3f  no-write=%.3f  Δ=%.3f",
                idx + 1, len(samples), avg_w, avg_nw, avg_w - avg_nw,
            )
            logger.info("  sample query : %s", query[:80])
            logger.info("  gold answer  : %s", gold_answer[:80])
            logger.info("  pred (write) : %s", pred_write[:80])
            logger.info("  pred (nowr.) : %s", pred_nowrite[:80])

    # ── 最终汇总 ─────────────────────────────────────────────────────────────
    final_w  = sum(f1_write_list)  / len(f1_write_list)
    final_nw = sum(f1_nowrite_list) / len(f1_nowrite_list)
    delta    = final_w - final_nw

    print("\n" + "=" * 60)
    print(f"Qwen3.5-9B Memory Oracle 评测结果（{len(samples)} 样本）")
    print(f"  Checkpoint : {CKPT_DIR}")
    print(f"  F1 (with NLM writes) : {final_w:.4f}")
    print(f"  F1 (no-write baseline): {final_nw:.4f}")
    print(f"  Δ (NLM 贡献)         : {delta:+.4f}")
    print("=" * 60)

    # 对比 7B 基准
    print("\n参考（Qwen2.5-7B, 200样本）：")
    print("  F1 (with writes) = 0.6740  baseline = 0.1844  Δ = +0.4896")


if __name__ == "__main__":
    main()
