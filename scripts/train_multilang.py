#!/usr/bin/env python3
# Copyright 2026 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
多语言混合训练流水线。

功能：
  1. 轮询等待各语言各阶段数据就绪（达到目标量的 90%）
  2. 用 torchrun 启动加权混合三阶段训练

用法：
    nohup python scripts/train_multilang.py > logs/train_multilang.log 2>&1 &

或指定自定义路径：
    python scripts/train_multilang.py \\
        --data-root /data/memory \\
        --ckpt-root checkpoints/oracle_multilang \\
        --model /models/Qwen3.5-0.8B \\
        --gpus 0,1,2,3,4,5,6 \\
        --port 29504
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 数据规划
# ---------------------------------------------------------------------------

# 各语言三阶段目标样本数 (stage1, stage2, stage3)
TARGETS: dict[str, tuple[int, int, int]] = {
    "zh": (5000, 10000, 3000),
    "en": (5000, 10000, 3000),
    "ja": (3000,  5000, 2000),
    "ko": (3000,  5000, 2000),
    "fr": (3000,  5000, 2000),
    "es": (3000,  5000, 2000),
    "de": (3000,  5000, 2000),
    "ar": (2000,  3000, 1000),
    "ru": (2000,  3000, 1000),
}

# 混合训练采样权重（未归一化）
WEIGHTS: dict[str, float] = {
    "zh": 25, "en": 25,
    "ja": 10, "ko": 10,
    "fr": 8,  "es": 8, "de": 8,
    "ar": 3,  "ru": 3,
}

# 就绪阈值：达到目标的 90% 即视为就绪
READY_RATIO = 0.90

# 三阶段训练超参
STAGE_HPS: dict[int, dict] = {
    1: {"lr": "1e-4",  "max_steps": 5000,  "save_steps": 1000, "seq_len": 2048},
    2: {"lr": "5e-5",  "max_steps": 10000, "save_steps": 2000, "seq_len": 4096},
    3: {"lr": "2e-5",  "max_steps": 3000,  "save_steps": 600,  "seq_len": 4096},
}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for _ in f)


def data_ready(data_root: Path, stage: int, langs: list[str]) -> tuple[bool, dict[str, int]]:
    """检查指定阶段的所有语言数据是否就绪。

    Returns:
        (all_ready, {lang: count})
    """
    counts: dict[str, int] = {}
    all_ready = True
    for lang in langs:
        path = data_root / f"memory_{lang}" / f"stage{stage}.jsonl"
        n = count_lines(path)
        counts[lang] = n
        target = TARGETS[lang][stage - 1]
        if n < target * READY_RATIO:
            all_ready = False
    return all_ready, counts


def wait_for_data(
    data_root: Path,
    stage: int,
    langs: list[str],
    poll_sec: int = 120,
    max_stall_sec: int = 3600,
) -> None:
    """阻塞直到所有语言的 stage{N} 数据就绪。

    如果数据量在 max_stall_sec 秒内没有任何增长，判定为生成进程已死，抛出异常。
    """
    log.info("=== 等待 Stage %d 数据就绪 ===", stage)
    last_total = -1
    stall_since = time.monotonic()

    while True:
        ready, counts = data_ready(data_root, stage, langs)
        current_total = sum(counts.values())

        lines = []
        for lang in langs:
            target = TARGETS[lang][stage - 1]
            n = counts[lang]
            mark = "✓" if n >= target * READY_RATIO else "…"
            lines.append(f"{lang}={n}/{target}{mark}")
        log.info("Stage %d: %s", stage, "  ".join(lines))

        if ready:
            log.info("Stage %d 数据全部就绪。", stage)
            return

        if current_total > last_total:
            last_total = current_total
            stall_since = time.monotonic()
        elif time.monotonic() - stall_since > max_stall_sec:
            raise RuntimeError(
                f"Stage {stage} 数据生成已停滞超过 {max_stall_sec // 60} 分钟（总量={current_total}）。"
                "请检查生成进程是否仍在运行。"
            )

        time.sleep(poll_sec)


def build_data_args(data_root: Path, stage: int, langs: list[str]) -> list[str]:
    """生成 --data file1 file2 ... --data-weights w1 w2 ... 参数列表。"""
    data_args: list[str] = ["--data"]
    weight_args: list[str] = ["--data-weights"]
    for lang in langs:
        path = data_root / f"memory_{lang}" / f"stage{stage}.jsonl"
        data_args.append(str(path))
        weight_args.append(str(WEIGHTS[lang]))
    return data_args + weight_args


def run_stage(
    stage: int,
    data_root: Path,
    ckpt_root: Path,
    langs: list[str],
    args: argparse.Namespace,
) -> None:
    """用 torchrun 启动单阶段训练，阻塞直到完成。"""
    hps = STAGE_HPS[stage]
    output_dir = ckpt_root / f"stage{stage}"
    log_path = Path(args.log_dir) / f"train_multilang_s{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # resume: Stage 2/3 从上一阶段 final 恢复
    resume_args: list[str] = []
    if stage > 1:
        prev_final = ckpt_root / f"stage{stage - 1}" / "final"
        if not prev_final.exists():
            raise RuntimeError(
                f"Stage {stage} 需要 Stage {stage - 1} 的 checkpoint，"
                f"但 {prev_final} 不存在。Stage {stage - 1} 是否成功完成？"
            )
        resume_args = ["--resume", str(prev_final)]

    n_gpus = len(args.gpus.split(","))

    cmd = [
        args.torchrun,
        f"--nproc_per_node={n_gpus}",
        f"--master_port={args.port}",
        args.train_script,
        "--stage", str(stage),
        "--lang", "en",           # 所有语言统一使用 EN markers
        "--model", args.model,
        "--output", str(output_dir),
        "--max-steps",  str(hps["max_steps"]),
        "--save-steps", str(hps["save_steps"]),
        "--lr", hps["lr"],
        "--lora-rank", str(args.lora_rank),
        "--num-memory-layers", "1",
        "--batch-size", "1",
        "--grad-accum", "4",
        "--seq-len", str(hps["seq_len"]),
        "--use-8bit-adam",
        "--multi-gpu",
        *build_data_args(data_root, stage, langs),
        *resume_args,
    ]

    log.info("=== Stage %d 训练启动 ===", stage)
    log.info("CMD: %s", " ".join(cmd))
    log.info("LOG: %s", log_path)

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus}

    # start_new_session=True：torchrun 进独立 session（新 process group）。
    # 这样 os.killpg 只打 torchrun 及其 worker，不会把 pipeline 自己也杀掉。
    proc: subprocess.Popen | None = None
    ret = -1
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, text=True, bufsize=1,
            start_new_session=True,
        )
        with log_path.open("w") as lf:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                lf.write(line)
        ret = proc.wait()
    except BaseException:
        # Ctrl+C、SIGTERM、日志文件打开失败等 — 确保 torchrun 及其 worker 都被终止
        if proc is not None and proc.poll() is None:
            log.warning("Stage %d 训练被中断，正在终止 torchrun (PID=%d)...", stage, proc.pid)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                log.warning("SIGTERM 超时，强制 SIGKILL (PID=%d)", proc.pid)
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
        raise

    if ret != 0:
        raise RuntimeError(f"Stage {stage} 训练失败（exit code={ret}）。详见 {log_path}")

    log.info("=== Stage %d 训练完成，checkpoint → %s/final ===", stage, output_dir)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    base = Path(__file__).parent.parent  # 项目根目录

    p = argparse.ArgumentParser(description="多语言混合训练流水线")
    p.add_argument("--data-root",    default=str(base / "data"),
                   help="多语言数据根目录（含 memory_zh/, memory_en/ 等子目录）")
    p.add_argument("--ckpt-root",    default=str(base / "checkpoints" / "oracle_multilang"),
                   help="Checkpoint 输出根目录")
    p.add_argument("--log-dir",      default=str(base / "logs"),
                   help="训练日志目录")
    p.add_argument("--model",        default="/home/shuzuan/models/Qwen/Qwen3___5-0.8B",
                   help="基础模型路径或 HF hub ID")
    p.add_argument("--gpus",         default="0,1,2,3,4,5,6,7",
                   help="CUDA_VISIBLE_DEVICES（逗号分隔）")
    p.add_argument("--port",         type=int, default=29504,
                   help="torchrun master port")
    p.add_argument("--lora-rank",    type=int, default=16)
    p.add_argument("--poll-sec",     type=int, default=120,
                   help="数据就绪轮询间隔（秒）")
    p.add_argument("--stages",       nargs="+", type=int, default=[1, 2, 3],
                   choices=[1, 2, 3], help="要运行的阶段，默认全部")
    p.add_argument("--torchrun",
                   default="/home/shuzuan/miniconda3/envs/sglang/bin/torchrun")
    p.add_argument("--train-script",
                   default=str(base / "scripts" / "train_memory_oracle.py"))
    # 评测参数（stage3 完成后自动运行 LoCoMo 评测）
    p.add_argument("--no-eval",      action="store_true",
                   help="跳过训练后评测")
    p.add_argument("--eval-data",    default=str(base / "data" / "locomo10.json"),
                   help="LoCoMo 评测数据集路径")
    p.add_argument("--eval-output",  default=str(base / "results" / "locomo_multilang.json"),
                   help="评测结果输出路径")
    p.add_argument("--judge",        default="MiniMax-Text-01",
                   help="LLM-as-Judge 模型名")
    p.add_argument("--judge-base-url", default="https://mini.origintask.cn/v1",
                   help="Judge API base URL")
    p.add_argument("--judge-api-key",  default="shuzuan2025-minimax",
                   help="Judge API key")
    p.add_argument("--eval-workers-per-gpu", type=int, default=4,
                   help="每张 GPU 上并行的 eval worker 数（默认 4）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    ckpt_root = Path(args.ckpt_root)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    langs = list(TARGETS.keys())

    log.info("=== 多语言混合训练流水线启动 ===")
    log.info("语言: %s", langs)
    log.info("权重: %s", {k: WEIGHTS[k] for k in langs})
    log.info("阶段: %s", args.stages)
    log.info("Checkpoint: %s", ckpt_root)

    for stage in sorted(args.stages):
        # 等待该阶段数据就绪（最长停滞 2 小时无进展则 abort）
        wait_for_data(data_root, stage, langs, poll_sec=args.poll_sec, max_stall_sec=7200)

        # 打印就绪后的汇总
        _, counts = data_ready(data_root, stage, langs)
        total = sum(counts.values())
        log.info("Stage %d 数据汇总: total=%d  %s",
                 stage, total,
                 "  ".join(f"{l}={counts[l]}" for l in langs))

        # 启动训练
        run_stage(stage, data_root, ckpt_root, langs, args)

    log.info("=== 全流水线完成 ===")
    log.info("最终 Checkpoint: %s/stage3/final", ckpt_root)

    if 3 in args.stages and not args.no_eval:
        run_eval(ckpt_root, args)


def run_eval(ckpt_root: Path, args: argparse.Namespace) -> None:
    """Stage3 训练完成后并行运行 LoCoMo 评测（每张 GPU 一个子进程）。"""
    import json

    oracle_path = ckpt_root / "stage3" / "final"
    if not oracle_path.exists():
        log.warning("Stage3 checkpoint 不存在，跳过评测：%s", oracle_path)
        return

    base = Path(__file__).parent.parent
    output_path = Path(args.eval_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gpus = [g.strip() for g in args.gpus.split(",")]
    wpg = args.eval_workers_per_gpu
    n_workers = len(gpus) * wpg

    env_base = {**os.environ, "OPENAI_API_KEY": args.judge_api_key}
    eval_script = str(base / "scripts" / "eval_locomo.py")

    log.info("=== 开始 LoCoMo 评测（%d GPU × %d workers = %d 路并行）===",
             len(gpus), wpg, n_workers)
    log.info("Oracle: %s", oracle_path)

    # 每个 worker 写到独立的 shard 文件，最后合并
    shard_paths = [
        output_path.parent / f"{output_path.stem}_shard{i}{output_path.suffix}"
        for i in range(n_workers)
    ]

    procs: list[subprocess.Popen] = []
    for i, shard in enumerate(shard_paths):
        gpu = gpus[i % len(gpus)]   # 轮询分配 GPU
        cmd = [
            sys.executable, eval_script,
            "--oracle",         str(oracle_path),
            "--data",           args.eval_data,
            "--output",         str(shard),
            "--judge",          args.judge,
            "--judge-base-url", args.judge_base_url,
            "--device",         "cuda:0",   # CUDA_VISIBLE_DEVICES 已限定单卡
            "--shard",          str(i),
            "--total-shards",   str(n_workers),
        ]
        env = {**env_base, "CUDA_VISIBLE_DEVICES": gpu}
        log.info("Worker %d GPU=%s: shard %d/%d", i, gpu, i, n_workers)
        procs.append(subprocess.Popen(cmd, env=env))

    # 等所有 worker 完成
    failed = False
    for i, p in enumerate(procs):
        ret = p.wait()
        if ret != 0:
            log.error("Worker %d 失败（exit=%d）", i, ret)
            failed = True

    if failed:
        log.error("部分评测 worker 失败，结果可能不完整")

    # 合并 shard 结果
    all_results: list[dict] = []
    for shard in shard_paths:
        if shard.exists():
            try:
                data = json.loads(shard.read_text())
                # 支持结果是列表或含 "results" 字段的 dict
                if isinstance(data, list):
                    all_results.extend(data)
                elif isinstance(data, dict):
                    all_results.extend(data.get("results", [data]))
            except Exception as e:
                log.warning("合并 shard %s 失败：%s", shard, e)

    if all_results:
        scores = [r.get("score", 0) for r in all_results if "score" in r]
        avg = sum(scores) / len(scores) if scores else 0.0
        merged = {"results": all_results, "avg_score": avg, "n": len(all_results)}
        output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
        log.info("=== 评测完成：avg_score=%.3f（%d 题），结果→%s ===",
                 avg, len(all_results), output_path)
    else:
        log.error("无有效评测结果可合并")


if __name__ == "__main__":
    main()
