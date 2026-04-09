#!/usr/bin/env python3
"""
TimeQA → Stage1 训练数据转化脚本

流程:
1. 分别翻译 train.json / dev.json / test.json 为中文知识点
2. train → train.jsonl, dev+test → eval.jsonl
3. 输出格式与 stage1_timeline_v2 完全一致，可直接用于训练

前置条件:
  conda activate sglang
  python -m sglang.launch_server \
      --model-path /home/shuzuan/models/MiniMax-M2.5 \
      --tp 4 --host 0.0.0.0 --port 30000 --trust-remote-code

使用:
  # 一步到位: 翻译 train/dev/test + 组装
  python usecase/convert_timeqa_to_stage1.py all \
      --input-dir data/generated/TimeQA \
      --output-dir data/generated/timeqa_stage1 \
      --api-url http://localhost:30000

  # 只翻译某个 split
  python usecase/convert_timeqa_to_stage1.py translate \
      --input data/generated/TimeQA/train.json \
      --output data/generated/TimeQA/train_zh.jsonl \
      --api-url http://localhost:30000

  # 只组装 (翻译完成后)
  python usecase/convert_timeqa_to_stage1.py assemble \
      --input-dir data/generated/TimeQA \
      --output-dir data/generated/timeqa_stage1
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm


TRANSLATE_PROMPT = """\
将以下英文问答对转化为中文知识点，直接输出JSON，禁止任何分析或思考过程。

格式:
{{"history": "中文陈述句(含时间人物事件)", "question": "中文问题", "answer": "中文答案"}}

英文问题: {question}
英文答案: {answer}"""


def load_timeqa(path: str) -> list[dict]:
    """加载 TimeQA 数据，过滤空 targets。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    valid = []
    for record in data:
        targets = record.get("targets", [])
        targets = [t for t in targets if t.strip()]
        if not targets:
            continue
        valid.append({
            "idx": record["idx"],
            "question": record["question"],
            "targets": targets,
            "level": record.get("level", ""),
        })
    return valid


def call_sglang_single(
    url: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> str:
    """调用 sglang OpenAI 兼容 API（单条）。"""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            if attempt == 2:
                return ""
            time.sleep(2)
    return ""


def call_sglang_batch(
    api_url: str,
    prompts: list[str],
    max_tokens: int = 1024,
    temperature: float = 0.3,
    max_workers: int = 32,
) -> list[str]:
    """并发调用 sglang API。"""
    url = f"{api_url}/v1/chat/completions"
    results = [""] * len(prompts)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(call_sglang_single, url, prompt, max_tokens, temperature): i
            for i, prompt in enumerate(prompts)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = ""

    return results


def parse_llm_output(text: str) -> dict | None:
    """从 LLM 输出中解析 JSON。"""
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if all(k in obj for k in ("history", "question", "answer")):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[^{}]*\"history\"[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if all(k in obj for k in ("history", "question", "answer")):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def _record_key(record: dict) -> str:
    """生成记录的唯一标识（idx 可能重复，需要加 question 区分）。"""
    return f"{record['idx']}||{record.get('question', record.get('original_question', ''))}"


def translate_records(
    records: list[dict],
    api_url: str,
    output_path: str,
    batch_size: int = 64,
    split_label: str = "",
) -> list[dict]:
    """翻译 TimeQA 记录为中文知识点。支持断点续传。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_keys: set[str] = set()
    existing_results: list[dict] = []
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                existing_keys.add(_record_key(obj))
                existing_results.append(obj)
        print(f"[{split_label}] 已有 {len(existing_keys)} 条，断点续传")

    todo = [r for r in records if _record_key(r) not in existing_keys]
    print(f"[{split_label}] 需翻译: {len(todo)}, 已完成: {len(existing_keys)}, 总计: {len(records)}")

    if not todo:
        return existing_results

    results = list(existing_results)
    failed = 0

    pbar = tqdm(total=len(todo), desc=f"翻译 {split_label}", unit="条")

    with open(output_path, "a", encoding="utf-8") as fout:
        for i in range(0, len(todo), batch_size):
            batch = todo[i : i + batch_size]
            prompts = [
                TRANSLATE_PROMPT.format(
                    question=r["question"],
                    answer=", ".join(r["targets"]),
                )
                for r in batch
            ]

            responses = call_sglang_batch(api_url, prompts)

            for record, response in zip(batch, responses):
                parsed = parse_llm_output(response)
                if parsed is None:
                    failed += 1
                    pbar.update(1)
                    tqdm.write(f"  [失败] idx={record['idx']} q={record['question'][:60]}")
                    if response:
                        tqdm.write(f"         raw={response[:200]}")
                    else:
                        tqdm.write(f"         raw=(空响应)")
                    continue
                result = {
                    "idx": record["idx"],
                    "history": parsed["history"],
                    "question": parsed["question"],
                    "answer": parsed["answer"],
                    "level": record["level"],
                    "original_question": record["question"],
                    "original_targets": record["targets"],
                }
                results.append(result)
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                pbar.update(1)

            fout.flush()
            pbar.set_postfix({"成功": len(results) - len(existing_results), "失败": failed})

    pbar.close()
    print(f"[{split_label}] 翻译完成: 成功 {len(results)}, 失败 {failed}")
    return results


def load_knowledge(path: str) -> list[dict]:
    """加载已翻译的知识点 JSONL。"""
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def assemble_split(
    knowledge: list[dict],
    output_path: Path,
    split_name: str,
    min_distractors: int = 1,
    max_distractors: int = 9,
    seed: int = 42,
) -> None:
    """将一组知识点组装成 stage1 训练数据。

    每条知识点都会作为目标问答出现一次。
    额外随机填充 min_distractors~max_distractors 条干扰 history，
    目标 history 和干扰 history 随机打乱顺序。
    最终每条数据有 (1 + num_distractors) 条 history_chunks。
    """
    rng = random.Random(seed)
    samples = []

    indices = list(range(len(knowledge)))
    rng.shuffle(indices)

    for target_idx in indices:
        target = knowledge[target_idx]
        # 随机选 1~9 条干扰 history
        num_distractors = rng.randint(min_distractors, max_distractors)
        distractor_pool = [i for i in range(len(knowledge)) if i != target_idx]
        if num_distractors > len(distractor_pool):
            num_distractors = len(distractor_pool)
        distractor_indices = rng.sample(distractor_pool, num_distractors)

        all_indices = distractor_indices + [target_idx]
        rng.shuffle(all_indices)
        history_chunks = [knowledge[i]["history"] for i in all_indices]

        episode_id = f"timeqa_{split_name}_{len(samples):06d}"
        sample = {
            "episode_id": episode_id,
            "question_id": f"{episode_id}:q:0:0",
            "history_chunks": history_chunks,
            "question_chunk": target["question"],
            "answer": target["answer"],
            "meta": {
                "schema_version": "timeqa_stage1",
                "source_idx": target["idx"],
                "level": target["level"],
                "num_history_chunks": len(history_chunks),
                "num_distractors": num_distractors,
            },
        }
        samples.append(sample)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"{split_name}: {len(samples)} samples → {output_path}")


# ── CLI commands ──────────────────────────────────────────────


def cmd_translate(args: argparse.Namespace) -> None:
    records = load_timeqa(args.input)
    print(f"加载 {len(records)} 条有效记录")
    split_label = Path(args.input).stem  # train / dev / test
    translate_records(records, args.api_url, args.output, args.batch_size, split_label)


def cmd_assemble(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # train_zh.jsonl → train.jsonl
    train_zh = input_dir / "train_zh.jsonl"
    if not train_zh.exists():
        raise FileNotFoundError(f"找不到 {train_zh}，请先运行 translate")
    train_knowledge = load_knowledge(str(train_zh))
    print(f"train 知识点: {len(train_knowledge)}")
    assemble_split(
        train_knowledge, output_dir / "train.jsonl", "train",
        min_distractors=args.min_distractors, max_distractors=args.max_distractors, seed=args.seed,
    )

    # dev_zh.jsonl + test_zh.jsonl → eval.jsonl
    eval_knowledge = []
    for name in ("dev_zh.jsonl", "test_zh.jsonl"):
        zh_path = input_dir / name
        if zh_path.exists():
            loaded = load_knowledge(str(zh_path))
            print(f"{name} 知识点: {len(loaded)}")
            eval_knowledge.extend(loaded)
    if eval_knowledge:
        assemble_split(
            eval_knowledge, output_dir / "eval.jsonl", "eval",
            min_distractors=args.min_distractors, max_distractors=args.max_distractors, seed=args.seed + 1,
        )
    else:
        print("警告: 没有找到 dev/test 翻译文件，跳过 eval.jsonl 生成")


def cmd_all(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 分别翻译 train / dev / test
    for split in ("train", "dev", "test"):
        src = input_dir / f"{split}.json"
        if not src.exists():
            print(f"跳过 {src} (不存在)")
            continue
        records = load_timeqa(str(src))
        print(f"\n[{split}] 加载 {len(records)} 条有效记录")
        zh_path = str(input_dir / f"{split}_zh.jsonl")
        translate_records(records, args.api_url, zh_path, args.batch_size, split)

    # 组装
    print("\n开始组装训练数据...")
    # train_zh → train.jsonl
    train_zh = input_dir / "train_zh.jsonl"
    if train_zh.exists():
        train_knowledge = load_knowledge(str(train_zh))
        assemble_split(
            train_knowledge, output_dir / "train.jsonl", "train",
            min_distractors=args.min_distractors, max_distractors=args.max_distractors, seed=args.seed,
        )

    # dev_zh + test_zh → eval.jsonl
    eval_knowledge = []
    for name in ("dev_zh.jsonl", "test_zh.jsonl"):
        zh_path = input_dir / name
        if zh_path.exists():
            loaded = load_knowledge(str(zh_path))
            eval_knowledge.extend(loaded)
    if eval_knowledge:
        assemble_split(
            eval_knowledge, output_dir / "eval.jsonl", "eval",
            min_distractors=args.min_distractors, max_distractors=args.max_distractors, seed=args.seed + 1,
        )

    print(f"\n完成! 输出目录: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TimeQA → Stage1 训练数据转化")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # translate: 翻译单个 split
    p_translate = subparsers.add_parser("translate", help="翻译单个 TimeQA split")
    p_translate.add_argument("--input", required=True, help="TimeQA JSON 文件路径 (如 train.json)")
    p_translate.add_argument("--output", required=True, help="输出 JSONL 路径 (如 train_zh.jsonl)")
    p_translate.add_argument("--api-url", default="http://localhost:30000")
    p_translate.add_argument("--batch-size", type=int, default=64)

    # assemble: 组装训练数据 (需要已翻译完)
    p_assemble = subparsers.add_parser("assemble", help="组装训练数据")
    p_assemble.add_argument("--input-dir", required=True, help="TimeQA 目录 (含 *_zh.jsonl)")
    p_assemble.add_argument("--output-dir", required=True, help="输出目录")
    p_assemble.add_argument("--min-distractors", type=int, default=1)
    p_assemble.add_argument("--max-distractors", type=int, default=9)
    p_assemble.add_argument("--seed", type=int, default=42)

    # all: 翻译全部 + 组装
    p_all = subparsers.add_parser("all", help="翻译 train/dev/test + 组装")
    p_all.add_argument("--input-dir", required=True, help="TimeQA 目录 (含 train/dev/test.json)")
    p_all.add_argument("--output-dir", required=True, help="输出目录")
    p_all.add_argument("--api-url", default="http://localhost:30000")
    p_all.add_argument("--batch-size", type=int, default=64)
    p_all.add_argument("--min-distractors", type=int, default=1)
    p_all.add_argument("--max-distractors", type=int, default=9)
    p_all.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.command == "translate":
        cmd_translate(args)
    elif args.command == "assemble":
        cmd_assemble(args)
    elif args.command == "all":
        cmd_all(args)


if __name__ == "__main__":
    main()
