#!/usr/bin/env python3
"""将 knowledge.jsonl 中的新数据同步到 train.jsonl（按 question_id 去重）。

knowledge.jsonl 支持两种格式：

1. 完整格式（和 train.jsonl 一致，直接使用）
2. 简洁格式（只需 history_chunks / question_chunk / answer）：
   {"history_chunks": ["..."], "question_chunk": "...", "answer": "..."}
   脚本会自动补齐 episode_id / question_id / meta。
"""

import json
import argparse
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "generated" / "stage1_timeline_v2"


def _next_knowledge_id(existing_ids: set[str]) -> int:
    """找到 knowledge_NNNNNN 系列的下一个编号。"""
    max_n = -1
    for qid in existing_ids:
        if qid.startswith("knowledge_"):
            parts = qid.split(":")
            try:
                n = int(parts[0].replace("knowledge_", ""))
                max_n = max(max_n, n)
            except ValueError:
                pass
    return max_n + 1


def _normalize_record(rec: dict, idx: int) -> dict:
    """如果缺少 episode_id / question_id / meta，自动补齐。"""
    if "question_id" in rec and "episode_id" in rec:
        # 完整格式，只补 meta 中缺失的关键字段
        meta = rec.get("meta", {})
        meta.setdefault("num_write_steps", len(rec.get("history_chunks", [])))
        meta.setdefault("step_id", len(rec.get("history_chunks", [])))
        rec["meta"] = meta
        return rec

    # 简洁格式，自动生成
    episode_id = f"knowledge_{idx:06d}"
    rec["episode_id"] = episode_id
    rec["question_id"] = f"{episode_id}:q:0:0"
    history = rec.get("history_chunks", [])
    rec["meta"] = {
        "num_steps": len(history) + 1,
        "num_write_steps": len(history),
        "num_query_steps": 1,
        "num_updates": 0,
        "schema_version": "stage1_timeline_v2",
        "step_id": len(history),
        "task_type": "knowledge",
    }
    return rec


def main():
    parser = argparse.ArgumentParser(description="同步 knowledge.jsonl → train.jsonl")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="数据目录")
    parser.add_argument("--dry-run", action="store_true", help="只打印要新增的条目，不写入")
    args = parser.parse_args()

    train_path = args.dir / "train.jsonl"
    knowledge_path = args.dir / "knowledge.jsonl"

    if not knowledge_path.exists():
        print(f"knowledge.jsonl 不存在: {knowledge_path}")
        return

    # 读取 train 中已有的 question_id
    existing_ids = set()
    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                existing_ids.add(rec["question_id"])

    # 读取 knowledge 并补齐字段
    next_idx = _next_knowledge_id(existing_ids)
    raw_records = []
    with open(knowledge_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_records.append(json.loads(line))

    # 先对简洁格式分配 id，再去重
    new_records = []
    for rec in raw_records:
        rec = _normalize_record(rec, next_idx)
        qid = rec["question_id"]
        if qid not in existing_ids:
            new_records.append(rec)
            existing_ids.add(qid)
            if qid.startswith("knowledge_"):
                next_idx += 1

    if not new_records:
        print("没有新数据需要同步。")
        return

    print(f"发现 {len(new_records)} 条新数据:")
    for r in new_records:
        print(f"  {r['question_id']}  {r.get('question_chunk', '')[:50]}")

    if args.dry_run:
        print("(dry-run 模式，未写入)")
        return

    with open(train_path, "a", encoding="utf-8") as f:
        for r in new_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = sum(1 for _ in open(train_path, encoding="utf-8") if _.strip())
    print(f"已追加 {len(new_records)} 条到 {train_path.name}，当前总计 {total} 条。")


if __name__ == "__main__":
    main()
