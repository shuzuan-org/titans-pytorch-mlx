#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


TRAIN_CONV_RANGE = (0, 8)
EVAL_CONV_RANGE = (8, 10)
SKIP_CATEGORIES = {5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert locomo10.json into stage1 train/eval jsonl files."
    )
    parser.add_argument(
        "--data-file",
        default="data/dataset/locomo/locomo10.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data/generated/locomo10_stage1",
    )
    parser.add_argument("--max-turns", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_locomo(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("locomo data must be a list")
    return payload


def ordered_sessions(conversation: dict[str, Any]) -> list[tuple[str, list[Any], str | None]]:
    sessions: list[tuple[str, list[Any], str | None]] = []
    for index in range(1, 100):
        session_key = f"session_{index}"
        if session_key not in conversation:
            break
        turns = conversation.get(session_key)
        if not isinstance(turns, list):
            continue
        date_time = conversation.get(f"{session_key}_date_time")
        sessions.append((session_key, turns, date_time if isinstance(date_time, str) else None))
    return sessions


def format_turn(turn: Any) -> str | None:
    if isinstance(turn, str):
        text = turn.strip()
        return text or None
    if not isinstance(turn, dict):
        return None
    text = str(turn.get("text", "")).strip()
    if not text:
        return None
    speaker = str(turn.get("speaker", "")).strip()
    dia_id = str(turn.get("dia_id", "")).strip()
    prefix = f"{speaker}: " if speaker else ""
    suffix = f" [{dia_id}]" if dia_id else ""
    return f"{prefix}{text}{suffix}"


def extract_history_chunks(conversation: dict[str, Any], max_turns: int) -> list[str]:
    chunks: list[str] = []
    for session_key, turns, date_time in ordered_sessions(conversation):
        if date_time:
            chunks.append(f"[{session_key} | {date_time}]")
        else:
            chunks.append(f"[{session_key}]")
        for turn in turns:
            text = format_turn(turn)
            if text:
                chunks.append(text)
    if len(chunks) <= max_turns:
        return chunks
    return chunks[-max_turns:]


def build_samples(
    conversations: list[dict[str, Any]],
    conv_range: tuple[int, int],
    max_turns: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for conv_idx in range(*conv_range):
        item = conversations[conv_idx]
        sample_id = str(item.get("sample_id", f"locomo-{conv_idx}"))
        history_chunks = extract_history_chunks(item["conversation"], max_turns=max_turns)
        qa_list = item.get("qa", [])
        if not isinstance(qa_list, list):
            continue
        for qa_index, qa in enumerate(qa_list):
            if not isinstance(qa, dict):
                continue
            category = int(qa.get("category", 0))
            if category in SKIP_CATEGORIES:
                continue
            question = str(qa.get("question", "")).strip()
            answer = str(qa.get("answer", "")).strip()
            if not question or not answer:
                continue
            evidence = qa.get("evidence", [])
            samples.append(
                {
                    "episode_id": sample_id,
                    "question_id": f"{sample_id}:q{qa_index}",
                    "history_chunks": history_chunks,
                    "question_chunk": question,
                    "answer": answer,
                    "meta": {
                        "source_dataset": "snap-research/locomo",
                        "locomo_conv_idx": conv_idx,
                        "locomo_category": category,
                        "locomo_evidence": evidence if isinstance(evidence, list) else [],
                        "num_history_chunks": len(history_chunks),
                    },
                }
            )
    return samples


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(name: str, rows: list[dict[str, Any]]) -> None:
    categories = Counter(row["meta"]["locomo_category"] for row in rows)
    print(name, "samples=", len(rows), "categories=", dict(sorted(categories.items())))
    if rows:
        sample = rows[0]
        print(name, "question_id=", sample["question_id"])
        print(name, "history_chunks=", len(sample["history_chunks"]))
        print(name, "question=", sample["question_chunk"])
        print(name, "answer=", sample["answer"])


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    conversations = load_locomo(args.data_file)
    train_rows = build_samples(conversations, TRAIN_CONV_RANGE, args.max_turns)
    eval_rows = build_samples(conversations, EVAL_CONV_RANGE, args.max_turns)
    random.shuffle(train_rows)

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "eval.jsonl", eval_rows)

    print(f"loaded_conversations={len(conversations)}")
    print(f"output_dir={output_dir}")
    summarize("train", train_rows)
    summarize("eval", eval_rows)


if __name__ == "__main__":
    main()
