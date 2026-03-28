#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib import error, request
from urllib.parse import urljoin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one stage1 eval usecase through HTTP service.")
    parser.add_argument(
        "--data-file",
        default="data/generated/stage1_timeline_v2/eval.jsonl",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--session-id", default="stage1-eval-demo")
    parser.add_argument("--base-url", default="http://111.6.70.85:10115")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=int, default=180)
    return parser.parse_args()


def load_sample(data_file: str, index: int) -> dict[str, Any]:
    path = Path(data_file)
    with path.open("r", encoding="utf-8") as handle:
        for current_index, line in enumerate(handle):
            if current_index == index:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("eval sample must be a JSON object")
                return payload
    raise IndexError(f"sample index {index} out of range for {data_file}")


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        urljoin(base_url.rstrip("/") + "/", path.lstrip("/")),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return read_json(req, timeout=timeout)


def delete_json(base_url: str, path: str, timeout: int) -> dict[str, Any]:
    req = request.Request(
        urljoin(base_url.rstrip("/") + "/", path.lstrip("/")),
        method="DELETE",
    )
    return read_json(req, timeout=timeout)


def read_json(req: request.Request, timeout: int) -> dict[str, Any]:
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {payload}") from exc


def main() -> None:
    args = parse_args()
    sample = load_sample(args.data_file, args.index)
    history_chunks = sample.get("history_chunks")
    question_chunk = sample.get("question_chunk")
    gold_answer = sample.get("answer")

    if not isinstance(history_chunks, list) or not all(isinstance(item, str) for item in history_chunks):
        raise ValueError("history_chunks must be a list of strings")
    if not isinstance(question_chunk, str) or not question_chunk:
        raise ValueError("question_chunk must be a non-empty string")
    if not isinstance(gold_answer, str):
        raise ValueError("answer must be a string")

    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": args.temperature > 0,
    }

    post_json(
        args.base_url,
        "/v1/memory/write",
        {
            "session_id": args.session_id,
            "contents": history_chunks,
        },
        timeout=args.request_timeout,
    )
    with_memory = post_json(
        args.base_url,
        "/v1/chat/respond",
        {
            "session_id": args.session_id,
            "query": question_chunk,
            "generation_config": generation_config,
        },
        timeout=args.request_timeout,
    )

    direct_qwen = post_json(
        args.base_url,
        "/v1/chat/respond",
        {
            "session_id": args.session_id,
            "query": question_chunk,
            "mode": "direct_backbone",
            "generation_config": generation_config,
        },
        timeout=args.request_timeout,
    )

    delete_json(args.base_url, f"/v1/sessions/{args.session_id}", timeout=args.request_timeout)
    without_memory = post_json(
        args.base_url,
        "/v1/chat/respond",
        {
            "session_id": args.session_id,
            "query": question_chunk,
            "generation_config": generation_config,
        },
        timeout=args.request_timeout,
    )

    print(f"Base URL: {args.base_url}")
    print(f"Request timeout: {args.request_timeout}s")
    print(f"Sample index: {args.index}")
    print(f"Question: {question_chunk}")
    print(f"Gold / 标准答案: {gold_answer}")
    print(f"With memory: {with_memory['answer']}")
    print(f"Without memory: {without_memory['answer']}")
    print(f"Direct Qwen: {direct_qwen['answer']}")


if __name__ == "__main__":
    main()
