#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from urllib import error, request

from usecase.locomobech.common import CATEGORY_NAMES, extract_plain_turns, load_locomo10, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run titans stage1 service on LoCoMo.")
    parser.add_argument("--data-file", default="data/dataset/locomo/locomo10.json")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="results/locomobech/titans_predictions.json")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-qa", type=int, default=None)
    return parser.parse_args()


def request_json(base_url: str, path: str, payload: dict | None = None, method: str = "POST") -> dict:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = request.Request(
        base_url.rstrip("/") + "/" + path.lstrip("/"),
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with request.urlopen(req, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {payload}") from exc


def truncate_answer(text: str) -> str:
    text = text.strip()
    for stop in ("\n\n", "\nQuestion", "\nQ:", "Answer:"):
        index = text.find(stop)
        if index != -1:
            text = text[:index].strip()
    return text


def main() -> None:
    args = parse_args()
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": args.temperature > 0,
    }
    data = load_locomo10(args.data_file)
    output = {}
    qa_count = 0
    for conv_idx, item in enumerate(data):
        session_id = f"locomo_stage1_{conv_idx}"
        turns = extract_plain_turns(item["conversation"])
        request_json(
            args.base_url,
            "/v1/memory/write",
            {"session_id": session_id, "contents": turns},
        )
        rows = []
        for qa in item.get("qa", []):
            category = int(qa.get("category", 0))
            if category == 5:
                continue
            question = str(qa.get("question", "")).strip()
            answer = str(qa.get("answer", "")).strip()
            response = request_json(
                args.base_url,
                "/v1/chat/respond",
                {
                    "session_id": session_id,
                    "query": question,
                    "generation_config": generation_config,
                },
            )
            rows.append(
                {
                    "question": question,
                    "answer": answer,
                    "response": truncate_answer(str(response.get("answer", ""))),
                    "category": CATEGORY_NAMES.get(category, str(category)),
                    "profile": response.get("profile", {}),
                }
            )
            qa_count += 1
            if args.max_qa and qa_count >= args.max_qa:
                break
        output[str(conv_idx)] = rows
        request_json(args.base_url, f"/v1/sessions/{session_id}", method="DELETE")
        if args.max_qa and qa_count >= args.max_qa:
            break
    save_json(args.output, output)
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()


