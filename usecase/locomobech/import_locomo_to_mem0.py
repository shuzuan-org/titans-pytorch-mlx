#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib import error, request

from usecase.locomobech.common import load_locomo10, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import LoCoMo into mem0 using mem0-style flow.")
    parser.add_argument("--data-file", default="data/dataset/locomo/locomo10.json")
    parser.add_argument("--base-url", default="http://58.211.6.130:10281")
    parser.add_argument("--output-dir", default="results/locomobech/mem0_import")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--request-delay", type=float, default=0.5)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    return parser.parse_args()


def post_json(base_url: str, path: str, payload: dict) -> dict:
    req = request.Request(
        base_url.rstrip("/") + "/" + path.lstrip("/"),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def make_user_ids(conversation: dict[str, Any], conv_idx: int) -> tuple[str, str]:
    speaker_a = str(conversation.get("speaker_a", f"speaker_a_{conv_idx}")).strip() or f"speaker_a_{conv_idx}"
    speaker_b = str(conversation.get("speaker_b", f"speaker_b_{conv_idx}")).strip() or f"speaker_b_{conv_idx}"
    return f"{speaker_a}_{conv_idx}", f"{speaker_b}_{conv_idx}"


def build_session_messages(conversation: dict[str, Any], session_key: str, speaker_a: str, speaker_b: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    chats = conversation.get(session_key, [])
    forward_messages: list[dict[str, str]] = []
    reverse_messages: list[dict[str, str]] = []
    for chat in chats:
        if not isinstance(chat, dict):
            continue
        text = str(chat.get("text", "")).strip()
        speaker = str(chat.get("speaker", "")).strip()
        if not text:
            continue
        if speaker == speaker_a:
            forward_messages.append({"role": "user", "content": f"{speaker_a}: {text}"})
            reverse_messages.append({"role": "assistant", "content": f"{speaker_a}: {text}"})
        elif speaker == speaker_b:
            forward_messages.append({"role": "assistant", "content": f"{speaker_b}: {text}"})
            reverse_messages.append({"role": "user", "content": f"{speaker_b}: {text}"})
    return forward_messages, reverse_messages


def chunk_messages(messages: list[dict[str, str]], batch_size: int) -> list[list[dict[str, str]]]:
    if len(messages) <= batch_size:
        return [messages] if messages else []
    chunks = []
    for index in range(0, len(messages), batch_size):
        chunk = messages[index:index + batch_size]
        if chunk and chunk[0]["role"] == "assistant" and chunks:
            prev = chunks[-1]
            for j in range(len(prev) - 1, -1, -1):
                if prev[j]["role"] == "user":
                    chunk = [prev[j]] + chunk
                    break
        if chunk:
            chunks.append(chunk)
    return chunks


def import_for_user(
    base_url: str,
    user_id: str,
    session_id: str,
    api: str,
    model: str,
    messages: list[dict[str, str]],
    batch_size: int,
    request_delay: float,
    skip_existing: bool,
) -> list[dict[str, Any]]:
    chunks = chunk_messages(messages, batch_size)
    results = []
    for chunk_idx, chunk in enumerate(chunks):
        try:
            result = post_json(
                base_url,
                f"/api/v1/users/{user_id}/imports/messages",
                {
                    "messages": chunk,
                    "session_id": f"{session_id}_chunk_{chunk_idx}",
                    "model": model,
                    "api": api,
                    "skip_existing": skip_existing,
                },
            )
            results.append(
                {
                    "chunk_idx": chunk_idx,
                    "message_count": len(chunk),
                    "status": "ok",
                    "result": result,
                }
            )
            print(f"    chunk {chunk_idx} ok ({len(chunk)} msgs)")
        except Exception as exc:
            results.append(
                {
                    "chunk_idx": chunk_idx,
                    "message_count": len(chunk),
                    "status": "failed",
                    "error": str(exc)[:200],
                }
            )
            print(f"    chunk {chunk_idx} FAILED: {str(exc)[:100]}")
        if request_delay > 0:
            time.sleep(request_delay)
    return results


def import_single_user(
    args: argparse.Namespace,
    data: list[dict[str, Any]],
    conv_idx: int,
    user_id: str,
    speaker_name: str,
    other_speaker_name: str,
    is_forward: bool,
) -> dict[str, Any]:
    item = data[conv_idx]
    conversation = item["conversation"]
    user_result = {
        "user_id": user_id,
        "conv_idx": conv_idx,
        "sample_id": item.get("sample_id", f"conv_{conv_idx}"),
        "role": "speaker_a" if is_forward else "speaker_b",
        "sessions": [],
        "total_chunks": 0,
        "ok_chunks": 0,
        "failed_chunks": 0,
    }
    for session_index in range(1, 100):
        session_key = f"session_{session_index}"
        if session_key not in conversation:
            break
        if not isinstance(conversation.get(session_key), list):
            continue
        date_time = str(conversation.get(f"{session_key}_date_time", "")).strip()
        session_id = f"{item.get('sample_id', f'conv_{conv_idx}')}_{session_key}"
        if is_forward:
            messages, _ = build_session_messages(conversation, session_key, speaker_name, other_speaker_name)
        else:
            _, messages = build_session_messages(conversation, session_key, other_speaker_name, speaker_name)
        print(f"  {user_id} {session_key} ({len(messages)} msgs)")
        chunks = import_for_user(
            args.base_url,
            user_id,
            session_id,
            api="locomobech_mem0",
            model="locomo10",
            messages=messages,
            batch_size=args.batch_size,
            request_delay=args.request_delay,
            skip_existing=args.skip_existing,
        )
        ok = sum(1 for c in chunks if c["status"] == "ok")
        failed = sum(1 for c in chunks if c["status"] == "failed")
        user_result["sessions"].append({
            "session_key": session_key,
            "date_time": date_time,
            "chunks": chunks,
            "ok": ok,
            "failed": failed,
        })
        user_result["total_chunks"] += len(chunks)
        user_result["ok_chunks"] += ok
        user_result["failed_chunks"] += failed
    return user_result


def main() -> None:
    args = parse_args()
    data = load_locomo10(args.data_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for conv_idx, item in enumerate(data):
        conversation = item["conversation"]
        speaker_a_user_id, speaker_b_user_id = make_user_ids(conversation, conv_idx)
        speaker_a = speaker_a_user_id.rsplit("_", 1)[0]
        speaker_b = speaker_b_user_id.rsplit("_", 1)[0]

        # speaker A
        result_path_a = output_dir / f"{speaker_a_user_id}.json"
        if result_path_a.exists():
            print(f"[skip] {speaker_a_user_id} already done")
        else:
            print(f"[start] {speaker_a_user_id}")
            result_a = import_single_user(args, data, conv_idx, speaker_a_user_id, speaker_a, speaker_b, is_forward=True)
            save_json(result_path_a, result_a)
            print(f"[done] {speaker_a_user_id}: ok={result_a['ok_chunks']} failed={result_a['failed_chunks']}")

        # speaker B
        result_path_b = output_dir / f"{speaker_b_user_id}.json"
        if result_path_b.exists():
            print(f"[skip] {speaker_b_user_id} already done")
        else:
            print(f"[start] {speaker_b_user_id}")
            result_b = import_single_user(args, data, conv_idx, speaker_b_user_id, speaker_b, speaker_a, is_forward=False)
            save_json(result_path_b, result_b)
            print(f"[done] {speaker_b_user_id}: ok={result_b['ok_chunks']} failed={result_b['failed_chunks']}")

    # 汇总
    summary = []
    for json_file in sorted(output_dir.glob("*.json")):
        user_data = json.loads(json_file.read_text(encoding="utf-8"))
        summary.append({
            "user_id": user_data["user_id"],
            "conv_idx": user_data["conv_idx"],
            "total_chunks": user_data["total_chunks"],
            "ok_chunks": user_data["ok_chunks"],
            "failed_chunks": user_data["failed_chunks"],
        })
    save_json(output_dir / "_summary.json", summary)
    print(f"\nsummary saved to {output_dir / '_summary.json'}")
    for row in summary:
        status = "ALL_OK" if row["failed_chunks"] == 0 else f"FAILED={row['failed_chunks']}"
        print(f"  {row['user_id']}: {status} (ok={row['ok_chunks']}/{row['total_chunks']})")


if __name__ == "__main__":
    main()
