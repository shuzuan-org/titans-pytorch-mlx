from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-ended",
    5: "adversarial",
}


def load_locomo10(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("locomo10 data must be a list")
    return payload


def extract_turns(conversation: dict[str, Any]) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for index in range(1, 100):
        session_key = f"session_{index}"
        if session_key not in conversation:
            break
        session = conversation[session_key]
        if not isinstance(session, list):
            continue
        for turn in session:
            if not isinstance(turn, dict):
                continue
            speaker = str(turn.get("speaker", "")).strip()
            text = str(turn.get("text", "")).strip()
            if not text:
                continue
            role = "user" if len(turns) % 2 == 0 else "assistant"
            turns.append({"role": role, "content": f"{speaker}: {text}" if speaker else text})
    return turns


def extract_plain_turns(conversation: dict[str, Any]) -> list[str]:
    turns: list[str] = []
    for index in range(1, 100):
        session_key = f"session_{index}"
        if session_key not in conversation:
            break
        session = conversation[session_key]
        if not isinstance(session, list):
            continue
        for turn in session:
            if not isinstance(turn, dict):
                continue
            speaker = str(turn.get("speaker", "")).strip()
            text = str(turn.get("text", "")).strip()
            if not text:
                continue
            turns.append(f"{speaker}: {text}" if speaker else text)
    return turns


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(str(prediction).lower().split())
    ref_tokens = set(str(reference).lower().split())
    if not pred_tokens or not ref_tokens:
        return float(str(prediction).strip().lower() == str(reference).strip().lower())
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def summarize_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["category"])].append(row)
    by_category = {}
    all_bleu: list[float] = []
    all_f1: list[float] = []
    all_llm: list[float] = []
    for category, items in sorted(grouped.items()):
        bleu_values = [float(item.get("bleu_score", 0.0)) for item in items]
        f1_values = [float(item.get("f1_score", 0.0)) for item in items]
        llm_values = [float(item.get("llm_score", 0.0)) for item in items]
        all_bleu.extend(bleu_values)
        all_f1.extend(f1_values)
        all_llm.extend(llm_values)
        by_category[category] = {
            "count": len(items),
            "bleu_score": sum(bleu_values) / len(bleu_values),
            "f1_score": sum(f1_values) / len(f1_values),
            "llm_score": sum(llm_values) / len(llm_values),
        }
    total = len(rows)
    return {
        "total": total,
        "overall": {
            "bleu_score": sum(all_bleu) / len(all_bleu) if all_bleu else 0.0,
            "f1_score": sum(all_f1) / len(all_f1) if all_f1 else 0.0,
            "llm_score": sum(all_llm) / len(all_llm) if all_llm else 0.0,
        },
        "by_category": by_category,
    }


def save_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

