from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def _normalize_text(text: str) -> str:
    return text.strip()


def _build_question_prompt(question: str) -> str:
    return f"问题：{_normalize_text(question)}\n答案："


@dataclass
class Stage1QueryInstance:
    episode_id: str
    question_id: str
    history_chunks: list[str]
    question: str
    answer: str
    prompt_text: str
    answer_text: str
    meta: dict[str, Any]


class Stage1Dataset(Dataset[Stage1QueryInstance]):
    def __init__(self, data_dir: str | Path, split: str = "train") -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.samples = self._load_split(split)

    def _load_split(self, split: str) -> list[Stage1QueryInstance]:
        if split == "train":
            return self._load_jsonl(self.data_dir / "train.jsonl")
        if split in {"eval", "eval_structured", "eval_itinerary"}:
            target = self.data_dir / "eval.jsonl"
            if not target.exists():
                if split == "eval_structured":
                    target = self.data_dir / "eval_structured.jsonl"
                elif split == "eval_itinerary":
                    target = self.data_dir / "eval_itinerary.jsonl"
            return self._load_jsonl(target)
        raise ValueError(f"Unknown split: {split}")

    def _load_jsonl(self, path: Path) -> list[Stage1QueryInstance]:
        samples: list[Stage1QueryInstance] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                samples.extend(self._normalize_row(row))
        return samples

    def _normalize_row(self, row: dict[str, Any]) -> list[Stage1QueryInstance]:
        if row.get("schema_version") == "stage1_timeline_v2" and "timeline" in row:
            return self._flatten_timeline_episode(row)

        if "history_chunks" in row and "question_chunk" in row and "answer" in row:
            question = _normalize_text(row["question_chunk"])
            answer = _normalize_text(row["answer"])
            meta = dict(row.get("meta", {}))
            return [
                Stage1QueryInstance(
                    episode_id=row.get("episode_id", row.get("sample_id", "legacy")),
                    question_id=row.get("question_id", f"{row.get('episode_id', 'legacy')}:q0"),
                    history_chunks=[_normalize_text(chunk) for chunk in row["history_chunks"]],
                    question=question,
                    answer=answer,
                    prompt_text=_build_question_prompt(question),
                    answer_text=answer,
                    meta=meta,
                )
            ]

        raise ValueError(f"Unsupported row format with keys: {sorted(row.keys())}")

    def _flatten_timeline_episode(self, row: dict[str, Any]) -> list[Stage1QueryInstance]:
        history_prefix: list[str] = []
        samples: list[Stage1QueryInstance] = []
        for step in row["timeline"]:
            if step["event_type"] in {"write", "noop"}:
                history_prefix.append(_normalize_text(step["text"]))
                continue
            if step["event_type"] != "query":
                continue
            for question in step["questions"]:
                text = _normalize_text(question["question"])
                answer = _normalize_text(question["answer"])
                sample_meta = {
                    **dict(row.get("meta", {})),
                    "schema_version": row.get("schema_version"),
                    "step_id": step["time_index"],
                    "target_fact_ids": question.get("target_fact_ids", []),
                    "stale_fact_ids": question.get("stale_fact_ids", []),
                    "target_entity": question.get("target_entity"),
                    "target_attribute": question.get("target_attribute"),
                    "valid_current_only": question.get("valid_current_only", True),
                }
                samples.append(
                    Stage1QueryInstance(
                        episode_id=row["sample_id"],
                        question_id=question["question_id"],
                        history_chunks=list(history_prefix),
                        question=text,
                        answer=answer,
                        prompt_text=_build_question_prompt(text),
                        answer_text=answer,
                        meta=sample_meta,
                    )
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Stage1QueryInstance:
        return self.samples[index]


def _mask_answer_only_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: list[int],
    pad_token_id: int,
) -> torch.Tensor:
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    labels[input_ids == pad_token_id] = -100
    for row_index, prompt_len in enumerate(prompt_lengths):
        labels[row_index, :prompt_len] = -100
    return labels


def stage1_collate_fn(
    batch: list[Stage1QueryInstance],
    tokenizer: Any,
    max_history_length: int = 256,
    max_question_length: int = 256,
    max_sequence_length: int = 512,
    loss_mask_scope: str = "answer_only",
) -> dict[str, Any]:
    if loss_mask_scope != "answer_only":
        raise ValueError(f"Unsupported loss_mask_scope: {loss_mask_scope}")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id")

    history_counts = [len(sample.history_chunks) for sample in batch]
    max_history_chunks = max(history_counts, default=0)
    batch_size = len(batch)

    history_input_ids = torch.full(
        (batch_size, max_history_chunks, max_history_length),
        fill_value=pad_token_id,
        dtype=torch.long,
    )
    history_attention_mask = torch.zeros(
        (batch_size, max_history_chunks, max_history_length),
        dtype=torch.long,
    )
    history_chunk_mask = torch.zeros((batch_size, max_history_chunks), dtype=torch.bool)

    flattened_history: list[str] = []
    flattened_positions: list[tuple[int, int]] = []
    for sample_index, sample in enumerate(batch):
        for chunk_index, chunk in enumerate(sample.history_chunks):
            flattened_history.append(chunk)
            flattened_positions.append((sample_index, chunk_index))

    if flattened_history:
        tokenized_history = tokenizer(
            flattened_history,
            padding="max_length",
            truncation=True,
            max_length=max_history_length,
            return_tensors="pt",
        )
        for flat_index, (sample_index, chunk_index) in enumerate(flattened_positions):
            history_input_ids[sample_index, chunk_index] = tokenized_history["input_ids"][flat_index]
            history_attention_mask[sample_index, chunk_index] = tokenized_history[
                "attention_mask"
            ][flat_index]
            history_chunk_mask[sample_index, chunk_index] = True

    question_prompts = [sample.prompt_text for sample in batch]
    prompt_tokens = tokenizer(
        question_prompts,
        padding="max_length",
        truncation=True,
        max_length=max_question_length,
        return_tensors="pt",
    )
    prompt_lengths = prompt_tokens["attention_mask"].sum(dim=1).tolist()

    full_texts = [f"{sample.prompt_text}{sample.answer_text}" for sample in batch]
    full_tokens = tokenizer(
        full_texts,
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length,
        return_tensors="pt",
    )
    labels = _mask_answer_only_labels(
        input_ids=full_tokens["input_ids"],
        attention_mask=full_tokens["attention_mask"],
        prompt_lengths=prompt_lengths,
        pad_token_id=pad_token_id,
    )

    step_ids = torch.tensor(
        [int(sample.meta.get("step_id", len(sample.history_chunks))) for sample in batch],
        dtype=torch.long,
    )
    write_counts = torch.tensor(
        [int(sample.meta.get("num_write_steps", 0)) for sample in batch],
        dtype=torch.long,
    )
    query_loss_mask = torch.ones(batch_size, dtype=torch.bool)

    return {
        "history_input_ids": history_input_ids,
        "history_attention_mask": history_attention_mask,
        "history_chunk_mask": history_chunk_mask,
        "question_input_ids": prompt_tokens["input_ids"],
        "question_attention_mask": prompt_tokens["attention_mask"],
        "input_ids": full_tokens["input_ids"],
        "attention_mask": full_tokens["attention_mask"],
        "labels": labels,
        "query_loss_mask": query_loss_mask,
        "step_ids": step_ids,
        "write_counts": write_counts,
        "answers": [sample.answer for sample in batch],
        "questions": [sample.question for sample in batch],
        "episode_ids": [sample.episode_id for sample in batch],
        "question_ids": [sample.question_id for sample in batch],
        "meta": [sample.meta for sample in batch],
    }


