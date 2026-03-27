#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


LAST_NAMES = [
    "陈", "李", "周", "王", "林", "许", "顾", "宋", "韩", "沈",
    "徐", "何", "程", "郑", "谢", "梁", "苏", "唐", "吴", "杨",
]

GIVEN_NAMES = [
    "默", "想", "宁", "安", "泽", "然", "川", "言", "悦", "清",
    "嘉", "航", "晨", "宇", "桐", "辰", "远", "一", "佳", "诚",
]

CITIES = ["北京", "上海", "苏州", "杭州", "深圳", "广州", "南京", "成都", "武汉"]
HOTELS = ["金鸡湖酒店", "浦东商务酒店", "城南国际酒店", "临湖宾馆"]
MEETING_TOPICS = ["项目排期", "预算评审", "联调安排", "客户回访", "交付计划"]
NOISE_TEMPLATES = [
    "{name} 今天参加了周会，主要讨论了{topic}。",
    "团队计划下周继续推进{topic}，并整理阶段总结。",
    "本周的沟通重点依然是{topic}和后续执行节奏。",
    "项目成员同步了当前状态，没有新增风险。",
    "本次讨论没有涉及客户最终排期，只是内部沟通。",
]

ATTRIBUTE_SPECS = {
    "employee_id": {
        "questions": [
            "{entity}的工号是什么？",
            "{entity}现在的工号是多少？",
            "请问{entity}的工号是？",
        ],
        "write": "{entity}的工号是 {value}。",
        "update": "更新后，{entity}现在的工号改为 {value}。",
    },
    "phone_tail": {
        "questions": [
            "{entity}的电话尾号是多少？",
            "{entity}现在的电话尾号是几位？",
            "请问{entity}的电话尾号是什么？",
        ],
        "write": "{entity}的电话尾号是 {value}。",
        "update": "更新后，{entity}现在的电话尾号改为 {value}。",
    },
    "work_city": {
        "questions": [
            "{entity}目前在哪个城市工作？",
            "{entity}现在在哪里工作？",
            "{entity}当前工作的城市是哪里？",
        ],
        "write": "{entity}目前在{value}工作。",
        "update": "更新后，{entity}现在改为在{value}工作。",
    },
    "trip_city": {
        "questions": [
            "{entity}现在要去哪里？",
            "{entity}这次出差去哪个城市？",
            "{entity}当前的出差目的地是哪里？",
        ],
        "write": "{entity}原定去{value}出差。",
        "update": "更新后，{entity}现在改为去{value}出差。",
    },
    "hotel": {
        "questions": [
            "{entity}现在订的酒店在哪里？",
            "{entity}当前预订的是哪家酒店？",
            "{entity}这次住的酒店叫什么？",
        ],
        "write": "{entity}当前预订的酒店是{value}。",
        "update": "更新后，{entity}现在改订在{value}。",
    },
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_name_pool(rng: random.Random, target_size: int = 300) -> list[str]:
    names: set[str] = set()
    while len(names) < target_size:
        last = rng.choice(LAST_NAMES)
        first = rng.choice(GIVEN_NAMES)
        if rng.random() < 0.45:
            first += rng.choice(GIVEN_NAMES)
        names.add(last + first)
    return sorted(names)


def employee_id(rng: random.Random) -> str:
    return f"{rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{rng.randint(10,99)}-{rng.randint(1000,9999)}"


def phone_tail(rng: random.Random) -> str:
    return f"{rng.randint(1000,9999)}"


def flight_no(rng: random.Random) -> str:
    return f"{rng.choice(['MU','CA','ZH','HU'])}{rng.randint(1000,9999)}"


def weekday_text(rng: random.Random) -> str:
    return rng.choice(["周一", "周二", "周三", "周四", "周五"])


def sample_value(rng: random.Random, attribute: str) -> str:
    if attribute == "employee_id":
        return employee_id(rng)
    if attribute == "phone_tail":
        return phone_tail(rng)
    if attribute in {"work_city", "trip_city"}:
        return rng.choice(CITIES)
    if attribute == "hotel":
        return rng.choice(HOTELS)
    raise ValueError(f"Unknown attribute: {attribute}")


def noise_sentence(rng: random.Random, names: list[str]) -> str:
    template = rng.choice(NOISE_TEMPLATES)
    return template.format(name=rng.choice(names), topic=rng.choice(MEETING_TOPICS))


def render_write_text(entity: str, attribute: str, value: str, overwrite: bool) -> str:
    template_key = "update" if overwrite else "write"
    return ATTRIBUTE_SPECS[attribute][template_key].format(entity=entity, value=value)


def render_question(entity: str, attribute: str) -> str:
    return ATTRIBUTE_SPECS[attribute]["questions"][0].format(entity=entity)


def make_fact_id(entity: str, attribute: str, step_index: int) -> str:
    return f"{entity}:{attribute}:{step_index}"


def choose_query_targets(
    rng: random.Random,
    current_state: dict[tuple[str, str], dict[str, Any]],
    last_asked_steps: dict[tuple[str, str], int],
    write_slot: tuple[str, str] | None,
) -> list[tuple[str, str]]:
    active_slots = list(current_state.keys())
    if not active_slots:
        return []
    targets: list[tuple[str, str]] = []
    recent_candidates = [slot for slot in active_slots if slot == write_slot]
    old_candidates = [slot for slot in active_slots if slot != write_slot]
    old_candidates.sort(key=lambda slot: (last_asked_steps.get(slot, -1000), current_state[slot]["time_index"]))
    max_questions = 2 if len(active_slots) > 1 else 1

    if recent_candidates and rng.random() < 0.7:
        targets.append(recent_candidates[0])

    for slot in old_candidates:
        if len(targets) >= max_questions:
            break
        if last_asked_steps.get(slot, -1000) >= current_state[slot]["time_index"]:
            continue
        targets.append(slot)

    if len(targets) < max_questions:
        remaining = [slot for slot in active_slots if slot not in targets]
        rng.shuffle(remaining)
        for slot in remaining:
            if len(targets) >= max_questions:
                break
            targets.append(slot)

    if not targets:
        targets.append(rng.choice(active_slots))
    return targets


def choose_question_variant(
    rng: random.Random,
    attribute: str,
    last_question_variants: dict[tuple[str, str], int],
    slot: tuple[str, str],
) -> str:
    variants = ATTRIBUTE_SPECS[attribute]["questions"]
    start_index = (last_question_variants.get(slot, -1) + 1) % len(variants)
    candidate_indices = list(range(len(variants)))
    ordered = candidate_indices[start_index:] + candidate_indices[:start_index]
    chosen_index = ordered[0] if rng.random() < 0.8 else rng.choice(candidate_indices)
    last_question_variants[slot] = chosen_index
    return variants[chosen_index]


def build_timeline_episode(
    rng: random.Random,
    names: list[str],
    sample_id: str,
    min_steps: int,
    max_steps: int,
) -> dict[str, Any]:
    num_steps = rng.randint(min_steps, max_steps)
    timeline: list[dict[str, Any]] = []
    current_state: dict[tuple[str, str], dict[str, Any]] = {}
    fact_lineage: dict[tuple[str, str], list[str]] = {}
    all_entities = rng.sample(names, k=rng.randint(4, 7))
    seen_slots: set[tuple[str, str]] = set()
    query_count = 0
    write_count = 0
    update_count = 0
    last_asked_steps: dict[tuple[str, str], int] = {}
    last_question_variants: dict[tuple[str, str], int] = {}

    for step_index in range(num_steps):
        active_slots = list(current_state.keys())
        can_update = bool(active_slots)
        do_write = not active_slots or rng.random() < 0.35
        write_slot: tuple[str, str] | None = None

        if do_write:
            overwrite = can_update and rng.random() < 0.45
            if overwrite:
                write_slot = rng.choice(active_slots)
                old_fact = current_state[write_slot]
                old_value = old_fact["value"]
            else:
                candidates = [
                    (entity, attribute)
                    for entity in all_entities
                    for attribute in ATTRIBUTE_SPECS
                    if (entity, attribute) not in seen_slots
                ]
                if not candidates:
                    write_slot = rng.choice(active_slots)
                    overwrite = True
                    old_fact = current_state[write_slot]
                    old_value = old_fact["value"]
                else:
                    write_slot = rng.choice(candidates)
                    old_fact = None
                    old_value = None

            entity, attribute = write_slot
            new_value = sample_value(rng, attribute)
            if overwrite:
                while new_value == old_value:
                    new_value = sample_value(rng, attribute)
            fact_id = make_fact_id(entity, attribute, step_index)
            timeline.append(
                {
                    "time_index": step_index,
                    "event_type": "write",
                    "text": render_write_text(entity, attribute, new_value, overwrite),
                    "write_candidate": True,
                    "should_write": True,
                    "fact_id": fact_id,
                    "entity": entity,
                    "attribute": attribute,
                    "value": new_value,
                    "overwrite": overwrite,
                    "replaced_fact_id": old_fact["fact_id"] if overwrite and old_fact else None,
                }
            )
            current_state[write_slot] = {
                "fact_id": fact_id,
                "value": new_value,
                "time_index": step_index,
            }
            fact_lineage.setdefault(write_slot, []).append(fact_id)
            seen_slots.add(write_slot)
            write_count += 1
            if overwrite:
                update_count += 1
        else:
            timeline.append(
                {
                    "time_index": step_index,
                    "event_type": "noop",
                    "text": noise_sentence(rng, names),
                    "write_candidate": False,
                    "should_write": False,
                }
            )

        if current_state:
            ask_now = write_slot is not None or rng.random() < 0.85
            if not ask_now:
                continue
            query_targets = choose_query_targets(rng, current_state, last_asked_steps, write_slot)
            questions = []
            for entity, attribute in query_targets:
                slot_state = current_state[(entity, attribute)]
                stale_ids = fact_lineage.get((entity, attribute), [])[:-1]
                last_asked_steps[(entity, attribute)] = step_index
                questions.append(
                    {
                        "question_id": f"{sample_id}:q:{step_index}:{len(questions)}",
                        "question": ATTRIBUTE_SPECS[attribute]["questions"][
                            rng.randrange(len(ATTRIBUTE_SPECS[attribute]["questions"]))
                        ].format(entity=entity),
                        "answer": slot_state["value"],
                        "target_fact_ids": [slot_state["fact_id"]],
                        "stale_fact_ids": stale_ids,
                        "target_entity": entity,
                        "target_attribute": attribute,
                        "answer_type": "current_valid_fact",
                        "valid_current_only": True,
                    }
                )
            timeline.append(
                {
                    "time_index": step_index,
                    "event_type": "query",
                    "questions": questions,
                }
            )
            query_count += len(questions)

    final_state = {
        entity: {
            attribute: current_state[(entity, attribute)]["value"]
            for entity2, attribute in current_state
            if entity2 == entity
        }
        for entity, _ in current_state
    }

    return {
        "sample_id": sample_id,
        "schema_version": "stage1_timeline_v2",
        "timeline": timeline,
        "final_state": final_state,
        "meta": {
            "num_steps": num_steps,
            "num_write_steps": write_count,
            "num_query_steps": query_count,
            "num_updates": update_count,
            "dense_question_sparse_write": True,
        },
    }


def flatten_episode(episode: dict[str, Any]) -> list[dict[str, Any]]:
    history_prefix: list[str] = []
    rows: list[dict[str, Any]] = []
    for step in episode["timeline"]:
        if step["event_type"] in {"write", "noop"}:
            history_prefix.append(step["text"])
            continue
        if step["event_type"] != "query":
            continue
        for question in step["questions"]:
            rows.append(
                {
                    "episode_id": episode["sample_id"],
                    "question_id": question["question_id"],
                    "history_chunks": list(history_prefix),
                    "question_chunk": question["question"],
                    "answer": question["answer"],
                    "meta": {
                        **episode["meta"],
                        "schema_version": episode["schema_version"],
                        "step_id": step["time_index"],
                        "task_type": question["answer_type"],
                        "target_entity": question["target_entity"],
                        "target_attribute": question["target_attribute"],
                        "target_fact_ids": question["target_fact_ids"],
                        "stale_fact_ids": question["stale_fact_ids"],
                        "overwrite_depth": len(question["stale_fact_ids"]),
                        "num_history_chunks": len(history_prefix),
                        "valid_current_only": question["valid_current_only"],
                    },
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_preview(path: Path, episodes: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    parts = ["# Stage1 Timeline v2 Preview\n"]
    for idx, episode in enumerate(episodes[:2], start=1):
        parts.append(f"\n## Episode {idx} - {episode['sample_id']}\n")
        for step in episode["timeline"][:10]:
            if step["event_type"] == "query":
                parts.append(f"- step {step['time_index']} [query]\n")
                for q in step["questions"]:
                    parts.append(f"  - Q: {q['question']}\n")
                    parts.append(f"    A: {q['answer']}\n")
            else:
                parts.append(
                    f"- step {step['time_index']} [{step['event_type']}] {step['text']}\n"
                )

    parts.append("\n## Flattened Samples\n")
    for row in rows[:4]:
        parts.append(f"\n### {row['question_id']}\n")
        parts.append(f"- history_chunks: {len(row['history_chunks'])}\n")
        parts.append(f"- question: {row['question_chunk']}\n")
        parts.append(f"- answer: {row['answer']}\n")
        parts.append(f"- meta: {json.dumps(row['meta'], ensure_ascii=False)}\n")

    path.write_text("".join(parts), encoding="utf-8")


def write_stats(path: Path, episodes: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    stats = {
        "schema_version": "stage1_timeline_v2",
        "num_episodes": len(episodes),
        "num_flattened_rows": len(rows),
        "avg_steps_per_episode": sum(ep["meta"]["num_steps"] for ep in episodes) / max(1, len(episodes)),
        "avg_write_steps_per_episode": sum(ep["meta"]["num_write_steps"] for ep in episodes) / max(1, len(episodes)),
        "avg_query_items_per_episode": sum(ep["meta"]["num_query_steps"] for ep in episodes) / max(1, len(episodes)),
        "num_updates": sum(ep["meta"]["num_updates"] for ep in episodes),
    }
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def build_split(
    rng: random.Random,
    names: list[str],
    count: int,
    prefix: str,
    min_steps: int,
    max_steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes = [
        build_timeline_episode(rng, names, f"{prefix}_{index:06d}", min_steps, max_steps)
        for index in range(count)
    ]
    rows = [row for episode in episodes for row in flatten_episode(episode)]
    return episodes, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=200)
    parser.add_argument("--eval-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/generated/stage1_timeline_v2"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    ensure_dir(args.output_dir)

    names = build_name_pool(rng)
    train_episodes, train_rows = build_split(
        rng,
        names,
        count=args.train_size,
        prefix="train",
        min_steps=args.min_steps,
        max_steps=args.max_steps,
    )
    eval_episodes, eval_rows = build_split(
        rng,
        names,
        count=args.eval_size,
        prefix="eval",
        min_steps=args.min_steps,
        max_steps=args.max_steps,
    )

    write_jsonl(args.output_dir / "train_episodes.jsonl", train_episodes)
    write_jsonl(args.output_dir / "eval_episodes.jsonl", eval_episodes)
    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "eval.jsonl", eval_rows)
    write_preview(args.output_dir / "preview.md", train_episodes, train_rows)
    write_stats(args.output_dir / "stats.json", train_episodes, train_rows)

    print(f"Generated stage1 timeline v2 dataset under: {args.output_dir}")


if __name__ == "__main__":
    main()

