#!/usr/bin/env python3
# Copyright 2026 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
用 Qwen3.5-9B（或其他 LLM）合成中文多 session 对话，构造 Stage 2 训练数据。

合成逻辑：
  1. 生成一段跨 3-5 session 的中文对话历史（含跨 session 引用）
  2. 生成一个 query（引用之前某个 session 的事件）
  3. 生成 target_memory（精简摘要，跨 session 融合）

输出格式（JSONL）：与 build_oracle_data.py 输出格式相同。

用法（本地 Qwen3.5-9B）：
    uv run python scripts/synth_conversations.py \\
        --backend local \\
        --model Qwen/Qwen3.5-9B-Instruct \\
        --n-samples 10000 \\
        --output data/oracle_synth_zh.jsonl

用法（OpenAI API）：
    OPENAI_API_KEY=xxx uv run python scripts/synth_conversations.py \\
        --backend openai \\
        --openai-model gpt-4o-mini \\
        --n-samples 5000 \\
        --output data/oracle_synth_zh.jsonl

注：合成数据可能有质量波动，建议人工抽检 1%。
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------

SYNTH_SYSTEM_PROMPT = """你是一个中文对话数据生成专家。你的任务是生成真实、自然的多 session 对话数据，用于训练记忆模型。"""

SYNTH_USER_PROMPT = """请生成一段跨 {n_sessions} 个 session 的中文对话历史。要求：

1. 对话主题随机选择（职场、生活、学习、健康、旅行、技术等）
2. 每个 session 10-20 轮对话
3. 后续 session 中用户明确引用之前 session 的事件（如"就像上次说的..."、"你还记得我提过..."）
4. 最后附上：(a) 一个基于历史的 query 问题，(b) 该 query 对应的精简记忆摘要（100字以内）

请严格按以下 JSON 格式输出，不要有任何额外文字：
{{
  "sessions": [
    {{
      "session_id": 1,
      "turns": [
        {{"speaker": "用户", "text": "..."}},
        {{"speaker": "助手", "text": "..."}}
      ]
    }}
  ],
  "query": "...",
  "target_memory": "..."
}}
"""

# ---------------------------------------------------------------------------
# 主题池（增加多样性）
# ---------------------------------------------------------------------------

TOPICS = [
    "求职面试和职业规划",
    "学习编程和技术成长",
    "健身计划和饮食管理",
    "旅行计划和目的地选择",
    "家庭关系和亲子教育",
    "理财投资和消费决策",
    "考研备考和学习方法",
    "创业项目和商业计划",
    "感情问题和人际关系",
    "购房置业和装修设计",
    "职场人际和团队管理",
    "兴趣爱好和技能提升",
]


# ---------------------------------------------------------------------------
# Backend: Local（Qwen3.5-9B via transformers）
# ---------------------------------------------------------------------------


class LocalBackend:
    def __init__(self, model_name: str, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("Loading local model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, system: str, user: str, max_new_tokens: int = 2000) -> str:
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(next(self.model.parameters()).device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Backend: OpenAI
# ---------------------------------------------------------------------------


class OpenAIBackend:
    def __init__(self, model: str = "gpt-4o-mini", base_url: str | None = None) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        self.client = openai.OpenAI(base_url=base_url) if base_url else openai.OpenAI()
        self.model = model

    def generate(self, system: str, user: str, max_new_tokens: int = 2000) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_new_tokens,
            temperature=0.8,
        )
        return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# 解析生成的 JSON
# ---------------------------------------------------------------------------


def parse_generated(text: str) -> dict | None:
    """从生成文本中提取 JSON（应对模型偶尔输出 markdown code block 的情况）。"""
    # 尝试直接解析
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 提取 ```json ... ``` 块
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 提取第一个 { ... } 块
    m = re.search(r"\{[\s\S]+\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


def json_to_sample(parsed: dict) -> dict | None:
    """将 parsed JSON 转成训练样本格式。"""
    sessions = parsed.get("sessions", [])
    query = parsed.get("query", "").strip()
    target_memory = parsed.get("target_memory", "").strip()

    if not sessions or not query or not target_memory:
        return None

    history: list[dict] = []
    for session in sessions:
        turns = session.get("turns", [])
        for turn in turns:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "").strip()
            if text:
                history.append({"role": "write", "content": f"{speaker}：{text}"})

    if not history:
        return None

    return {
        "history": history,
        "query": query,
        "target_memory": target_memory,
        "source": "synth",
    }


# ---------------------------------------------------------------------------
# 主生成循环
# ---------------------------------------------------------------------------


def generate_samples(
    backend: Any,
    n_samples: int,
    output_path: Path,
    n_sessions_range: tuple[int, int] = (3, 5),
    retry_limit: int = 3,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    failures = 0
    max_failures = n_samples * 3  # 防止无限循环

    with output_path.open("w", encoding="utf-8") as f:
        while count < n_samples and failures < max_failures:
            n_sess = random.randint(*n_sessions_range)
            topic = random.choice(TOPICS)

            user_prompt = SYNTH_USER_PROMPT.format(n_sessions=n_sess)
            user_prompt = f"对话主题：{topic}\n\n" + user_prompt

            success = False
            for attempt in range(retry_limit):
                try:
                    raw = backend.generate(SYNTH_SYSTEM_PROMPT, user_prompt)
                    parsed = parse_generated(raw)
                    if parsed is None:
                        log.debug("Parse failed (attempt %d), raw: %.100s...", attempt + 1, raw)
                        failures += 1
                        continue

                    sample = json_to_sample(parsed)
                    if sample is None:
                        log.debug("Invalid sample structure (attempt %d)", attempt + 1)
                        failures += 1
                        continue

                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    f.flush()
                    count += 1
                    success = True
                    break

                except Exception as e:
                    log.warning("Generation error (attempt %d/%d): %s", attempt + 1, retry_limit, e)
                    time.sleep(2 ** attempt)  # exponential backoff
                    failures += 1

            if not success:
                log.debug("Sample %d failed after %d retries", count + 1, retry_limit)

            if count % 100 == 0 and count > 0:
                log.info("Generated %d / %d samples (failures=%d)", count, n_samples, failures)

    log.info("Done: %d samples generated, %d failures", count, failures)
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthesize Chinese multi-session conversations")

    p.add_argument("--backend", choices=["local", "openai"], default="local")
    p.add_argument("--model", default="Qwen/Qwen3.5-9B-Instruct",
                   help="Local model name (for --backend local)")
    p.add_argument("--openai-model", default="gpt-4o-mini",
                   help="OpenAI model name (for --backend openai)")
    p.add_argument("--openai-base-url", default=None,
                   help="Custom OpenAI base URL (e.g. for proxies)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-samples", type=int, default=10000)
    p.add_argument("--output", required=True, help="Output JSONL file")
    p.add_argument("--min-sessions", type=int, default=3)
    p.add_argument("--max-sessions", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.backend == "local":
        backend = LocalBackend(args.model, args.device)
    else:
        backend = OpenAIBackend(args.openai_model, args.openai_base_url)

    output_path = Path(args.output)
    log.info("Generating %d samples → %s (backend=%s)", args.n_samples, output_path, args.backend)

    count = generate_samples(
        backend=backend,
        n_samples=args.n_samples,
        output_path=output_path,
        n_sessions_range=(args.min_sessions, args.max_sessions),
    )
    log.info("Final count: %d samples", count)


if __name__ == "__main__":
    main()
