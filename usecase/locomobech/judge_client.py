from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI




ANSWER_PROMPT = """
You are an intelligent memory assistant tasked with answering a question using only the provided memories.

Instructions:
1. Use only the provided memories.
2. Prefer direct evidence over guesswork.
3. If memories conflict, prefer the most recent relevant memory.
4. Convert relative time references into specific answers when possible.
5. Answer briefly, ideally under 5-6 words.

Memories:
{memories}

Question: {question}
Answer:
""".strip()


ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given:
1. a question,
2. a gold answer,
3. a generated answer.

Be generous. If the generated answer matches the same fact, date, time period, or entity in meaning, label it CORRECT.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Return only a JSON object: {"label": "CORRECT"} or {"label": "WRONG"}
""".strip()


class MiniMaxJudge:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model


    def answer_from_memories(self, question: str, memories: list[str]) -> str:
        prompt = ANSWER_PROMPT.format(
            question=question,
            memories=json.dumps(memories, ensure_ascii=False, indent=2),
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return (response.choices[0].message.content or "").strip()

    def score(self, question: str, gold_answer: str, generated_answer: str) -> int:
        prompt = ACCURACY_PROMPT.format(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = response.choices[0].message.content or ""
        except Exception:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = response.choices[0].message.content or ""
        label = self._extract_label(content)
        return 1 if label == "CORRECT" else 0

    @staticmethod
    def _extract_label(content: str) -> str:
        try:
            payload = json.loads(content)
            label = str(payload.get("label", "")).upper().strip()
            if label in {"CORRECT", "WRONG"}:
                return label
        except Exception:
            pass
        match = re.search(r"\b(CORRECT|WRONG)\b", content.upper())
        if match:
            return match.group(1)
        return "WRONG"


