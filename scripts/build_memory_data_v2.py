#!/usr/bin/env python3
# Copyright 2026 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
记忆训练数据 v2 生成脚本。

解决 v1 的核心问题：
  - Stage 1 (CMRC/DuReader)：文档 QA，不是记忆任务
  - Stage 2 (MSC)：target 是助手下一句回复，不是记忆摘要

v2 设计原则：
  1. 每条样本混合"重要写入"和"噪音写入"——强迫模型学选择性记忆
  2. target_memory 是跨 write 的综合摘要，不是对话续写
  3. 时序感：写入带时间标记，同一实体状态可更新
  4. 三类数据形态，循序渐进：
       Stage 1 热身：单 session，直接事实，无噪音
       Stage 2 多轮：多 session，混合噪音，跨轮综合
       Stage 3 更新：同实体多次更新，target 只含最新状态

输出格式（JSONL，与 v1 兼容）：
{
  "history":       [{"role": "write", "content": "...", "importance": 0|1}, ...],
  "query":         "...",
  "target_memory": "...",   # 真正的记忆摘要，非对话回复
  "source":        "memory_v2_s1/s2/s3"
}
importance 嵌在每个 turn 内（1=重要，0=噪音），与 train_memory_oracle.py Stage 3 格式兼容。

断点续跑：输出文件使用追加模式，重复运行同一命令会从断点继续，不会截断已有数据。

用法（本地模型）：
    python scripts/build_memory_data_v2.py \\
        --stage 1 --n-samples 5000 --output data/memory_v2/stage1.jsonl \\
        --backend local --model /home/shuzuan/models/Qwen/Qwen3___5-9B

用法（OpenAI 兼容 API）：
    OPENAI_API_KEY=xxx python scripts/build_memory_data_v2.py \\
        --stage 2 --n-samples 20000 --output data/memory_v2/stage2.jsonl \\
        --backend openai --openai-model gpt-4o-mini

质量要求：建议人工抽检 2%；生成后运行 --validate 检查格式。
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
# 主题池（50+ 覆盖个人/职场/企业场景）
# ---------------------------------------------------------------------------

TOPICS_PERSONAL = [
    "求职面试和职业转型",
    "职场晋升和薪资谈判",
    "创业项目从0到1",
    "副业变现和时间管理",
    "感情发展和恋爱决策",
    "婚姻规划和家庭组建",
    "分手修复和情感重建",
    "与父母的沟通和代际冲突",
    "子女教育和亲子关系",
    "健身计划执行和体重管理",
    "慢性病管理和康复过程",
    "心理健康和情绪调节",
    "焦虑抑郁和心理咨询",
    "考研备考和学习规划",
    "出国留学申请和准备",
    "技能学习和职业转型",
    "购房贷款和置业决策",
    "装修设计和搬家过程",
    "理财投资和风险管理",
    "消费决策和预算管理",
    "旅行规划和目的地选择",
    "宠物养育和健康管理",
    "兴趣爱好深耕和技能提升",
    "社交焦虑和人际关系",
    "友谊维护和社交圈变化",
]

TOPICS_WORK = [
    "重要项目进展和里程碑",
    "客户关系维护和需求变化",
    "团队冲突和人事变动",
    "技术架构决策和重构",
    "产品迭代和用户反馈",
    "商务谈判和合同签署",
    "预算申请和资源协调",
    "绩效评估和职级调整",
    "跨部门协作和沟通问题",
    "新员工培养和团队建设",
    "危机处理和应急响应",
    "竞品分析和市场变化",
    "融资进展和投资人沟通",
    "供应链管理和供应商关系",
    "数据分析和决策依据",
]

TOPICS_LIFE_CHANGES = [
    "城市迁移和生活适应",
    "职业瓶颈和中年危机",
    "亲人生病和照料压力",
    "重大失去和悲伤处理",
    "意外收获和人生转折",
    "慢性压力积累和崩溃重建",
    "价值观冲突和人生选择",
    "退休规划和人生下半场",
]

ALL_TOPICS = TOPICS_PERSONAL + TOPICS_WORK + TOPICS_LIFE_CHANGES

# 时间跨度选项
TIME_SPANS = [
    ("一周内", 2),
    ("一个月", 3),
    ("三个月", 4),
    ("半年", 5),
    ("一年", 6),
]

# 查询类型
QUERY_TYPES = [
    "当前状态",       # 用户现在的情况是什么
    "重要事件",       # 这段时间发生了哪些关键事情
    "变化历程",       # 某事物是如何变化的
    "决策建议",       # 基于历史背景，给出建议
    "情感状态",       # 用户的心理/情绪状态
    "关系进展",       # 某段关系/项目的进展
    "待解问题",       # 还有哪些未解决的问题
]

# ---------------------------------------------------------------------------
# English topic pool & prompts
# ---------------------------------------------------------------------------

TOPICS_PERSONAL_EN = [
    "job search, career change and interview prep",
    "promotion negotiation and salary discussion",
    "starting a side business from scratch",
    "romantic relationship development and dating decisions",
    "marriage planning and family building",
    "breakup recovery and emotional rebuilding",
    "parent-child conflict and generational gap",
    "fitness routine and weight management",
    "chronic illness management and recovery",
    "mental health and anxiety management",
    "graduate school prep and study planning",
    "studying abroad: application and preparation",
    "home buying, mortgage and moving",
    "personal finance, investing and budgeting",
    "travel planning and destination choices",
    "pet care and health management",
    "social anxiety and building relationships",
    "friendship maintenance and social circle changes",
]

TOPICS_WORK_EN = [
    "project milestones and progress updates",
    "client relationship management and changing requirements",
    "team conflict and staff changes",
    "technical architecture decisions and refactoring",
    "product iteration and user feedback",
    "business negotiation and contract signing",
    "budget planning and resource allocation",
    "performance review and promotion",
    "cross-team collaboration and communication",
    "onboarding and team building",
    "crisis management and incident response",
    "competitive analysis and market changes",
    "fundraising progress and investor communications",
    "data analysis and decision-making",
]

TOPICS_LIFE_EN = [
    "relocating to a new city and adjusting",
    "career plateau and midlife transition",
    "caring for a sick family member",
    "grief and coping with major loss",
    "unexpected windfall and life turning point",
    "chronic stress accumulation and burnout recovery",
    "values conflict and life choices",
    "retirement planning and next chapter",
]

ALL_TOPICS_EN = TOPICS_PERSONAL_EN + TOPICS_WORK_EN + TOPICS_LIFE_EN

TIME_SPANS_EN = [
    ("within a week", 2),
    ("over a month", 3),
    ("over three months", 4),
    ("over six months", 5),
    ("over a year", 6),
]

QUERY_TYPES_EN = [
    "current status",
    "key events",
    "how things changed",
    "decision advice based on history",
    "emotional state",
    "relationship or project progress",
    "unresolved issues",
]

SYSTEM_PROMPT_EN = """You are an expert at generating memory training data. \
Your data is used to train an AI long-term memory module.

Core requirements:
- target_memory must be a "memory summary", NOT a conversational reply
- write entries must include unrelated daily noise so the model learns to filter
- write entries should have time labels to show temporal order
- later sessions should naturally reference earlier events"""

STAGE1_PROMPT_EN = """Generate one memory training sample (warm-up stage).

Topic: {topic}
Requirements:
1. Write entries: 4-8 total, time span about one week, natural language with time labels
2. 1-3 important entries (related to the query); rest are daily noise (unrelated)
3. query: a question about the important content
4. target_memory: concise summary of only the important content (under 50 words)

Important: target_memory is a MEMORY SUMMARY, not an answer. Format: "The user [did/experienced/is]..."

Output strictly as JSON, no extra text:
{{
  "writes": [
    {{"time": "Monday", "content": "...", "is_noise": false}},
    {{"time": "Tuesday", "content": "Had takeout for lunch", "is_noise": true}},
    ...
  ],
  "query": "...",
  "target_memory": "The user ..."
}}"""

STAGE2_PROMPT_EN = """Generate one memory training sample (multi-session stage).

Topic: {topic}
Time span: {time_span}
Number of sessions: {n_sessions}
Query type: {query_type}

Requirements:
1. Write entries: 10-20 total, across {n_sessions} sessions, 2-5 entries per session
2. Important entries (2-5): key events that affect the user's life or decisions; must be referenced in later sessions
3. Noise entries (6-12): unrelated daily trivia
4. At least 1 cross-session reference ("like I mentioned before", "remember when I said...")
5. For "how things changed" or "current status" queries: same entity should update across sessions

target_memory requirements:
- Summarize only entries relevant to the query
- Ignore noise
- Under 100 words, natural language
- Describe background memory, not answer the query directly

Output strictly as JSON, no extra text:
{{
  "sessions": [
    {{
      "session_id": 1,
      "time_label": "3 months ago",
      "writes": [
        {{"content": "...", "is_noise": false}},
        {{"content": "Nice weather today", "is_noise": true}}
      ]
    }}
  ],
  "query": "...",
  "target_memory": "..."
}}"""

STAGE3_PROMPT_EN = """Generate one memory training sample (information update stage).

Topic: {topic}
Requirements: Show the same entity changing state across time (new info should override old info)

Must include:
1. The same entity (person/project/relationship/status) at 3 different time points
   Example: ["Alex is an intern" → "Alex became a full-time engineer" → "Alex was promoted to lead"]
2. 1-2 noise entries at each time point
3. query asks about the entity's CURRENT state
4. target_memory describes only the LATEST state (no history recap)

Output strictly as JSON, no extra text:
{{
  "entity": "name of the tracked entity",
  "updates": [
    {{
      "time_label": "6 months ago",
      "signal": "entity's state description",
      "noise": ["noise content 1"]
    }},
    {{
      "time_label": "3 months ago",
      "signal": "updated state",
      "noise": ["noise content"]
    }},
    {{
      "time_label": "last week",
      "signal": "latest state",
      "noise": ["noise content"]
    }}
  ],
  "query": "...",
  "target_memory": "(latest state only) ..."
}}"""


# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是记忆训练数据生成专家。你生成的数据用于训练AI的长期记忆模块。

核心要求：
- target_memory 必须是"记忆摘要"，不是对话回复
- 写入记录中必须混入与query无关的日常琐碎（噪音），模型需要学会忽略它们
- 写入记录必须有时间标记，体现时序感
- 后期 session 中用户自然引用早期事件，体现跨 session 关联"""

# Stage 1：单 session 热身
STAGE1_PROMPT = """生成一个记忆训练样本（热身阶段）。

主题：{topic}
要求：
1. 写入记录：4-8条，时间跨度约一周，用自然语言带时间标记
2. 其中1-3条是重要内容（与query相关），其余是日常噪音（与query无关）
3. query：关于重要内容的询问
4. target_memory：只包含重要内容的简洁摘要（50字以内）

重要：target_memory 是记忆摘要，不是回答问题，格式是"用户[做了/经历了/处于]..."

严格按以下JSON格式输出，不要有任何额外文字：
{{
  "writes": [
    {{"time": "周一", "content": "...", "is_noise": false}},
    {{"time": "周二", "content": "今天吃了外卖", "is_noise": true}},
    ...
  ],
  "query": "...",
  "target_memory": "用户..."
}}"""

# Stage 2：多 session 综合
STAGE2_PROMPT = """生成一个记忆训练样本（多 session 综合阶段）。

主题：{topic}
时间跨度：{time_span}
session 数：{n_sessions}
查询类型：{query_type}

要求：
1. 写入记录：10-20条，跨 {n_sessions} 个 session，每个 session 2-5条
2. 重要记录（2-5条）：影响用户生活/决策的关键事件，需在后续 session 中被引用
3. 噪音记录（6-12条）：日常琐碎，与query完全无关
4. 至少1处跨 session 引用（用户提到"上次说的"、"之前提过"等）
5. 如果是"变化历程"或"当前状态"类query，同一实体在不同时间有状态更新

target_memory 要求：
- 只综合与 query 相关的重要记录
- 忽略噪音内容
- 100字以内，口语化自然语言
- 不是回答query，而是描述相关背景记忆

严格按以下JSON格式输出，不要有任何额外文字：
{{
  "sessions": [
    {{
      "session_id": 1,
      "time_label": "3个月前",
      "writes": [
        {{"content": "...", "is_noise": false}},
        {{"content": "今天天气不错", "is_noise": true}}
      ]
    }}
  ],
  "query": "...",
  "target_memory": "..."
}}"""

# Stage 3：信息更新（同实体多次更新，target 只含最新状态）
STAGE3_PROMPT = """生成一个记忆训练样本（信息更新阶段）。

主题：{topic}
要求：体现同一实体在不同时间的状态更新（旧信息应被新信息覆盖）

必须包含：
1. 同一实体（人/项目/关系/状态）在3个不同时间点的变化记录
   例：["张三是实习生" → "张三转正成工程师" → "张三升任主管"]
2. 每个时间点混入1-2条噪音记录
3. query 询问该实体的"当前"状态
4. target_memory 只描述最新状态，不列举变化历程

严格按以下JSON格式输出，不要有任何额外文字：
{{
  "entity": "被追踪的实体名称",
  "updates": [
    {{
      "time_label": "6个月前",
      "signal": "实体的状态描述",
      "noise": ["噪音内容1"]
    }},
    {{
      "time_label": "3个月前",
      "signal": "更新后的状态",
      "noise": ["噪音内容"]
    }},
    {{
      "time_label": "上周",
      "signal": "最新状态",
      "noise": ["噪音内容"]
    }}
  ],
  "query": "...",
  "target_memory": "（只描述最新状态）..."
}}"""


# ---------------------------------------------------------------------------
# 解析生成的 JSON
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    if "</think>" in text:
        text = text[text.rfind("</think>") + len("</think>"):].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[\s\S]+\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# 样本格式转换
# ---------------------------------------------------------------------------


def _make_turn(content: str, time_label: str, importance: int) -> dict:
    """生成训练脚本兼容的 turn dict（importance 嵌入 turn，与 train_memory_oracle.py Stage 3 格式一致）。"""
    text = f"[{time_label}] {content}" if time_label else content
    return {"role": "write", "content": text, "importance": importance}


def _stage1_to_sample(parsed: dict) -> dict | None:
    writes = parsed.get("writes", [])
    query = parsed.get("query", "").strip()
    target = parsed.get("target_memory", "").strip()
    if not writes or not query or not target:
        return None

    history = []
    for w in writes:
        content = w.get("content", "").strip()
        if not content:
            continue
        history.append(_make_turn(
            content,
            time_label=w.get("time", "").strip(),
            importance=0 if w.get("is_noise", False) else 1,
        ))

    if not history:
        return None
    return {"history": history, "query": query, "target_memory": target, "source": "memory_v2_s1"}


def _stage2_to_sample(parsed: dict) -> dict | None:
    sessions = parsed.get("sessions", [])
    query = parsed.get("query", "").strip()
    target = parsed.get("target_memory", "").strip()
    if not sessions or not query or not target:
        return None

    history = []
    for sess in sessions:
        time_label = sess.get("time_label", "")
        for w in sess.get("writes", []):
            content = w.get("content", "").strip()
            if not content:
                continue
            history.append(_make_turn(
                content,
                time_label=time_label,
                importance=0 if w.get("is_noise", False) else 1,
            ))

    if not history:
        return None
    return {"history": history, "query": query, "target_memory": target, "source": "memory_v2_s2"}


def _stage3_to_sample(parsed: dict) -> dict | None:
    updates = parsed.get("updates", [])
    query = parsed.get("query", "").strip()
    target = parsed.get("target_memory", "").strip()
    if not updates or not query or not target:
        return None

    history = []
    for upd in updates:
        time_label = upd.get("time_label", "")
        signal = upd.get("signal", "").strip()
        if signal:
            history.append(_make_turn(signal, time_label=time_label, importance=1))
        for n in upd.get("noise", []):
            if isinstance(n, str):
                text = n.strip()
            elif isinstance(n, dict):
                text = (n.get("content") or n.get("text") or "").strip()
            else:
                text = ""
            if text:
                history.append(_make_turn(text, time_label=time_label, importance=0))

    if not history:
        return None
    return {"history": history, "query": query, "target_memory": target, "source": "memory_v2_s3"}


# ---------------------------------------------------------------------------
# 后处理验证
# ---------------------------------------------------------------------------


_ASSISTANT_REPLY_STARTS_ZH = (
    "好的", "当然", "没问题", "我来", "让我", "首先", "您好",
    "根据您", "当然可以", "我认为您", "我来帮", "我可以", "我建议",
    "请问", "非常感谢", "很高兴",
)
_ASSISTANT_REPLY_WORDS_ZH = ("您", "请您", "如您所")

_ASSISTANT_REPLY_STARTS_EN = (
    "Sure,", "Of course,", "Certainly,", "No problem,", "Let me", "First,",
    "I'd be happy", "I can help", "I think you", "I recommend", "I suggest",
    "Thank you for", "Great question",
)
_ASSISTANT_REPLY_WORDS_EN = ("you should", "you need to", "I recommend that you")


def _validate_sample(sample: dict, lang: str = "zh") -> tuple[bool, str]:
    """基本质量检查，返回 (is_valid, reason)。"""
    h = sample.get("history", [])
    q = sample.get("query", "")
    t = sample.get("target_memory", "")

    if len(h) < 3:
        return False, f"too few writes: {len(h)}"
    if len(q) < 5:
        return False, "query too short"
    if len(t) < 10:
        return False, "target too short"
    if len(t) > 300:
        return False, f"target too long: {len(t)}"

    # importance 嵌在每个 turn dict 里
    imp = [turn.get("importance", 1) for turn in h]
    if sum(imp) == 0:
        return False, "no important writes"
    if sum(imp) == len(imp):
        return False, "no noise writes (model won't learn to filter)"

    # target 不应该是助手回复风格
    bad_starts = _ASSISTANT_REPLY_STARTS_EN if lang == "en" else _ASSISTANT_REPLY_STARTS_ZH
    bad_words  = _ASSISTANT_REPLY_WORDS_EN  if lang == "en" else _ASSISTANT_REPLY_WORDS_ZH
    t_lower = t.lower()
    for bad_start in bad_starts:
        if t_lower.startswith(bad_start.lower()):
            return False, f"target looks like assistant reply: {t[:30]}"
    for word in bad_words:
        if word.lower() in t_lower:
            return False, f"target contains assistant phrase '{word}': {t[:40]}"

    return True, "ok"


# ---------------------------------------------------------------------------
# LLM Backends
# ---------------------------------------------------------------------------


class LocalBackend:
    def __init__(self, model_name: str, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("Loading local model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, system: str, user: str, max_new_tokens: int = 2048) -> str:
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        # Qwen3.5 支持 enable_thinking=False 关闭 thinking 模式；
        # 其他模型不支持该参数，fallback 到不带此参数。
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.tokenizer(text, return_tensors="pt").to(
            next(self.model.parameters()).device
        )
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.85,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


class OpenAIBackend:
    def __init__(self, model: str = "gpt-4o-mini", base_url: str | None = None) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        self.client = openai.OpenAI(base_url=base_url) if base_url else openai.OpenAI()
        self.model = model

    def generate(self, system: str, user: str, max_new_tokens: int = 2048) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_new_tokens,
            temperature=0.85,
        )
        return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# 生成循环
# ---------------------------------------------------------------------------


def _build_prompt(stage: int, lang: str = "zh") -> str:
    if lang == "en":
        topic = random.choice(ALL_TOPICS_EN)
        if stage == 1:
            return STAGE1_PROMPT_EN.format(topic=topic)
        elif stage == 2:
            span_label, n_sess = random.choice(TIME_SPANS_EN)
            qtype = random.choice(QUERY_TYPES_EN)
            return STAGE2_PROMPT_EN.format(
                topic=topic, time_span=span_label,
                n_sessions=n_sess, query_type=qtype,
            )
        else:
            return STAGE3_PROMPT_EN.format(topic=topic)
    else:
        topic = random.choice(ALL_TOPICS)
        if stage == 1:
            return STAGE1_PROMPT.format(topic=topic)
        elif stage == 2:
            span_label, n_sess = random.choice(TIME_SPANS)
            qtype = random.choice(QUERY_TYPES)
            return STAGE2_PROMPT.format(
                topic=topic, time_span=span_label,
                n_sessions=n_sess, query_type=qtype,
            )
        else:
            return STAGE3_PROMPT.format(topic=topic)


def _parse_and_convert(text: str, stage: int) -> dict | None:
    parsed = _extract_json(text)
    if parsed is None:
        return None
    if stage == 1:
        return _stage1_to_sample(parsed)
    elif stage == 2:
        return _stage2_to_sample(parsed)
    else:
        return _stage3_to_sample(parsed)


def _count_existing(path: Path) -> int:
    """统计已有合法行数，用于断点续跑。"""
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def generate_samples(
    backend: Any,
    stage: int,
    n_samples: int,
    output_path: Path,
    lang: str = "zh",
    retry_limit: int = 3,
    max_consecutive_failures: int = 50,
) -> tuple[int, int]:
    """生成 n_samples 条样本，追加写入 output_path。返回 (本次新增数, 本次拒绝数)。

    使用追加模式（"a"）——断点续跑安全，不会截断已有数据。
    启动时统计已有行数，从断点继续直到总量达到 n_samples。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    already = _count_existing(output_path)
    if already >= n_samples:
        log.info("Already have %d >= %d samples, nothing to do.", already, n_samples)
        return 0, 0
    remaining = n_samples - already
    log.info("Resuming: already=%d, need %d more.", already, remaining)

    count = 0       # 本次新增
    rejected = 0    # 本次拒绝
    consecutive_failures = 0

    with output_path.open("a", encoding="utf-8") as f:
        while count < remaining:
            if consecutive_failures >= max_consecutive_failures:
                log.error(
                    "Aborting: %d consecutive failures. Check prompt/model quality.",
                    consecutive_failures,
                )
                break

            prompt = _build_prompt(stage, lang)
            sys_prompt = SYSTEM_PROMPT_EN if lang == "en" else SYSTEM_PROMPT
            sample = None

            for retry in range(retry_limit):
                try:
                    raw = backend.generate(sys_prompt, prompt)
                    sample = _parse_and_convert(raw, stage)
                    if sample is not None:
                        break
                    log.debug("Parse failed (retry %d), raw[:120]: %s", retry + 1, raw[:120])
                except Exception as e:
                    log.warning("Generation error (retry %d/%d): %s", retry + 1, retry_limit, e)
                    time.sleep(2 ** retry)

            if sample is None:
                rejected += 1
                consecutive_failures += 1
                continue

            valid, reason = _validate_sample(sample, lang)
            if not valid:
                rejected += 1
                consecutive_failures += 1
                log.debug("Sample rejected: %s", reason)
                continue

            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            f.flush()
            count += 1
            consecutive_failures = 0  # 成功则重置

            if count % 100 == 0:
                total_done = already + count
                log.info(
                    "Stage %d | total=%d/%d | new=%d | rejected=%d | reject_rate=%.1f%%",
                    stage, total_done, n_samples, count, rejected,
                    100.0 * rejected / max(count + rejected, 1),
                )

    return count, rejected


# ---------------------------------------------------------------------------
# 验证模式：统计已生成数据的质量
# ---------------------------------------------------------------------------


def validate_file(path: Path) -> None:
    total = valid = noise_ratio_sum = 0.0
    errors: dict[str, int] = {}

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                errors["json_decode"] = errors.get("json_decode", 0) + 1
                continue

            ok, reason = _validate_sample(sample)
            if ok:
                valid += 1
                imp = [t.get("importance", 1) for t in sample.get("history", [])]
                if imp:
                    noise_ratio_sum += 1.0 - sum(imp) / len(imp)
            else:
                errors[reason] = errors.get(reason, 0) + 1

    log.info("=== Validation: %s ===", path)
    log.info("  Total:   %d", int(total))
    log.info("  Valid:   %d (%.1f%%)", int(valid), 100.0 * valid / max(total, 1))
    log.info("  Avg noise ratio: %.1f%%", 100.0 * noise_ratio_sum / max(valid, 1))
    if errors:
        log.info("  Rejection reasons:")
        for reason, cnt in sorted(errors.items(), key=lambda x: -x[1]):
            log.info("    %-40s %d", reason, cnt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Memory training data v2 builder")

    p.add_argument("--stage", type=int, choices=[1, 2, 3], default=2,
                   help="1=warmup(single-sess), 2=multi-sess, 3=update")
    p.add_argument("--output", required=True, help="Output JSONL file")
    p.add_argument("--n-samples", type=int, default=10000)
    p.add_argument("--backend", choices=["local", "openai"], default="local")
    p.add_argument("--model", default="Qwen/Qwen3.5-9B-Instruct",
                   help="Local model (for --backend local)")
    p.add_argument("--openai-model", default="gpt-4o-mini")
    p.add_argument("--openai-base-url", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lang", default="zh", choices=["zh", "en"],
                   help="Generation language: zh=Chinese (default), en=English")
    p.add_argument("--validate", action="store_true",
                   help="Validate existing output file instead of generating")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    output_path = Path(args.output)

    if args.validate:
        if not output_path.exists():
            log.error("File not found: %s", output_path)
            sys.exit(1)
        validate_file(output_path)
        return

    if args.backend == "local":
        backend = LocalBackend(args.model, args.device)
    else:
        backend = OpenAIBackend(args.openai_model, args.openai_base_url)

    topics_pool = ALL_TOPICS_EN if args.lang == "en" else ALL_TOPICS
    log.info("=== Memory Data v2 Stage %d [lang=%s] ===", args.stage, args.lang)
    log.info("  backend=%s  n=%d  output=%s", args.backend, args.n_samples, output_path)
    log.info("  topics pool: %d topics", len(topics_pool))

    count, rejected = generate_samples(
        backend, args.stage, args.n_samples, output_path, lang=args.lang
    )

    log.info("Done: %d generated, %d rejected (%.1f%% rejection rate)",
             count, rejected, 100.0 * rejected / max(count + rejected, 1))

    # 自动验证
    validate_file(output_path)


if __name__ == "__main__":
    main()
