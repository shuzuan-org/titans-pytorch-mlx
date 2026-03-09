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
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait as futures_wait
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

ALL_TOPICS_EN = TOPICS_PERSONAL_EN + TOPICS_WORK_EN + TOPICS_LIFE_EN  # 40 topics

# ---------------------------------------------------------------------------
# 多语言支持：统一使用 EN 模板，在 system prompt 里指定目标语言
# 训练时统一用 [WRITE]/[QUERY]/[MEMORY] markers（EN markers），内容可以是任意语言
# ---------------------------------------------------------------------------

# 支持的语言列表（lang_code → 语言名）
SUPPORTED_LANGS: dict[str, str] = {
    "zh": "Simplified Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "ar": "Arabic",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
    "vi": "Vietnamese",
}

# 非 zh/en 语言的语言约束追加到 system prompt
_LANG_INSTRUCTION = (
    "\n\nLanguage requirement: ALL content in the JSON output "
    "(writes/sessions, query, target_memory) MUST be written in {lang_name}. "
    "Do NOT use English for any conversation content — only use English for "
    "JSON field names and structural keywords."
)

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
    {{"time": "Tuesday", "content": "(mundane daily event unrelated to query, specific and unique)", "is_noise": true}}
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
        {{"content": "(key event related to the topic, specific details)", "is_noise": false}},
        {{"content": "(mundane daily event unrelated to query, specific and unique — no weather/food clichés)", "is_noise": true}}
      ]
    }}
  ],
  "query": "...",
  "target_memory": "..."
}}"""

# Stage 3 使用多步生成（INIT → CONT × N → QUERY），见 _generate_one_long_s3_sample()。
# 不使用单次调用方案：400-600 条 entry 单次输出质量差且容易用省略号敷衍。


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
    {{"time": "周一", "content": "（与主题相关的关键事件，具体描述）", "is_noise": false}},
    {{"time": "周二", "content": "（与查询完全无关的日常琐事，具体且唯一，禁用天气/饮食套话）", "is_noise": true}},
    {{"time": "周三", "content": "（另一条关键信息或状态变化）", "is_noise": false}}
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
1. 写入记录：10-15条，跨 {n_sessions} 个 session，每个 session 2-4条
2. 重要记录（5-8条）：影响用户生活/决策的关键事件，需在后续 session 中被引用
3. 噪音记录（4-6条）：日常琐碎，与query完全无关；噪音不超过总条数的40%
4. 至少1处跨 session 引用（用户提到"上次说的"、"之前提过"等）
5. 如果是"变化历程"或"当前状态"类query，同一实体在不同时间有状态更新

噪音多样性要求（重要）：
- 每条噪音内容必须具体且唯一，禁止重复同类短语
- 禁止使用"天气不错/阳光很好"或"吃了牛肉面/外卖/咖啡"作为模板，这些过于泛滥
- 噪音必须覆盖不同生活维度，例如：
  * 购物/快递：买了什么、快递延误、退货
  * 出行/交通：堵车、找停车位、骑车摔跤
  * 健康/睡眠：失眠、运动、感冒、体检
  * 娱乐：看了什么剧/书/展览、打游戏
  * 社交：朋友临时取消约会、邻居的事
  * 家务/家庭：修东西、宠物、父母来访
  * 工作琐事：系统崩溃、打印机卡纸、会议取消

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
        {{"content": "（与主题相关的关键事件）", "is_noise": false}},
        {{"content": "（与主题无关的具体日常琐事，每条都不同）", "is_noise": true}}
      ]
    }}
  ],
  "query": "...",
  "target_memory": "..."
}}"""

# Stage 3：分段生成（多步 API 调用），对标 LOCOMO 真实长对话场景（400-600 轮）
# 生成流程：INIT → CONT × (n_sessions-1) → QUERY，三类 prompt。

# ------ 中文版 ------

STAGE3_INIT_PROMPT_ZH = """生成一段长期对话记忆训练样本的第1个会话（共约{n_sessions}个会话）。

主题：{topic}

本会话要求：
1. 引入3-5个核心实体（人物/项目/关系），后续会话将持续追踪它们的演变
2. 包含{turns}条记录，每条一行，noise占40-60%（日常琐碎，与核心主题无关）
3. 不要加"X天前"类时间标签，用自然语言体现时序感
4. entities 列表中注明每个实体的初始角色（如"张三（求职者）"）

噪音多样性要求：噪音内容必须具体且唯一，覆盖不同生活维度（购物、出行、健康、娱乐、社交、家务、工作琐事等），严禁使用"吃外卖/吃什么/天气不错"等过于泛滥的模板。

严格按以下JSON格式输出，不要有任何额外文字：
{{
  "entities": ["（实体A，格式：姓名+角色，如 张三（初创公司创始人））", "（实体B）", "（实体C）"],
  "session_id": 1,
  "writes": [
    {{"content": "（实体A的初始状态或背景信息，具体描述）", "is_noise": false}},
    {{"content": "（与所有追踪实体无关的生活细节，具体且独特）", "is_noise": true}},
    {{"content": "（实体B的初始状态或引入事件）", "is_noise": false}},
    {{"content": "（不同生活维度的另一条琐事，类型与上条不同）", "is_noise": true}}
  ]
}}"""

STAGE3_CONT_PROMPT_ZH = """这是一段长期对话记忆的第{session_id}个会话（共约{n_sessions}个）。

持续追踪的实体：
{entities_str}

前序关键事件（摘要，按时间顺序）：
{key_events_str}

本会话要求：
1. 包含{turns}条记录，其中 is_noise=true 的条数必须在 {turns} 的 40-50% 之间
2. 自然推进实体状态：可以有新发展、状态改变、引用之前事件（"上次提到的..."）
3. 至少1条跨会话引用（提及之前发生的事）
4. 不加时间标签
5. 噪音内容要求：必须覆盖不同生活维度，每条噪音类型不同——
   购物/快递、出行/交通/停车、健康/睡眠/运动、娱乐（电影/书/游戏/展览）、
   社交（朋友/邻居/家人临时事件）、家务/宠物/维修、工作琐事（系统崩溃/会议取消等）
   严禁连续使用同类噪音（如多条都是购物）

严格按以下JSON格式输出，不要有任何额外文字：
{{
  "session_id": {session_id},
  "writes": [
    {{"content": "（具体的关键信息，推进某实体状态）", "is_noise": false}},
    {{"content": "（具体的日常琐事，与所有实体无关，类型独特）", "is_noise": true}}
  ]
}}"""

STAGE3_QUERY_PROMPT_ZH = """基于以下长期对话的全部关键事件，生成一个高质量的多跳查询问题和记忆摘要。

持续追踪的实体：
{entities_str}

全部关键事件（按时间顺序，共{n_signals}条）：
{key_events_str}

要求：
- query 必须是以下类型之一：
  a. 多跳问题：需综合不同时期的信息才能回答（如"X事件发生时，Y的状态是什么？"）
  b. 变化追踪：某实体从开始到最近的完整变化历程
  c. 跨实体关联：两个实体之间关系或相互影响的演变
- target_memory：综合相关事件，150字以内，自然语言，描述背景记忆（不是直接回答问题）

严格按以下JSON格式输出，不要有任何额外文字：
{{
  "query": "...",
  "target_memory": "..."
}}"""

# ------ 英文版 ------

STAGE3_INIT_PROMPT_EN = """Generate session 1 of a long-term memory training sample (approx. {n_sessions} sessions total).

Topic: {topic}

Requirements for this session:
1. Introduce 3-5 core entities (people/projects/relationships) that will evolve across subsequent sessions
2. Include {turns} write entries, with 40-60% noise (unrelated daily chatter)
3. No explicit time labels like "3 months ago" — use natural conversational cues for temporal flow
4. In the entities list, note each entity's initial role (e.g., "Alex (job seeker)")

Noise diversity rules: Every noise entry must be specific and unique, spanning different life domains (shopping, commute, health, entertainment, social, home/pets, work annoyances, etc.). NEVER use "had lunch/coffee" or "nice weather" — these are overused templates.

Output strictly as JSON, no extra text:
{{
  "entities": ["(Entity A: name + role, e.g. Alex (startup founder))", "(Entity B)", "(Entity C)"],
  "session_id": 1,
  "writes": [
    {{"content": "(Entity A's initial state or background, specific details)", "is_noise": false}},
    {{"content": "(Mundane daily event unrelated to all tracked entities, concrete and unique)", "is_noise": true}},
    {{"content": "(Entity B's initial state or first appearance event)", "is_noise": false}},
    {{"content": "(Different life domain mundane event, different category from the noise above)", "is_noise": true}}
  ]
}}"""

STAGE3_CONT_PROMPT_EN = """This is session {session_id} of a long-term memory conversation (approx. {n_sessions} sessions total).

Entities being tracked:
{entities_str}

Key events from previous sessions (chronological summary):
{key_events_str}

Requirements for this session:
1. Include {turns} write entries; is_noise=true entries must be 40-50% of {turns}
2. Naturally advance entity states: new developments, status changes, cross-session references ("remember when I mentioned...")
3. At least 1 cross-session callback referencing a past event
4. No time labels
5. Noise diversity: each noise entry must be from a different life domain — shopping/deliveries, commute/traffic/parking, health/sleep/exercise, entertainment (movies/books/games/exhibits), social (friends/neighbors/family dropping by), home/pets/repairs, work annoyances (system crash, meeting cancelled, etc.). NEVER repeat the same noise category consecutively.

Output strictly as JSON, no extra text:
{{
  "session_id": {session_id},
  "writes": [
    {{"content": "(concrete key event advancing an entity's state)", "is_noise": false}},
    {{"content": "(mundane daily event unrelated to all entities, unique and specific)", "is_noise": true}}
  ]
}}"""

STAGE3_QUERY_PROMPT_EN = """Based on the full key events from a long-term conversation, generate a high-quality multi-hop query and memory summary.

Entities tracked:
{entities_str}

All key events (chronological, {n_signals} total):
{key_events_str}

Requirements:
- query must be one of:
  a. Multi-hop: requires combining info from different time periods (e.g., "When X happened, what was Y's situation?")
  b. Change tracking: full evolution of one entity from start to most recent state
  c. Cross-entity: how two entities' relationship or mutual influence evolved
- target_memory: synthesizes relevant events, under 150 words, natural language, describes background memory (not a direct answer)

Output strictly as JSON, no extra text:
{{
  "query": "...",
  "target_memory": "..."
}}\""""


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


# ---------------------------------------------------------------------------
# Stage 3 多步生成（分段 API 调用，对标 LOCOMO ~500 轮对话规模）
# ---------------------------------------------------------------------------

# 每次生成参数随机范围
_S3_N_SESSIONS_RANGE = (15, 20)  # 总 session 数
_S3_TURNS_RANGE = (15, 25)       # 每个 session 的轮数
_S3_MAX_CONTEXT_SIGNALS = 20     # 传给 cont call 的最近关键事件数


def _fmt_entities(entities: list[str]) -> str:
    return "\n".join(f"- {e}" for e in entities)


def _fmt_key_events(signals: list[str], limit: int | None = None) -> str:
    lst = signals[-limit:] if limit and len(signals) > limit else signals
    return "\n".join(f"{i+1}. {e}" for i, e in enumerate(lst))


def _parse_session_writes(data: dict) -> list[dict]:
    """从 session API 响应中提取 writes 列表。"""
    return [
        w for w in data.get("writes", [])
        if isinstance(w, dict) and w.get("content", "").strip()
    ]


def _generate_one_long_s3_sample(
    backend: Any,
    sys_prompt: str,
    topic: str,
    lang: str,
    retry_limit: int,
) -> dict | None:
    """通过多步 API 调用生成一条长期记忆训练样本（~400-500 轮）。

    流程：
      Call 0 (init):  生成第 1 个 session + 引入实体列表
      Call 1..N-1 (cont): 依次生成后续 session，传入前序关键事件做上下文
      Call N (query):  基于全部关键事件生成 query + target_memory

    任何一步失败返回 None（调用方可重试）。
    """
    n_sessions = random.randint(*_S3_N_SESSIONS_RANGE)
    turns = random.randint(*_S3_TURNS_RANGE)

    is_zh = (lang == "zh")

    # ── Call 0: Init ──────────────────────────────────────────────────────────
    init_prompt = (STAGE3_INIT_PROMPT_ZH if is_zh else STAGE3_INIT_PROMPT_EN).format(
        topic=topic,
        n_sessions=n_sessions,
        turns=turns,
    )
    for attempt in range(retry_limit):
        try:
            raw = backend.generate(sys_prompt, init_prompt)
            init_data = _extract_json(raw)
            if init_data and init_data.get("writes") and init_data.get("entities"):
                break
            log.debug("S3 init parse fail (attempt %d)", attempt + 1)
        except Exception as e:
            log.warning("S3 init API error (attempt %d): %s", attempt + 1, e)
            time.sleep(min(2 ** attempt, 30))
    else:
        return None

    entities: list[str] = init_data.get("entities", [])
    all_sessions: list[dict] = [{"session_id": 1, "writes": _parse_session_writes(init_data)}]
    # 追踪所有 signal 内容，用于 cont 和 query 调用
    all_signals: list[str] = [
        w["content"] for w in all_sessions[0]["writes"] if not w.get("is_noise")
    ]

    # ── Calls 1..N-1: Continuation ───────────────────────────────────────────
    cont_tmpl = STAGE3_CONT_PROMPT_ZH if is_zh else STAGE3_CONT_PROMPT_EN
    for sid in range(2, n_sessions + 1):
        turns_this = random.randint(*_S3_TURNS_RANGE)
        cont_prompt = cont_tmpl.format(
            session_id=sid,
            n_sessions=n_sessions,
            entities_str=_fmt_entities(entities),
            key_events_str=_fmt_key_events(all_signals, _S3_MAX_CONTEXT_SIGNALS),
            turns=turns_this,
        )
        for attempt in range(retry_limit):
            try:
                raw = backend.generate(sys_prompt, cont_prompt)
                sess_data = _extract_json(raw)
                if sess_data and sess_data.get("writes"):
                    break
                log.debug("S3 cont sess %d parse fail (attempt %d)", sid, attempt + 1)
            except Exception as e:
                log.warning("S3 cont sess %d API error (attempt %d): %s", sid, attempt + 1, e)
                time.sleep(min(2 ** attempt, 30))
        else:
            return None

        writes = _parse_session_writes(sess_data)
        all_sessions.append({"session_id": sid, "writes": writes})
        all_signals.extend(w["content"] for w in writes if not w.get("is_noise"))

    # ── Call N: Query + Target ────────────────────────────────────────────────
    query_tmpl = STAGE3_QUERY_PROMPT_ZH if is_zh else STAGE3_QUERY_PROMPT_EN
    query_prompt = query_tmpl.format(
        entities_str=_fmt_entities(entities),
        key_events_str=_fmt_key_events(all_signals, limit=50),
        n_signals=len(all_signals),
    )
    for attempt in range(retry_limit):
        try:
            raw = backend.generate(sys_prompt, query_prompt)
            qdata = _extract_json(raw)
            if qdata and qdata.get("query") and qdata.get("target_memory"):
                break
            log.debug("S3 query parse fail (attempt %d)", attempt + 1)
        except Exception as e:
            log.warning("S3 query API error (attempt %d): %s", attempt + 1, e)
            time.sleep(min(2 ** attempt, 30))
    else:
        return None

    query = qdata["query"].strip()
    target = qdata["target_memory"].strip()

    # ── Flatten all sessions → history ────────────────────────────────────────
    history = []
    for sess in all_sessions:
        for w in sess["writes"]:
            content = w.get("content", "").strip()
            if content:
                history.append(_make_turn(content, time_label="", importance=0 if w.get("is_noise") else 1))

    return {"history": history, "query": query, "target_memory": target, "source": "memory_v2_s3"}


# ---------------------------------------------------------------------------
# Stage 3 自博弈验证（LLM-as-Judge）
# ---------------------------------------------------------------------------

_S3_JUDGE_SYS_ZH = "你是一位严格的记忆训练数据验证专家，负责评估数据质量是否满足长期记忆训练要求。"
_S3_JUDGE_SYS_EN = "You are a strict quality validator for long-term memory training data."

_S3_JUDGE_USER_ZH = """请验证以下记忆训练数据是否合格。

【数据统计】
总轮次：{n_turns}（重要={n_important}，噪音={n_noise}）

【前15条内容】
{first_turns}

【后15条内容】
{last_turns}

【Query】
{query}

【Target Memory（前300字）】
{target_preview}

【验证标准】（全部满足才通过）
1. 总轮次 ≥ 100
2. 每条内容具体自然，无省略号占位符（"..."）
3. 重要记录（is_noise=false）包含具体可记忆的信息
4. 后期内容有对早期事件的引用或状态更新（前后记忆对照）
5. 至少2个实体有跨轮次的明显状态变化
6. query 需综合多轮信息才能回答（不能只看1条记录）
7. target_memory 准确综合了相关背景，不是直接回答问题

只输出JSON，不要任何解释：{{"pass": true, "reason": "ok"}} 或 {{"pass": false, "reason": "具体不通过的原因"}}"""

_S3_JUDGE_USER_EN = """Validate the following memory training data.

[Statistics]
Total turns: {n_turns} (important={n_important}, noise={n_noise})

[First 15 entries]
{first_turns}

[Last 15 entries]
{last_turns}

[Query]
{query}

[Target Memory (first 300 chars)]
{target_preview}

[Validation criteria] (ALL must pass)
1. Total turns ≥ 100
2. Each entry is specific and natural, no ellipsis placeholders ("...")
3. Important entries (is_noise=false) contain concrete memorable facts
4. Later entries reference or update earlier events (cross-session callbacks)
5. At least 2 entities show clear state changes across turns
6. Query requires synthesizing info from multiple turns (not answerable from 1 entry)
7. Target memory accurately synthesizes relevant background, not a direct answer

Output JSON only: {{"pass": true, "reason": "ok"}} or {{"pass": false, "reason": "specific reason"}}"""


def _judge_stage3_sample(backend: Any, sample: dict, lang: str) -> bool:
    """自博弈验证：用 LLM 扮演验证师评估 stage3 样本质量。

    只传统计信息 + 首尾各 15 条（避免传入超大 context），验证 API 调用 max_tokens=150。
    验证 API 异常时默认放行，避免阻塞生成。
    """
    h = sample.get("history", [])
    n_turns = len(h)
    n_important = sum(1 for t in h if t.get("importance", 0) == 1)
    n_noise = n_turns - n_important

    first_turns = "\n".join(t.get("content", "") for t in h[:15])
    last_turns = "\n".join(t.get("content", "") for t in h[-15:])

    is_zh = (lang == "zh")
    judge_sys = _S3_JUDGE_SYS_ZH if is_zh else _S3_JUDGE_SYS_EN
    judge_user = (_S3_JUDGE_USER_ZH if is_zh else _S3_JUDGE_USER_EN).format(
        n_turns=n_turns,
        n_important=n_important,
        n_noise=n_noise,
        first_turns=first_turns,
        last_turns=last_turns,
        query=sample.get("query", "")[:300],
        target_preview=sample.get("target_memory", "")[:300],
    )

    try:
        raw = backend.generate(judge_sys, judge_user, max_new_tokens=150)
        if not raw:
            log.warning("S3 judge returned empty response (content filter?) — accepting sample")
            return True
        result = _extract_json(raw)
        if result is not None:
            passed = bool(result.get("pass", False))
            reason = result.get("reason", "")
            if not passed:
                log.warning("S3 judge REJECT: %s", reason)
            return passed
        log.warning("S3 judge response not parseable — accepting sample: %s", raw[:200])
        return True  # 无法解析时放行，避免阻塞生成
    except Exception as e:
        log.warning("S3 judge API error: %s — accepting sample", e)
        return True  # 验证异常时放行


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


# Stage 1: 单轮对话，target 是单条记忆摘要，短。
# Stage 2: 多 session 积累，target 合并多条记忆，最长。
# Stage 3: 长期多实体整合（LOCOMO 对标），target 综合多 session，允许更长。
_TARGET_MAX_LEN = {1: 300, 2: 800, 3: 1000}

# Stage 3 最少轮数（目标 225-500 轮，100轮是最低线，过滤严重退化输出）
_STAGE3_MIN_TURNS = 100


def _validate_sample(sample: dict, lang: str = "zh", stage: int = 1) -> tuple[bool, str]:
    """基本质量检查，返回 (is_valid, reason)。"""
    h = sample.get("history", [])
    q = sample.get("query", "")
    t = sample.get("target_memory", "")

    min_turns = _STAGE3_MIN_TURNS if stage == 3 else 3
    if len(h) < min_turns:
        return False, f"too few writes: {len(h)} (min={min_turns})"
    if len(q) < 5:
        return False, "query too short"
    if len(t) < 10:
        return False, "target too short"
    max_len = _TARGET_MAX_LEN.get(stage, 800)
    if len(t) > max_len:
        return False, f"target too long: {len(t)} > {max_len}"

    # importance 嵌在每个 turn dict 里
    imp = [turn.get("importance", 1) for turn in h]
    if sum(imp) == 0:
        return False, "no important writes"
    if sum(imp) == len(imp):
        return False, "no noise writes (model won't learn to filter)"

    # target 不应该是助手回复风格（只对 zh/en 做语言特定检查）
    if lang == "en":
        bad_starts, bad_words = _ASSISTANT_REPLY_STARTS_EN, _ASSISTANT_REPLY_WORDS_EN
    elif lang == "zh":
        bad_starts, bad_words = _ASSISTANT_REPLY_STARTS_ZH, _ASSISTANT_REPLY_WORDS_ZH
    else:
        bad_starts, bad_words = _ASSISTANT_REPLY_STARTS_EN, ()  # EN 格式兜底
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
            dtype=torch.bfloat16,
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
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        kwargs: dict = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self.client = openai.OpenAI(**kwargs, timeout=3600)
        self.model = model

    def generate(self, system: str, user: str, max_new_tokens: int = 2048) -> str:
        # 用 streaming 模式：代理持续收到 token 就不会因空闲而 504
        chunks = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_new_tokens,
            temperature=0.85,
            stream=True,
        )
        content = ""
        for chunk in chunks:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                content += delta
        return content


# ---------------------------------------------------------------------------
# 生成循环
# ---------------------------------------------------------------------------


def _get_system_prompt(lang: str) -> str:
    """返回数据生成用的 system prompt（告知目标语言）。"""
    if lang == "zh":
        return SYSTEM_PROMPT
    base = SYSTEM_PROMPT_EN
    if lang != "en":
        lang_name = SUPPORTED_LANGS.get(lang, lang)
        base += _LANG_INSTRUCTION.format(lang_name=lang_name)
    return base


def _pick_topic(lang: str) -> str:
    """随机选择一个话题（Stage 3 多步生成直接调用）。"""
    return random.choice(ALL_TOPICS if lang == "zh" else ALL_TOPICS_EN)


def _build_prompt(stage: int, lang: str = "zh") -> str:
    """Stage 1/2 返回格式化 prompt 字符串。Stage 3 不使用此函数（见 try_one）。"""
    if lang == "zh":
        topic = random.choice(ALL_TOPICS)
        if stage == 1:
            return STAGE1_PROMPT.format(topic=topic)
        else:  # stage == 2
            span_label, n_sess = random.choice(TIME_SPANS)
            qtype = random.choice(QUERY_TYPES)
            return STAGE2_PROMPT.format(
                topic=topic, time_span=span_label,
                n_sessions=n_sess, query_type=qtype,
            )
    else:
        topic = random.choice(ALL_TOPICS_EN)
        if stage == 1:
            return STAGE1_PROMPT_EN.format(topic=topic)
        else:  # stage == 2
            span_label, n_sess = random.choice(TIME_SPANS_EN)
            qtype = random.choice(QUERY_TYPES_EN)
            return STAGE2_PROMPT_EN.format(
                topic=topic, time_span=span_label,
                n_sessions=n_sess, query_type=qtype,
            )


def _parse_and_convert(text: str, stage: int) -> dict | None:
    parsed = _extract_json(text)
    if parsed is None:
        return None
    if stage == 1:
        return _stage1_to_sample(parsed)
    elif stage == 2:
        return _stage2_to_sample(parsed)
    else:
        raise ValueError(f"_parse_and_convert called with stage={stage}; stage 3 uses multi-step pipeline")


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


def _generate_samples_concurrent(
    backend: Any,
    stage: int,
    n_samples: int,
    output_path: Path,
    lang: str,
    retry_limit: int,
    concurrency: int,
) -> tuple[int, int]:
    """多线程并发版本，适用于 API backend（如 OpenAI 兼容接口）。

    主线程负责收集结果写文件，worker 线程只做 API 调用 + JSON 解析。
    支持动态补充 worker：每完成一个样本就提交一个新任务，保持 pool 满载。
    """
    already = _count_existing(output_path)
    if already >= n_samples:
        log.info("Already have %d >= %d, nothing to do.", already, n_samples)
        return 0, 0
    remaining = n_samples - already
    log.info("Concurrent [workers=%d]: need %d more.", concurrency, remaining)

    sys_prompt = _get_system_prompt(lang)

    def try_one() -> dict | None:
        """生成并验证一条样本。

        Stage 3：走多步流水线（INIT→CONT×N→QUERY），每步 retry 在
                 _generate_one_long_s3_sample() 内部处理，此处仅做外层验证。
        Stage 1/2：单次 API 调用 + 解析 + 验证，最多 retry_limit 次。
        """
        if stage == 3:
            topic = _pick_topic(lang)
            sample = _generate_one_long_s3_sample(backend, sys_prompt, topic, lang, retry_limit)
            if sample is None:
                return None
            valid, reason = _validate_sample(sample, lang, stage)
            if not valid:
                log.warning("Stage3 validate failed: %s | turns=%d",
                            reason, len(sample.get("history", [])))
                return None
            if not _judge_stage3_sample(backend, sample, lang):
                log.warning("Stage3 judge rejected | turns=%d", len(sample.get("history", [])))
                return None
            return sample

        # Stage 1/2: single API call
        prompt = _build_prompt(stage, lang)
        for retry in range(retry_limit):
            try:
                raw = backend.generate(sys_prompt, prompt, max_new_tokens=2048)
                sample = _parse_and_convert(raw, stage)
                if sample is None:
                    log.warning("Parse failed (retry %d/%d) — first 500 chars: %s",
                                retry + 1, retry_limit, raw[:500].replace("\n", "\\n"))
                    continue
                valid, reason = _validate_sample(sample, lang, stage)
                if not valid:
                    log.warning("Validate failed (retry %d/%d): %s | turns=%d",
                                retry + 1, retry_limit, reason,
                                len(sample.get("history", [])))
                    continue
                return sample
            except Exception as e:
                wait = min(2 ** retry, 30)
                log.warning("API error (retry %d/%d): %s — wait %ds", retry + 1, retry_limit, e, wait)
                time.sleep(wait)
        return None

    count = 0
    rejected = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            pending: set = set()

            def _fill() -> None:
                slots = concurrency - len(pending)
                for _ in range(slots):
                    pending.add(executor.submit(try_one))

            _fill()

            while count < remaining:
                done, pending = futures_wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    if count >= remaining:
                        break
                    result = fut.result()
                    if result is not None:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f.flush()
                        count += 1
                        if count % 100 == 0:
                            log.info(
                                "Stage %d [concurrent] | total=%d/%d | new=%d | rejected=%d | rate=%.1f%%",
                                stage, already + count, n_samples, count, rejected,
                                100.0 * rejected / max(count + rejected, 1),
                            )
                    else:
                        rejected += 1

                _fill()

    return count, rejected


def generate_samples(
    backend: Any,
    stage: int,
    n_samples: int,
    output_path: Path,
    lang: str = "zh",
    retry_limit: int = 3,
    max_consecutive_failures: int = 50,
    concurrency: int = 1,
) -> tuple[int, int]:
    """生成 n_samples 条样本，追加写入 output_path。返回 (本次新增数, 本次拒绝数)。

    concurrency > 1 时使用多线程并发模式（适合 API backend）。
    使用追加模式（"a"）——断点续跑安全，不会截断已有数据。
    """
    if concurrency > 1:
        return _generate_samples_concurrent(
            backend, stage, n_samples, output_path, lang, retry_limit, concurrency
        )

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

    sys_prompt = _get_system_prompt(lang)
    with output_path.open("a", encoding="utf-8") as f:
        while count < remaining:
            if consecutive_failures >= max_consecutive_failures:
                log.error(
                    "Aborting: %d consecutive failures. Check prompt/model quality.",
                    consecutive_failures,
                )
                break

            # Stage 3: 多步流水线（INIT→CONT×N→QUERY），不走单次调用
            if stage == 3:
                topic = _pick_topic(lang)
                sample = _generate_one_long_s3_sample(backend, sys_prompt, topic, lang, retry_limit)
            else:
                prompt = _build_prompt(stage, lang)
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

            valid, reason = _validate_sample(sample, lang, stage)
            if not valid:
                rejected += 1
                consecutive_failures += 1
                log.debug("Sample rejected: %s", reason)
                continue

            if stage == 3 and not _judge_stage3_sample(backend, sample, lang):
                rejected += 1
                consecutive_failures += 1
                log.warning("Stage3 judge rejected | turns=%d", len(sample.get("history", [])))
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


def validate_file(path: Path, lang: str, stage: int) -> None:
    total = 0
    valid = 0
    noise_ratio_sum = 0.0
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

            ok, reason = _validate_sample(sample, lang=lang, stage=stage)
            if ok:
                valid += 1
                imp = [t.get("importance", 1) for t in sample.get("history", [])]
                if imp:
                    noise_ratio_sum += 1.0 - sum(imp) / len(imp)
            else:
                errors[reason] = errors.get(reason, 0) + 1

    log.info("=== Validation: %s ===", path)
    log.info("  Total:   %d", total)
    log.info("  Valid:   %d (%.1f%%)", valid, 100.0 * valid / max(total, 1))
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
    p.add_argument("--openai-model", default="MiniMax-Text-01")
    p.add_argument("--openai-base-url", default=None)
    p.add_argument("--openai-api-key", default=None,
                   help="API key (overrides OPENAI_API_KEY env var)")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Concurrent API calls (>1 uses thread pool, recommended for API backend)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lang", default="zh", choices=sorted(SUPPORTED_LANGS.keys()),
                   help="Content language (default: zh). Markers are always [WRITE]/[QUERY]/[MEMORY].")
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
        validate_file(output_path, lang=args.lang, stage=args.stage)
        return

    if args.backend == "local":
        if args.stage >= 2:
            log.error(
                "Stage %d 禁止使用 --backend local。"
                "0.8B 模型质量不足以生成 Stage 2/3 复杂多 session 训练数据，"
                "且速度极慢（约 0.8 条/min/worker vs API 的 40+ 条/min）。"
                "请改用 --backend openai --openai-model MiniMax-Text-01 "
                "--openai-base-url https://mini.origintask.cn/v1 --concurrency 10",
                args.stage,
            )
            sys.exit(1)
        backend = LocalBackend(args.model, args.device)
    else:
        backend = OpenAIBackend(args.openai_model, args.openai_base_url, args.openai_api_key)

    topics_pool = ALL_TOPICS_EN if args.lang == "en" else ALL_TOPICS
    log.info("=== Memory Data v2 Stage %d [lang=%s] ===", args.stage, args.lang)
    log.info("  backend=%s  n=%d  output=%s", args.backend, args.n_samples, output_path)
    log.info("  topics pool: %d topics", len(topics_pool))

    count, rejected = generate_samples(
        backend, args.stage, args.n_samples, output_path,
        lang=args.lang, concurrency=args.concurrency,
    )

    log.info("Done: %d generated, %d rejected (%.1f%% rejection rate)",
             count, rejected, 100.0 * rejected / max(count + rejected, 1))

    # 自动验证
    validate_file(output_path, lang=args.lang, stage=args.stage)


if __name__ == "__main__":
    main()
