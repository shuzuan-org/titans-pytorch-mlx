from __future__ import annotations

from pathlib import Path

DEFAULT_STAGE1_PROMPT_VERSION = "v2"
STAGE1_PROMPT_VERSIONS = ("v2", "v3", "v4")


def normalize_stage1_text(text: str) -> str:
    return text.strip()


def validate_stage1_prompt_version(prompt_version: str) -> str:
    if prompt_version not in STAGE1_PROMPT_VERSIONS:
        raise ValueError(
            f"Unsupported prompt_version: {prompt_version}. Expected one of {STAGE1_PROMPT_VERSIONS}"
        )
    return prompt_version


def build_stage1_question_prompt(question: str, prompt_version: str = DEFAULT_STAGE1_PROMPT_VERSION) -> str:
    prompt_version = validate_stage1_prompt_version(prompt_version)
    question = normalize_stage1_text(question)
    if prompt_version == "v2":
        return f"问题：{question}\n答案："
    if prompt_version == "v3":
        return (
            "根据当前历史信息，回答下列问题。\n"
            "请只输出最终答案，不要解释。\n"
            f"问题：{question}\n"
            "回答："
        )
    return (
        "根据当前历史信息，先聚焦与问题最相关的历史要点，再给出最终答案。\n"
        "请只输出最终答案，不要解释。\n"
        f"问题：{question}\n"
        "最终答案："
    )


def default_stage1_checkpoint_dir(prompt_version: str) -> str:
    prompt_version = validate_stage1_prompt_version(prompt_version)
    return f"checkpoints/stage1_timeline_{prompt_version}"



def default_stage1_checkpoint_path(prompt_version: str) -> Path:
    return Path(default_stage1_checkpoint_dir(prompt_version)) / "final_model" / "stage1_state.pt"
