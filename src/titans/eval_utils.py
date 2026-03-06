# Copyright 2026 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
共享评测工具：模型加载、答案生成、中文 F1 评分。

供 tptt_babilong.py 和 longbench_tptt.py 使用，避免重复代码。
"""

from __future__ import annotations

import logging
import string
from collections import Counter
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 中文 F1 评分（对齐 LongBench 官方实现）
# ---------------------------------------------------------------------------

# 完整中文标点集合（LongBench 官方使用的同款列表）
_CN_PUNCTUATION = (
    "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～"
    "｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛""„‟…‧﹏."
)
_ALL_PUNCTUATION = set(_CN_PUNCTUATION + string.punctuation)


def normalize_zh(text: str) -> str:
    """小写 + 去除中英文标点 + 合并空白。对齐 LongBench 官方 normalize_answer_zh。"""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _ALL_PUNCTUATION)
    text = " ".join(text.split())
    return text


def _char_tokens(text: str) -> list[str]:
    """字符级分词：中文每字一 token，非中文按空格切词后再按字符拆。"""
    tokens: list[str] = []
    for word in text.split():
        for ch in word:
            tokens.append(ch)
    return tokens


def f1_score_zh(prediction: str, ground_truth: str) -> float:
    """字符级 F1，先 normalize 再计算。"""
    pred = normalize_zh(prediction)
    gold = normalize_zh(ground_truth)
    pred_tokens = _char_tokens(pred)
    gold_tokens = _char_tokens(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_example(prediction: str, answers: list[str]) -> float:
    """取预测与所有参考答案 F1 的最大值。"""
    if not answers:
        return 0.0
    return max(f1_score_zh(prediction, ans) for ans in answers)


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------


def load_baseline(
    model_name: str,
    device: str,
    dtype: torch.dtype,
):
    """加载原始 Qwen2.5（无 memory 注入）作为 baseline。"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("需要 transformers 库")

    logger.info("加载 baseline 模型: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map={"": device}, trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def _resolve_checkpoint(checkpoint: str | Path) -> Path:
    """将 checkpoint 参数解析为具体的 .pt 文件路径。

    接受两种形式：
    - 直接 .pt 文件路径（如 step_0005000.pt）
    - 包含 step_*.pt 文件的目录（取字母序最大，即步数最高的那个）

    Raises:
        ValueError: 路径既不是 .pt 文件也不是目录
        FileNotFoundError: 文件/目录不存在，或目录内无 step_*.pt
    """
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint 路径不存在: {path}")

    if path.is_file():
        if path.suffix != ".pt":
            raise ValueError(f"Checkpoint 文件必须以 .pt 结尾，得到: {path}")
        return path

    if path.is_dir():
        # 检查是否是 peft 格式（调用方处理）
        if (path / "adapter_config.json").exists():
            return path  # 标记为 peft 目录，由调用方处理
        pt_files = sorted(path.glob("step_*.pt"))
        if not pt_files:
            raise FileNotFoundError(
                f"目录 {path} 中没有找到 step_*.pt 文件，"
                f"也没有 adapter_config.json（peft 格式）"
            )
        chosen = pt_files[-1]
        logger.info("目录模式：选择最新 checkpoint %s（共 %d 个）", chosen.name, len(pt_files))
        return chosen

    raise ValueError(f"Checkpoint 必须是 .pt 文件或目录，得到: {path}")


def _infer_lora_config(lora_state: dict) -> dict | None:
    """从 lora_state 的 key/shape 推断 LoRA 配置。

    key 形如 'base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight'
    rank  = lora_A weight 的 shape[0]
    target_modules = lora_A 前面一级的模块名（如 q_proj, v_proj...）
    """
    if not lora_state:
        return None
    target_modules: set[str] = set()
    rank: int | None = None
    for name, tensor in lora_state.items():
        if "lora_A" in name:
            if rank is None:
                rank = tensor.shape[0]
            parts = name.split(".")
            try:
                idx = parts.index("lora_A")
                if idx > 0:
                    target_modules.add(parts[idx - 1])
            except ValueError:
                pass
    if rank is None:
        return None
    return {"r": rank, "lora_alpha": rank * 2, "target_modules": sorted(target_modules)}


def load_tptt_model(
    model_name: str,
    device: str,
    dtype: torch.dtype,
    checkpoint: str | None,
    num_memory_layers: int,
):
    """加载注入了 NeuralLongTermMemory 的 Qwen2.5 模型。

    checkpoint 可以是：
    - None：使用随机初始化的 memory（测试注入正确性）
    - 具体 .pt 文件路径：加载该文件（自动推断并应用 LoRA）
    - 含 step_*.pt 的目录：自动选最新文件
    - peft 格式目录（含 adapter_config.json）：通过 PeftModel 加载
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("需要 transformers 库")

    from titans.config import TitansConfig
    from titans.tptt import inject_memory_into_qwen

    logger.info("加载 TPTT 模型: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map={"": device}, trust_remote_code=True,
    )

    model_dim: int = model.config.hidden_size
    mem_cfg = TitansConfig(dim=model_dim, num_memory_layers=num_memory_layers, use_conv=False)
    inject_memory_into_qwen(model, mem_cfg)
    logger.info("NeuralLongTermMemory 注入完成（dim=%d，layers=%d）", model_dim, num_memory_layers)

    if checkpoint is not None:
        ckpt_path = _resolve_checkpoint(checkpoint)

        # peft 格式目录（training 结束时 model.save_pretrained 生成）
        if ckpt_path.is_dir() and (ckpt_path / "adapter_config.json").exists():
            try:
                from peft import PeftModel
            except ImportError:
                raise ImportError("加载 peft checkpoint 需要：pip install peft")
            model = PeftModel.from_pretrained(model, str(ckpt_path))
            logger.info("已加载 peft checkpoint: %s", ckpt_path)
        else:
            # step_*.pt 格式
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            memory_state = ckpt.get("memory_state", {})
            lora_state = ckpt.get("lora_state", {})

            lora_cfg = _infer_lora_config(lora_state)
            if lora_cfg:
                # 有 LoRA：先给 eval 模型应用相同的 LoRA 结构，
                # 使参数名带上 "base_model.model." 前缀，与 checkpoint 名称对齐；
                # 再直接加载，无需 strip 前缀。
                try:
                    from peft import LoraConfig, TaskType, get_peft_model
                    lora_config = LoraConfig(
                        r=lora_cfg["r"],
                        lora_alpha=lora_cfg["lora_alpha"],
                        target_modules=lora_cfg["target_modules"],
                        bias="none",
                        task_type=TaskType.CAUSAL_LM,
                    )
                    model = get_peft_model(model, lora_config)
                    logger.info(
                        "已应用 LoRA (r=%d, target=%s)",
                        lora_cfg["r"], lora_cfg["target_modules"],
                    )
                    combined = {**memory_state, **lora_state}
                except ImportError:
                    logger.warning("peft 未安装，LoRA 权重跳过（pip install peft）；仅加载 memory")
                    combined = memory_state
                    # 无 peft 包装，需 strip 前缀
                    _P = "base_model.model."
                    combined = {(k[len(_P):] if k.startswith(_P) else k): v
                                for k, v in combined.items()}
            else:
                # 无 LoRA，strip 前缀后加载 memory 权重
                _P = "base_model.model."
                combined = {(k[len(_P):] if k.startswith(_P) else k): v
                            for k, v in memory_state.items()}

            model_params = dict(model.named_parameters())
            loaded, unexpected = [], []
            for name, tensor in combined.items():
                if name in model_params:
                    model_params[name].data.copy_(tensor)
                    loaded.append(name)
                else:
                    unexpected.append(name)
            missing = [n for n in model_params if n not in combined]
            logger.info(
                "已加载 checkpoint: %s (step=%d, loaded=%d, missing=%d, unexpected=%d)",
                ckpt_path.name, ckpt.get("step", -1),
                len(loaded), len(missing), len(unexpected),
            )

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_answer(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 64,
    max_ctx: int = 32768,
    device=None,
) -> str:
    """生成答案。

    注意：use_cache=True（HuggingFace 默认）时，generation 阶段每个新 token 都会
    触发一次 decoder layer 前向，QwenLayerWithMemory 的 memory_state 会在生成过程中
    持续积累。这是当前实现的已知行为，不影响单样本评测的正确性，但会使同一文档的
    多次生成结果略有差异。如需完全确定性，可在调用前 reset_memory_states(model)。
    """
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_ctx)
    if device is None:
        device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    output_ids = model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    generated = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip().split("\n")[0].strip()
