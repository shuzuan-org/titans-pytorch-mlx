# Copyright 2026 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
MemoryOracle — 独立记忆管理器，给主 LLM（GPT-4 等）供应上下文。

核心机制
--------
Write 模式：每条新消息 → NLM 解析梯度更新权重，无生成（< 10ms on GPU）
Read  模式：当前 query → 从 NLM 权重中生成短文字摘要 → 传给主 LLM（< 800ms）

NLM 权重 = 持久化记忆存储，跨 session 通过 save/load 序列化到磁盘。

使用示例
--------
    oracle = MemoryOracle.from_pretrained(
        "checkpoints/memory_oracle_stage3/",
        device="cuda",
    )
    oracle.write("用户：我最近在面试三家公司...")
    oracle.write("助手：建议你重点关注技术成长空间...")
    summary = oracle.read("我现在遇到 offer 选择困难，怎么办？")
    # → "用户面试了3家公司，薪资要求30K+，更看重技术成长..."

    oracle.save("memory_states/session_001.pt")
    # 下次接着用：oracle.load("memory_states/session_001.pt")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from titans.config import TitansConfig
from titans.memory import MemoryState
from titans.qwen35_injection import (
    Qwen35LayerWithMemory,
    freeze_memory_updates,
    get_nlm_states,
    inject_memory_into_qwen35,
    reset_memory_states,
    set_nlm_states,
    unfreeze_memory_updates,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Read 模式 prompt 模板
# ---------------------------------------------------------------------------

# 必须与 scripts/train_memory_oracle.py 的 SYS_PROMPT / WRITE_PREFIX / READ_PREFIX /
# TARGET_PREFIX 完全一致，否则推理格式与训练格式不匹配，模型输出乱码。
_SYS_PROMPT = (
    "你是一个记忆助手，负责记录和召回对话历史中的关键信息。"
    "在【写入】模式下更新记忆；在【查询】模式下生成精简记忆摘要。"
)
_WRITE_PREFIX = "【写入】"
_READ_PREFIX = "【查询】"
_TARGET_PREFIX = "【记忆】"

# 对外暴露，供训练脚本 import 避免字面量重复
MEMORY_SYS_PROMPT = _SYS_PROMPT
MEMORY_WRITE_TEMPLATE = _WRITE_PREFIX + "{message}\n"
MEMORY_READ_TEMPLATE = _READ_PREFIX + "{query}\n" + _TARGET_PREFIX

# ---------------------------------------------------------------------------
# English prompt constants (for English from-scratch training)
# ---------------------------------------------------------------------------

_SYS_PROMPT_EN = (
    "You are a memory assistant. Record and recall key information from conversation history. "
    "In [WRITE] mode, update memory. In [QUERY] mode, generate a concise memory summary."
)
_WRITE_PREFIX_EN = "[WRITE]"
_READ_PREFIX_EN = "[QUERY]"
_TARGET_PREFIX_EN = "[MEMORY]"

MEMORY_SYS_PROMPT_EN = _SYS_PROMPT_EN
MEMORY_WRITE_TEMPLATE_EN = _WRITE_PREFIX_EN + "{message}\n"
MEMORY_READ_TEMPLATE_EN = _READ_PREFIX_EN + "{query}\n" + _TARGET_PREFIX_EN


# ---------------------------------------------------------------------------
# MemoryOracle
# ---------------------------------------------------------------------------


class MemoryOracle:
    """Qwen3.5-0.8B + NLM 的独立记忆管理器。

    Attributes:
        model:      注入 NLM 后的 Qwen3.5 模型（含 LoRA，如已微调）
        tokenizer:  对应 tokenizer
        device:     运行设备
        dtype:      计算精度（默认 bfloat16）
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_seq_len: int = 8192,
        max_read_new_tokens: int = 200,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.max_read_new_tokens = max_read_new_tokens
        # 写入缓冲区：所有 write() 调用的消息缓存于此。
        # read() 将 buffer 中所有消息与 query 拼接成一条完整序列，一次 forward，
        # 这与训练时的 single-pass 格式完全一致（NLM attention context 正确）。
        # commit() 将 buffer 提交为 background_state 并清空，支持超长 session。
        self._write_buffer: list[str] = []
        # 跨 session 的持久背景记忆：由 load() 设置，或由 commit() 更新。
        # read() 先恢复此 state，再在其基础上处理 _write_buffer（当前 session）。
        self._background_state: dict[int, Any] | None = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path,
        base_model: str = "Qwen/Qwen3.5-0.8B",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        num_memory_layers: int = 1,
        memory_lr: float = 0.1,
        memory_momentum: float = 0.9,
        memory_decay: float = 0.01,
        **kwargs: Any,
    ) -> "MemoryOracle":
        """NLM 注入 + LoRA + checkpoint 一键加载。

        Args:
            checkpoint_dir: 含 final/（peft 格式）或 step_*.pt 文件的目录。
                            若目录下有 adapter_config.json 则按 peft 格式加载。
                            否则尝试加载最新 step_*.pt。
            base_model:     HuggingFace model ID，默认 Qwen3.5-0.8B。
            device:         "cuda" or "cpu".
            dtype:          计算精度。
            **kwargs:       透传给 cls.__init__。
        """
        ckpt_dir = Path(checkpoint_dir)

        # 加载 tokenizer
        tokenizer_src = str(ckpt_dir) if (ckpt_dir / "tokenizer.json").exists() else base_model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # 加载基模型
        logger.info("Loading base model: %s", base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=dtype,
            device_map={"": device},
            trust_remote_code=True,
        )

        # NLM 注入：Qwen3.5 多模态模型的 hidden_size 在 text_config 里
        if hasattr(model.config, "text_config"):
            model_dim: int = model.config.text_config.hidden_size
        else:
            model_dim: int = model.config.hidden_size
        mem_cfg = TitansConfig(
            dim=model_dim,
            num_memory_layers=num_memory_layers,
            memory_lr=memory_lr,
            memory_momentum=memory_momentum,
            memory_decay=memory_decay,
            use_conv=False,
        )
        logger.info("Injecting NLM (dim=%d, num_memory_layers=%d)", model_dim, num_memory_layers)
        inject_memory_into_qwen35(model, mem_cfg)

        # 加载 checkpoint 权重（返回值可能是新的 PeftModel 对象）
        model = cls._load_weights(model, ckpt_dir, device)

        model.eval()
        return cls(model, tokenizer, device=device, dtype=dtype, **kwargs)

    @staticmethod
    def _load_weights(model: nn.Module, ckpt_dir: Path, device: str) -> nn.Module:
        """Try peft adapter first, then fall back to step_*.pt.

        Returns the (possibly new) model object — PeftModel.from_pretrained and
        merge_and_unload both return new objects, so the caller MUST use the
        return value, not rely on in-place mutation.
        """
        final_dir = ckpt_dir / "final"

        if final_dir.is_dir() and (final_dir / "adapter_config.json").exists():
            try:
                from peft import PeftModel
                logger.info("Loading peft adapter from %s", final_dir)
                # 不调用 merge_and_unload()：
                # - merge 后 LoRA 权重直接写入 base model，若 LoRA 过拟合则破坏 base model 的生成能力
                # - 保留 peft 模式，LoRA 权重以低秩矩阵形式存在，对 base model 影响可控
                model = PeftModel.from_pretrained(model, str(final_dir))
                logger.info("Peft adapter loaded (not merged).")
                return model
            except Exception as e:
                logger.warning("peft load failed (%s), trying step_*.pt", e)

        # Fallback: latest step_*.pt
        step_files = sorted(ckpt_dir.glob("step_*.pt"))
        if step_files:
            ckpt_path = step_files[-1]
            logger.info("Loading checkpoint: %s", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            all_params = dict(model.named_parameters())

            def _load_state(state_dict: dict) -> None:
                for name, tensor in state_dict.items():
                    # Strip peft prefix if present
                    name = name.replace("base_model.model.", "", 1)
                    if name in all_params:
                        all_params[name].data.copy_(tensor.to(all_params[name].device))
                    else:
                        logger.debug("Skipping unknown param: %s", name)

            for key in ("memory_state", "lora_state"):
                if key in ckpt:
                    _load_state(ckpt[key])
            logger.info("Checkpoint loaded from %s", ckpt_path)
        else:
            logger.info("No checkpoint found in %s — using random init", ckpt_dir)

        return model

    # ------------------------------------------------------------------
    # Write / Read
    # ------------------------------------------------------------------

    def write(self, message: str) -> None:
        """将一条消息追加到写入缓冲区（延迟写入 NLM）。

        NLM 不在此时更新——所有消息在 read() / commit() / save() 时才被
        一次性 forward 进 NLM，与训练时的 single-pass 格式完全一致。

        Args:
            message: 原始消息文本。
        """
        self._write_buffer.append(message)

    @torch.no_grad()
    def read(self, query: str) -> str:
        """从 NLM 记忆中召回与 query 相关的摘要。

        两段式推理，与训练格式对齐，同时防止生成 token 污染 NLM state：

        Pass 1（NLM unfrozen）：
            Forward SYS_PROMPT + [写入]×N + [查询]query 完整序列，
            NLM 在此过程中积累 state（与训练 single-pass 一致）。
            保留 KV cache 供 Pass 2 使用。

        Pass 2（NLM frozen）：
            以 [记忆] marker 为 input、Pass 1 的 KV cache 为上下文，
            调用 generate() 生成记忆摘要。
            NLM frozen → 生成 token 不写入记忆，消除状态污染。

        session 生命周期：
            - 若 load() 曾设置 background_state，先恢复之（跨 session 背景记忆），
              再在其基础上处理当前 _write_buffer（当前 session）。
            - 这样 save/load 持久化的 NLM state 真正参与推理。

        Args:
            query: 查询文本。

        Returns:
            记忆摘要字符串（100 字以内）。
        """
        # ── 步骤 0：重置 NLM，恢复跨 session 背景记忆 ──────────────────────
        reset_memory_states(self.model)
        if self._background_state is not None:
            set_nlm_states(self.model, self._background_state)
        unfreeze_memory_updates(self.model)

        # ── 步骤 1：Pass 1 — 构建 write prefix，forward 更新 NLM state ──────
        write_prefix = _SYS_PROMPT + "\n"
        for msg in self._write_buffer:
            write_prefix += MEMORY_WRITE_TEMPLATE.format(message=msg)
        write_prefix += _READ_PREFIX + query + "\n"  # 不含 [TARGET]

        write_inputs = self.tokenizer(
            write_prefix,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_seq_len,
        ).to(self.device)

        write_output = self.model(
            input_ids=write_inputs["input_ids"],
            attention_mask=write_inputs["attention_mask"],
            use_cache=True,
        )

        # ── 步骤 2：freeze NLM，防止生成 token 写入记忆 ─────────────────────
        freeze_memory_updates(self.model)

        # ── 步骤 3：Pass 2 — 从 [记忆] marker 开始 generate ─────────────────
        target_marker_ids = self.tokenizer(
            _TARGET_PREFIX,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(self.device)

        # attention_mask 覆盖 past（write prefix）+ current（target marker）
        # 显式传入避免框架升级时 prepare_inputs_for_generation 行为变化
        past_len = write_inputs["input_ids"].shape[1]
        marker_len = target_marker_ids.shape[1]
        gen_attn_mask = torch.ones(1, past_len + marker_len, dtype=torch.long, device=self.device)

        try:
            output_ids = self.model.generate(
                input_ids=target_marker_ids,
                attention_mask=gen_attn_mask,
                past_key_values=write_output.past_key_values,
                max_new_tokens=self.max_read_new_tokens,
                do_sample=False,           # greedy：记忆召回必须确定性
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.3,
            )
        finally:
            unfreeze_memory_updates(self.model)

        # output_ids 包含 target_marker_ids 本身，跳过它们取生成部分
        n_marker = target_marker_ids.shape[1]
        new_tokens = output_ids[0, n_marker:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # 截断训练格式 token：模型有时在正确答案后继续生成假对话
        for stop in (_WRITE_PREFIX, _READ_PREFIX, _TARGET_PREFIX, "\n\n", "<think>"):
            idx = text.find(stop)
            if idx != -1:
                text = text[:idx].strip()
        return text

    # ------------------------------------------------------------------
    # Memory state persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """将 NLM 权重状态序列化到磁盘。

        先将 _write_buffer 中所有待处理消息 forward 进 NLM（commit），
        确保保存的 state 包含当前 session 所有写入内容。
        只保存 NLM memory_state（weights + momentum），体积小（~数 MB）。
        LoRA 和 NLM 结构参数用 from_pretrained 加载，不在这里存储。
        """
        # 提交 buffer → 确保保存的 state 反映所有已写消息
        self._rebuild_nlm_state()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        states = get_nlm_states(self.model)
        # 转为 CPU 再序列化，方便跨机器使用
        cpu_states = {
            k: (
                None if v is None
                else {
                    "weights": [w.cpu() for w in v.weights],
                    "momentum": [m.cpu() for m in v.momentum],
                }
            )
            for k, v in states.items()
        }
        torch.save(cpu_states, path)
        logger.info("NLM states saved → %s", path)

    def load(self, path: str | Path) -> None:
        """从磁盘恢复 NLM 权重状态，设为跨 session 背景记忆。

        加载后，下次 read() 会先恢复此 state，再在其基础上处理 _write_buffer，
        确保跨 session 的持久记忆真正参与推理（不会被 read() 的 reset 覆盖）。

        用 weights_only=False：NLM state 文件是本项目自己写出的结构化数据
        （dict + list of tensors + None），不是第三方 pickle，不存在任意代码执行风险。
        """
        path = Path(path)
        cpu_states = torch.load(path, map_location="cpu", weights_only=False)
        states = {
            k: (
                None if v is None
                else MemoryState(
                    weights=v["weights"],
                    momentum=v["momentum"],
                )
            )
            for k, v in cpu_states.items()
        }
        # 同时设置 background_state（供 read() 恢复）和 live NLM state
        self._background_state = states
        set_nlm_states(self.model, states)
        logger.info("NLM states loaded ← %s", path)

    def reset(self) -> None:
        """清除所有 NLM 状态、背景记忆和写入缓冲区（开始全新 session 时调用）。"""
        reset_memory_states(self.model)
        self._write_buffer = []
        self._background_state = None
        logger.debug("NLM states, background state, and write buffer reset.")

    # ------------------------------------------------------------------
    # 内部工具：将 buffer 提交到 NLM state（不生成文本）
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _rebuild_nlm_state(self) -> None:
        """将 background_state + _write_buffer forward 进 NLM（无生成）。

        用于 save()、commit() 需要将待处理 writes 反映到 NLM state 的场景。
        调用后 live NLM state = background_state ∪ buffer_writes。
        """
        reset_memory_states(self.model)
        if self._background_state is not None:
            set_nlm_states(self.model, self._background_state)
        if not self._write_buffer:
            return
        unfreeze_memory_updates(self.model)
        prefix = _SYS_PROMPT + "\n"
        for msg in self._write_buffer:
            prefix += MEMORY_WRITE_TEMPLATE.format(message=msg)
        inputs = self.tokenizer(
            prefix,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_seq_len,
        ).to(self.device)
        self.model(input_ids=inputs["input_ids"], use_cache=False)

    def commit(self) -> None:
        """将 _write_buffer 提交为持久背景记忆，然后清空 buffer。

        适用于超长 session（buffer 过大时）：先 commit()，再继续 write()。
        commit() 之后的 read() 会从已提交 state 继续积累新 writes，
        而不是从零重建整段历史，避免 buffer 无限增长。

        典型用法：
            for msg in conversation:
                oracle.write(msg)
                if len(oracle._write_buffer) > 200:
                    oracle.commit()
            oracle.save("session.pt")
        """
        n = len(self._write_buffer)
        self._rebuild_nlm_state()
        self._background_state = get_nlm_states(self.model)
        self._write_buffer = []
        logger.debug("Committed %d messages to NLM background state.", n)

    # ------------------------------------------------------------------
    # Convenience: 统计当前记忆容量
    # ------------------------------------------------------------------

    def memory_stats(self) -> dict[str, Any]:
        """返回当前 NLM 状态的统计信息（用于调试）。"""
        n_layers = 0
        n_initialized = 0
        total_params = 0
        for module in self.model.modules():
            if isinstance(module, Qwen35LayerWithMemory):
                n_layers += 1
                if module.memory_state is not None:
                    n_initialized += 1
                    for w in module.memory_state.weights:
                        total_params += w.numel()
        return {
            "nlm_layers": n_layers,
            "initialized_layers": n_initialized,
            "nlm_weight_params": total_params,
        }


# ---------------------------------------------------------------------------
# GPT-4 集成示例（standalone 函数，不依赖 openai SDK 导入）
# ---------------------------------------------------------------------------


def chat_with_memory(
    user_message: str,
    oracle: "MemoryOracle",
    llm_fn: Any,
) -> str:
    """记忆增强的对话函数。

    Args:
        user_message: 用户消息。
        oracle:       MemoryOracle 实例。
        llm_fn:       外部 LLM 调用函数，签名 (messages: list[dict]) -> str。
                      示例：lambda msgs: openai.chat.completions.create(
                                model="gpt-4", messages=msgs
                            ).choices[0].message.content

    Returns:
        LLM 生成的回复。
    """
    # 1. 召回相关记忆
    memory_context = oracle.read(user_message)

    # 2. 构造 LLM 请求
    messages = [
        {"role": "system", "content": f"背景记忆：{memory_context}"},
        {"role": "user", "content": user_message},
    ]
    response = llm_fn(messages)

    # 3. 把本轮对话写入记忆
    oracle.write(f"用户：{user_message}")
    oracle.write(f"助手：{response}")

    return response
