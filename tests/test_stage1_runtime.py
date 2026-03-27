from __future__ import annotations

import tempfile
from types import SimpleNamespace

import torch

from titans.stage1_models import (
    DenseTimelineMemory,
    FrozenBackboneWithTimelineMemory,
    Stage1ModelConfig,
)
from titans.stage1_runtime import Stage1DeploymentRuntime


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def __call__(self, texts: list[str], padding: bool, truncation: bool, return_tensors: str) -> dict[str, torch.Tensor]:
        del truncation, return_tensors
        encoded = [[(ord(ch) % 17) + 1 for ch in text] or [1] for text in texts]
        max_len = max(len(row) for row in encoded)
        padded = [row + [self.pad_token_id] * (max_len - len(row)) for row in encoded]
        masks = [[1 if token != self.pad_token_id else 0 for token in row] for row in padded]
        if padding:
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        raise AssertionError("padding=True expected")

    def decode(self, ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "|".join(str(i) for i in ids)


class FakeBackboneAdapter(torch.nn.Module):
    def __init__(self, config: Stage1ModelConfig) -> None:
        super().__init__()
        self._hidden_size = 6
        self.model = torch.nn.Embedding(128, self._hidden_size)
        self.tokenizer = FakeTokenizer()
        self.freeze_parameters()
        self.config = config

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def freeze_parameters(self) -> None:
        self.model.requires_grad_(False)
        self.model.eval()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        base = input_ids.to(dtype=torch.float32).unsqueeze(-1)
        offsets = torch.arange(self.hidden_size, device=input_ids.device, dtype=torch.float32)
        return base + offsets.view(1, 1, -1)

    def encode_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeds = self.embed_input_ids(input_ids)
        return embeds * attention_mask.unsqueeze(-1).to(embeds.dtype)

    def forward_with_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        del attention_mask, labels
        batch, seq, _ = inputs_embeds.shape
        logits = torch.zeros(batch, seq, 128, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        return SimpleNamespace(loss=torch.tensor(0.0, device=inputs_embeds.device), logits=logits)

    def generate_from_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config: dict[str, int | float | bool] | None = None,
    ) -> torch.Tensor:
        del attention_mask, generation_config
        summary = inputs_embeds[:, 0, :].sum(dim=-1).round().to(dtype=torch.long)
        return torch.stack([summary, summary + 1, summary + 2], dim=1)


def build_fake_model() -> FrozenBackboneWithTimelineMemory:
    config = Stage1ModelConfig(backbone_name="fake", memory_slots=4)
    original_backbone = FrozenBackboneWithTimelineMemory.__init__.__globals__["FrozenBackboneAdapter"]
    try:
        FrozenBackboneWithTimelineMemory.__init__.__globals__["FrozenBackboneAdapter"] = FakeBackboneAdapter
        model = FrozenBackboneWithTimelineMemory(config)
    finally:
        FrozenBackboneWithTimelineMemory.__init__.__globals__["FrozenBackboneAdapter"] = original_backbone
    model.memory = DenseTimelineMemory(hidden_size=model.backbone.hidden_size, memory_slots=config.memory_slots)
    model.eval()
    return model


def test_write_changes_state_and_increments_version() -> None:
    model = build_fake_model()
    runtime = Stage1DeploymentRuntime(model)

    initial = model.init_session_state("s1")
    initial_memory = initial.memory_state.clone()
    result = runtime.write_memory("s1", contents=["张三住在上海", "李四住在杭州"])

    updated = runtime.store.get("s1")
    assert updated is not None
    assert result.memory_version == 2
    assert result.num_items_written == 2
    assert updated.stats.num_writes == 2
    assert not torch.allclose(initial_memory, updated.memory_state)


def test_chat_does_not_modify_memory_state() -> None:
    model = build_fake_model()
    runtime = Stage1DeploymentRuntime(model)
    runtime.write_memory("s1", content="张三住在上海")

    before = model.clone_session_state(runtime.store.get("s1"))
    chat_result = runtime.chat_with_memory("s1", "张三住在哪？", include_debug=True)
    after = runtime.store.get("s1")

    assert after is not None
    assert chat_result.memory_version == before.memory_version
    assert torch.allclose(before.memory_state, after.memory_state)
    assert after.stats.num_queries == before.stats.num_queries + 1
    assert chat_result.retrieval_weights is not None


def test_snapshot_roundtrip_preserves_retrieval() -> None:
    model = build_fake_model()
    runtime = Stage1DeploymentRuntime(model)
    runtime.write_memory("s1", contents=["alpha", "beta"])
    state = runtime.store.get("s1")
    assert state is not None

    retrieved_before, weights_before, _, _ = model.retrieve_from_state(state, "alpha?")

    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        runtime.save_session_snapshot("s1", tmp.name)
        runtime.delete_session("s1")
        restored = runtime.load_session_snapshot(tmp.name)

    retrieved_after, weights_after, _, _ = model.retrieve_from_state(restored, "alpha?")
    assert torch.allclose(retrieved_before, retrieved_after)
    assert torch.allclose(weights_before, weights_after)


