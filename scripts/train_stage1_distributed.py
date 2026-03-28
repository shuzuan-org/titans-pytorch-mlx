#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from titans.stage1_data import Stage1Dataset, stage1_collate_fn
from titans.stage1_models import Stage1ModelConfig, build_stage1_model

try:
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, set_seed

    HAS_ACCELERATE = True
except ImportError:  # pragma: no cover
    Accelerator = None
    DistributedDataParallelKwargs = None
    HAS_ACCELERATE = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Stage1TrainingConfig:
    data_dir: str = "data/generated/stage1_timeline_v2"
    checkpoint_dir: str = "checkpoints/stage1_timeline_v2"
    backbone: str = "qwen"
    backbone_name: str = "Qwen/Qwen2.5-7B-Instruct"
    torch_dtype: str = "auto"
    attn_implementation: str | None = None
    trust_remote_code: bool = False
    history_backbone_mode: str = "timeline"
    memory_update_source: str = "last_hidden"
    num_retrieved_memory_tokens: int = 16
    loss_mask_scope: str = "answer_only"
    memory_slots: int = 16
    memory_hidden_mult: float = 2.0
    memory_dropout: float = 0.0
    max_history_length: int = 128
    max_question_length: int = 96
    max_sequence_length: int = 128
    epochs: int = 1
    max_steps: int = -1
    batch_size: int = 1
    eval_batch_size: int = 1
    grad_accum: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_ratio: float = 0.03
    precision: str = "bf16"
    gradient_checkpointing: bool = False
    save_every: int = 200
    eval_every: int = 100
    log_every: int = 10
    num_workers: int = 0
    seed: int = 42
    resume: str | None = None
    wandb: bool = False
    wandb_project: str = "titans-stage1-timeline"
    wandb_run_name: str | None = None


class Stage1Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: Stage1TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloaders: dict[str, DataLoader],
    ) -> None:
        self.config = config

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.grad_accum,
            mixed_precision=config.precision if config.precision != "none" else "no",
            kwargs_handlers=[ddp_kwargs],
            log_with="wandb" if config.wandb else None,
        )

        self.model = model
        optimizer_kwargs: dict[str, Any] = {
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "betas": (0.9, 0.95),
        }
        if torch.cuda.is_available():
            optimizer_kwargs["fused"] = True
        self.optimizer = torch.optim.AdamW(
            self.model.get_trainable_parameters(),
            **optimizer_kwargs,
        )

        if config.max_steps > 0:
            self.total_steps = config.max_steps
        else:
            updates_per_epoch = max(1, math.ceil(len(train_dataloader) / config.grad_accum))
            self.total_steps = updates_per_epoch * config.epochs

        warmup_steps = int(self.total_steps * config.warmup_ratio)
        self.scheduler = self._get_cosine_schedule(warmup_steps, self.total_steps)

        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            train_dataloader,
            self.scheduler,
        )
        self.eval_dataloaders = {
            name: self.accelerator.prepare(dataloader)
            for name, dataloader in eval_dataloaders.items()
        }

        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")
        self.checkpoint_dir = Path(config.checkpoint_dir)
        if self.accelerator.is_main_process:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if config.wandb and self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=config.wandb_project,
                config=vars(config),
                init_kwargs={"wandb": {"name": config.wandb_run_name}},
            )

    def _get_cosine_schedule(
        self,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr_ratio: float = 0.1,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        self.model.train()
        with self.accelerator.accumulate(self.model):
            outputs = self.model(**batch)
            loss = outputs["loss"]
            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients and self.config.grad_clip > 0:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        metrics = {"loss": float(loss.detach().item())}
        if "answer_loss" in outputs:
            metrics["answer_loss"] = float(outputs["answer_loss"].item())
        if "write_gate_loss" in outputs:
            metrics["write_gate_loss"] = float(outputs["write_gate_loss"].item())
        return metrics

    @torch.no_grad()
    def evaluate_one(self, name: str, dataloader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        for batch in tqdm(
            dataloader,
            desc=f"Eval {name}",
            leave=False,
            disable=not self.accelerator.is_main_process,
        ):
            outputs = self.model(**batch)
            total_loss += float(outputs["loss"].detach().item())
            total_batches += 1

        gathered_loss = self.accelerator.gather(
            torch.tensor([total_loss], device=self.accelerator.device)
        ).sum().item()
        gathered_batches = self.accelerator.gather(
            torch.tensor([total_batches], device=self.accelerator.device)
        ).sum().item()
        avg_loss = gathered_loss / max(1, gathered_batches)
        return {f"eval/{name}_loss": avg_loss}

    def evaluate(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for name, dataloader in self.eval_dataloaders.items():
            metrics.update(self.evaluate_one(name, dataloader))
        self.model.train()
        return metrics

    def save_checkpoint(self, name: str) -> None:
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            return

        path = self.checkpoint_dir / name
        path.mkdir(parents=True, exist_ok=True)
        self.accelerator.save_state(str(path))

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        total_params, trainable_params = unwrapped_model.count_parameters()
        trainable_param_names = {
            name for name, parameter in unwrapped_model.named_parameters() if parameter.requires_grad
        }
        state_dict = unwrapped_model.state_dict()
        checkpoint_payload = {
            "config": vars(self.config),
            "model_config": vars(unwrapped_model.config),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in state_dict.items()
                if name in trainable_param_names
            },
        }
        torch.save(checkpoint_payload, path / "stage1_state.pt")
        with (path / "trainer_state.json").open("w", encoding="utf-8") as f:
            json.dump(
                {"global_step": self.global_step, "epoch": self.epoch},
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info("Saved checkpoint to %s", path)

    def load_checkpoint(self, path: Path) -> None:
        self.accelerator.load_state(str(path))
        trainer_state_path = path / "trainer_state.json"
        if trainer_state_path.exists():
            trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
            self.global_step = int(trainer_state.get("global_step", 0))
            self.epoch = int(trainer_state.get("epoch", 0))
        logger.info("Loaded checkpoint from %s", path)

    def train(self) -> None:
        start_time = time.time()
        running_metrics: dict[str, float] = {}
        running_count = 0
        pbar = tqdm(
            total=self.total_steps,
            initial=self.global_step,
            desc="Training",
            disable=not self.accelerator.is_main_process,
        )

        while self.global_step < self.total_steps:
            self.epoch += 1
            for batch in self.train_dataloader:
                if self.global_step >= self.total_steps:
                    break

                metrics = self.train_step(batch)
                if not self.accelerator.sync_gradients:
                    continue

                self.global_step += 1
                pbar.update(1)
                running_count += 1
                for key, value in metrics.items():
                    running_metrics[key] = running_metrics.get(key, 0.0) + value

                if self.global_step % self.config.log_every == 0:
                    averaged = {
                        key: value / max(1, running_count)
                        for key, value in running_metrics.items()
                    }
                    current_lr = self.scheduler.get_last_lr()[0]
                    if self.accelerator.is_main_process:
                        postfix = {key: f"{value:.4f}" for key, value in averaged.items()}
                        postfix["lr"] = f"{current_lr:.2e}"
                        pbar.set_postfix(postfix)
                        self.accelerator.log(
                            {f"train/{key}": value for key, value in averaged.items()} | {"train/lr": current_lr},
                            step=self.global_step,
                        )
                    running_metrics = {}
                    running_count = 0

                if self.config.eval_every > 0 and self.global_step % self.config.eval_every == 0:
                    eval_metrics = self.evaluate()
                    if self.accelerator.is_main_process:
                        formatted_metrics = ", ".join(
                            f"{key}={value:.4f}" for key, value in sorted(eval_metrics.items())
                        )
                        logger.info(
                            "Eval step %s: %s",
                            self.global_step,
                            formatted_metrics,
                        )
                        self.accelerator.log(eval_metrics, step=self.global_step)
                    mean_eval_loss = sum(eval_metrics.values()) / max(1, len(eval_metrics))
                    if mean_eval_loss < self.best_eval_loss:
                        self.best_eval_loss = mean_eval_loss
                        self.save_checkpoint("best_model")

                if self.config.save_every > 0 and self.global_step % self.config.save_every == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

        self.save_checkpoint("final_model")
        elapsed = time.time() - start_time
        logger.info("Training completed in %.2f minutes", elapsed / 60)
        self.accelerator.end_training()


def parse_args() -> Stage1TrainingConfig:
    parser = argparse.ArgumentParser(description="Train stage1 timeline memory with accelerate")
    parser.add_argument("--data-dir", default="data/generated/stage1_timeline_v2")
    parser.add_argument("--checkpoint-dir", default="checkpoints/stage1_timeline_v2")
    parser.add_argument("--backbone", default="qwen", choices=["qwen"])
    parser.add_argument("--backbone-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--history-backbone-mode", default="timeline")
    parser.add_argument("--memory-update-source", default="last_hidden")
    parser.add_argument("--num-retrieved-memory-tokens", type=int, default=16)
    parser.add_argument("--loss-mask-scope", default="answer_only")
    parser.add_argument("--memory-slots", type=int, default=16)
    parser.add_argument("--memory-hidden-mult", type=float, default=2.0)
    parser.add_argument("--memory-dropout", type=float, default=0.0)
    parser.add_argument("--max-history-length", type=int, default=128)
    parser.add_argument("--max-question-length", type=int, default=96)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--precision", default="bf16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="titans-stage1-timeline")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()
    return Stage1TrainingConfig(**vars(args))


def build_dataloaders(
    config: Stage1TrainingConfig,
    tokenizer: Any,
) -> tuple[DataLoader, dict[str, DataLoader]]:
    collate = partial(
        stage1_collate_fn,
        tokenizer=tokenizer,
        max_history_length=config.max_history_length,
        max_question_length=config.max_question_length,
        max_sequence_length=config.max_sequence_length,
        loss_mask_scope=config.loss_mask_scope,
    )
    train_dataset = Stage1Dataset(config.data_dir, split="train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    eval_dataloaders = {
        "timeline": DataLoader(
            Stage1Dataset(config.data_dir, split="eval"),
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=collate,
        )
    }
    return train_dataloader, eval_dataloaders


def main() -> None:
    if not HAS_ACCELERATE:
        raise ImportError("accelerate is required. Install with: pip install accelerate")

    config = parse_args()
    set_seed(config.seed)

    model_config = Stage1ModelConfig(
        backbone_name=config.backbone_name,
        torch_dtype=config.torch_dtype,
        attn_implementation=config.attn_implementation,
        history_backbone_mode=config.history_backbone_mode,
        memory_update_source=config.memory_update_source,
        num_retrieved_memory_tokens=config.num_retrieved_memory_tokens,
        loss_mask_scope=config.loss_mask_scope,
        memory_slots=config.memory_slots,
        memory_hidden_mult=config.memory_hidden_mult,
        memory_dropout=config.memory_dropout,
        trust_remote_code=config.trust_remote_code,
    )
    model = build_stage1_model(model_config)
    total_params, trainable_params = model.count_parameters()
    logger.info("Total parameters: %s", f"{total_params:,}")
    logger.info("Trainable parameters: %s", f"{trainable_params:,}")

    train_dataloader, eval_dataloaders = build_dataloaders(
        config=config,
        tokenizer=model.backbone.tokenizer,
    )
    trainer = Stage1Trainer(model, config, train_dataloader, eval_dataloaders)

    if config.resume:
        trainer.load_checkpoint(Path(config.resume))
    trainer.train()


if __name__ == "__main__":
    main()


