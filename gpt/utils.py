"""
Utility functions for GPT-1 project.

Features (all via wandb):
    - Experiment tracking with full hyperparameter logging
    - Training / validation / eval metric logging
    - GPU memory monitoring (peak allocation)
    - Throughput tracking (tokens/sec)
"""

import math
import sys
import time
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any

import torch
import wandb
import yaml


# ====================================================================
# Config dataclass — single source of truth for all hyperparameters
# ====================================================================
@dataclass
class ExperimentConfig:
    """All hyperparameters from the GPT-1 paper, grouped by stage.

    Defaults match the paper exactly. Override via YAML or CLI::

        # YAML only
        cfg = load_config("configs/pretrain.yaml")

        # YAML + CLI overrides  (e.g. --lr 1e-4 --batch_size 32)
        cfg = load_config("configs/pretrain.yaml", cli_args=sys.argv[1:])
    """

    # ── Model architecture (paper Table 1) ──────────────────────────
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072                    # 4 × d_model
    vocab_size: int = 40_000
    max_seq_len: int = 512
    dropout: float = 0.1

    # ── Stage 1: unsupervised pre-training ──────────────────────────
    batch_size: int = 64
    micro_batch_size: int = 8           # actual GPU batch; accumulate → effective batch_size
    lr: float = 2.5e-4                  # peak learning rate
    warmup_steps: int = 2_000           # linear warmup from 0 → lr
    total_steps: int = 800_000          # ~100 epochs on BooksCorpus
    weight_decay: float = 0.01          # L2 on all params except biases & LayerNorm
    grad_clip: float = 1.0              # global gradient-norm clipping
    lr_schedule: str = "cosine"         # "cosine" | "linear"

    # ── Adam optimizer ──────────────────────────────────────────────
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    # ── Stage 2: supervised fine-tuning ─────────────────────────────
    task_name: str = ""
    ft_lr: float = 6.25e-5
    ft_epochs: int = 3
    ft_batch_size: int = 32
    ft_warmup_frac: float = 0.002       # warmup over 0.2% of training steps
    ft_lr_schedule: str = "linear"      # "linear" | "cosine"
    lambda_aux: float = 0.5             # L₃ = L₂ + λ·L₁
    label_smoothing: float = 0.0        # paper doesn't specify; default off

    # ── Precision & performance ─────────────────────────────────────
    use_amp: bool = True                # mixed-precision (fp16)
    grad_checkpoint: bool = False       # gradient checkpointing
    compile: bool = False               # use torch.compile

    # ── Infrastructure ──────────────────────────────────────────────
    wandb_project: str = "GPT1"
    checkpoint_dir: str = "checkpoints/"
    data_dir: str = "data/"
    resume_from: str = ""               # path to checkpoint (empty = fresh start)

    # ── Logging intervals ───────────────────────────────────────────
    log_interval: int = 50              # log metrics every N steps
    eval_interval: int = 1_000          # run validation every N steps
    save_interval: int = 5_000          # save checkpoint every N steps

    # ── Data pipeline ───────────────────────────────────────────────────
    dataset_name: str = "lucadiliello/bookcorpusopen"
    val_split: float = 0.005
    num_workers: int = 4

    # ── Logging intervals ───────────────────────────────────────────
    log_interval: int = 50              # log metrics every N steps
    eval_interval: int = 1_000          # run validation every N steps
    save_interval: int = 5_000          # save checkpoint every N steps
    generate_interval: int = 1_000      # generate text samples every N steps


    # ── Derived (computed, not set in YAML) ─────────────────────────
    @property
    def grad_accum_steps(self) -> int:
        """Number of micro-batches to accumulate for effective batch_size."""
        assert self.batch_size % self.micro_batch_size == 0, (
            f"batch_size ({self.batch_size}) must be divisible by "
            f"micro_batch_size ({self.micro_batch_size})"
        )
        return self.batch_size // self.micro_batch_size


# ====================================================================
# Config loading — YAML files + CLI overrides
# ====================================================================
def load_config(
    yaml_path: str | Path | None = None,
    cli_args: list[str] | None = None,
    **overrides: Any,
) -> ExperimentConfig:
    """Build an ExperimentConfig with layered overrides.

    Priority (highest wins):  CLI flags  >  **overrides  >  YAML  >  defaults

    Args:
        yaml_path: Path to a YAML config file. Fields in the file override
            dataclass defaults.
        cli_args: Raw CLI args (e.g. ``sys.argv[1:]``). Supports
            ``--field_name value`` syntax for any ExperimentConfig field.
        **overrides: Direct keyword overrides (handy in notebooks).

    Returns:
        A fully-populated ExperimentConfig.

    Examples::

        # From code / notebook
        cfg = load_config("configs/pretrain.yaml", lr=1e-4)

        # From a training script's CLI
        cfg = load_config(sys.argv[1], cli_args=sys.argv[2:])
    """
    valid_fields = {f.name: f for f in fields(ExperimentConfig)}
    merged: dict[str, Any] = {}

    # Layer 1: YAML file
    if yaml_path is not None:
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            yaml_data = yaml.safe_load(f) or {}
        unknown = set(yaml_data) - set(valid_fields)
        if unknown:
            raise ValueError(
                f"Unknown config fields in {path}: {unknown}. "
                f"Valid fields: {sorted(valid_fields)}"
            )
        merged.update(yaml_data)

    # Layer 2: keyword overrides
    unknown = set(overrides) - set(valid_fields)
    if unknown:
        raise ValueError(f"Unknown config overrides: {unknown}")
    merged.update(overrides)

    # Layer 3: CLI args  (--field value)
    if cli_args:
        merged.update(_parse_cli_overrides(cli_args, valid_fields))

    # Cast values to the correct types (YAML may give int where we need float, etc.)
    for key, value in merged.items():
        expected_type = valid_fields[key].type
        merged[key] = _cast(value, expected_type)

    return ExperimentConfig(**merged)


def _parse_cli_overrides(
    args: list[str], valid_fields: dict,
) -> dict[str, Any]:
    """Parse ``--field_name value`` pairs from a CLI arg list."""
    overrides: dict[str, Any] = {}
    i = 0
    while i < len(args):
        token = args[i]
        if token.startswith("--"):
            key = token.lstrip("-")
            if key not in valid_fields:
                raise ValueError(
                    f"Unknown CLI flag: {token}. "
                    f"Valid flags: {sorted(valid_fields)}"
                )
            if i + 1 >= len(args):
                raise ValueError(f"CLI flag {token} requires a value")
            overrides[key] = args[i + 1]
            i += 2
        else:
            i += 1  # skip positional args (e.g. config path)
    return overrides


def _cast(value: Any, type_hint: str) -> Any:
    """Cast a raw value (from YAML or CLI) to the expected Python type."""
    # type_hint is a string like 'int', 'float', 'str', 'bool'
    if type_hint == "bool" or type_hint is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)
    if type_hint == "int" or type_hint is int:
        return int(value)
    if type_hint == "float" or type_hint is float:
        return float(value)
    return str(value)


# ====================================================================
# GPU helpers
# ====================================================================
def get_gpu_info() -> dict[str, Any]:
    """Static GPU metadata — call once at the start of a run."""
    if not torch.cuda.is_available():
        return {"gpu": "none"}
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_name": props.name,
        "gpu_count": torch.cuda.device_count(),
        "gpu_total_memory_MB": round(props.total_memory / 1024**2, 1),
    }


def _gpu_metrics() -> dict[str, float]:
    """Live GPU memory stats — called every logging step.

    Uses max_memory_allocated() which tracks the *peak* since the last
    reset, so it is never 0 once any CUDA work has happened. This is
    the same approach used in the Attention_pytorch reference trainer.
    """
    if not torch.cuda.is_available():
        return {}
    total = torch.cuda.get_device_properties(0).total_memory
    return {
        "gpu/memory_allocated_GB": round(
            torch.cuda.memory_allocated() / 1024**3, 3
        ),
        "gpu/peak_memory_GB": round(
            torch.cuda.max_memory_allocated() / 1024**3, 3
        ),
        "gpu/memory_reserved_GB": round(
            torch.cuda.memory_reserved() / 1024**3, 3
        ),
        "gpu/utilization_pct": round(
            torch.cuda.memory_allocated() / total * 100, 2
        ),
    }


# ====================================================================
# Experiment tracker — wandb wrapper
# ====================================================================
class ExperimentTracker:
    """Thin, opinionated wrapper around wandb for GPT-1 training.

    Usage::

        with ExperimentTracker(config, "pretrain-v1") as tracker:
            for step in range(total_steps):
                t0 = time.time()
                loss, grad_norm = train_step(...)
                tracker.log_train_step(
                    step=step, loss=loss, lr=scheduler.get_lr(),
                    grad_norm=grad_norm,
                    tokens_processed=batch_size * seq_len,
                    step_time=time.time() - t0,
                )
    """

    def __init__(
        self,
        config: ExperimentConfig,
        run_name: str,
        tags: list[str] | None = None,
        notes: str = "",
    ):
        self.config = config
        self.run = wandb.init(
            project=config.wandb_project,
            name=run_name,
            config={**asdict(config), **get_gpu_info()},
            tags=tags or [],
            notes=notes,
            save_code=True,
        )

        # Tell wandb that "gpu/*" metrics use the training step as x-axis
        # This ensures they appear as proper charts rather than being ignored
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("gpu/*", step_metric="train/step")
        wandb.define_metric("throughput/*", step_metric="train/step")
        wandb.define_metric("val/*", step_metric="train/step")

    # ── context manager ─────────────────────────────────────────────
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.finish()
        return False

    # ── training ────────────────────────────────────────────────────
    def log_train_step(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: float | None = None,
        tokens_processed: int | None = None,
        step_time: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log one training step — everything goes into a single
        wandb.log() call so all metrics share the same step."""
        metrics: dict[str, Any] = {
            "train/loss": loss,
            "train/lr": lr,
            "train/step": step,
        }

        if grad_norm is not None:
            metrics["train/grad_norm"] = grad_norm

        # throughput
        if tokens_processed is not None and step_time is not None and step_time > 0:
            metrics["throughput/tokens_per_sec"] = round(tokens_processed / step_time, 1)
            metrics["throughput/step_time_sec"] = round(step_time, 4)

        # GPU memory — single combined dict, one wandb.log call
        metrics.update(_gpu_metrics())

        if extra:
            metrics.update(extra)

        wandb.log(metrics, step=step)

    # ── validation ──────────────────────────────────────────────────
    def log_validation(
        self,
        step: int,
        val_loss: float,
        perplexity: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log validation metrics. Perplexity auto-computed when omitted."""
        metrics: dict[str, Any] = {
            "val/loss": val_loss,
            "val/perplexity": perplexity if perplexity is not None else math.exp(val_loss),
        }
        if extra:
            metrics.update(extra)
        wandb.log(metrics, step=step)

    # ── evaluation benchmarks ───────────────────────────────────────
    def log_eval_results(
        self,
        results: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log eval benchmark scores (e.g. GLUE tasks)."""
        prefixed = {f"eval/{k}": v for k, v in results.items()}
        if step is not None:
            wandb.log(prefixed, step=step)
        else:
            for k, v in prefixed.items():
                wandb.run.summary[k] = v

    def log_eval_table(self, results: dict[str, dict[str, float]]) -> None:
        """Log a my-result-vs-paper comparison table."""
        table = wandb.Table(columns=["Task", "My Result", "Paper Result", "Delta"])
        for task, scores in results.items():
            delta = scores["my"] - scores["paper"]
            table.add_data(task, scores["my"], scores["paper"], round(delta, 2))
        wandb.log({"eval/comparison_table": table})

    # ── lifecycle ───────────────────────────────────────────────────
    def finish(self) -> None:
        """Clean shutdown — always call this (or use the context manager)."""
        wandb.finish()


# ====================================================================
# Misc utilities
# ====================================================================
def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Trainable / non-trainable parameter breakdown."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }