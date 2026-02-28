"""
Smoke-test for wandb integration.

Simulates a short training run with realistic tensor sizes to verify
that ALL metric categories log — especially GPU memory.
"""

import time
import math
import torch
from gpt.utils import ExperimentConfig, ExperimentTracker

# ── setup ───────────────────────────────────────────────────────────
config = ExperimentConfig()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Realistic sizes matching the config
B, T, D = config.micro_batch_size, config.max_seq_len, config.d_model
tokens_per_step = B * T

with ExperimentTracker(
    config=config,
    run_name="test-wandb-gpu-metrics",
    tags=["test", "smoke"],
    notes="Verify GPU metrics log correctly",
) as tracker:

    # Allocate tensors on GPU so memory counters are non-zero
    hidden = torch.randn(B, T, D, device=device)
    logits = torch.randn(B, T, config.vocab_size, device=device)

    print(f"\n{'='*60}")
    print(f"  Device : {device}")
    if device == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  Alloc  : {torch.cuda.memory_allocated()/1024**3:.3f} GB")
        print(f"  Peak   : {torch.cuda.max_memory_allocated()/1024**3:.3f} GB")
    print(f"{'='*60}\n")

    # ── training steps ──────────────────────────────────────────────
    for step in range(10):
        t0 = time.time()

        # Fake forward pass  — forces real GPU work
        _ = hidden @ hidden.transpose(-1, -2)
        if device == "cuda":
            torch.cuda.synchronize()

        step_time = time.time() - t0

        tracker.log_train_step(
            step=step,
            loss=8.0 - step * 0.5,
            lr=1e-4,
            grad_norm=2.0 / (step + 1),
            tokens_processed=tokens_per_step,
            step_time=step_time,
        )

        if device == "cuda":
            print(
                f"  step {step:>2d}  |  loss={8.0 - step*0.5:.2f}  |  "
                f"tok/s={tokens_per_step/step_time:,.0f}  |  "
                f"gpu_alloc={torch.cuda.memory_allocated()/1024**3:.3f} GB  |  "
                f"gpu_peak={torch.cuda.max_memory_allocated()/1024**3:.3f} GB"
            )
        else:
            print(f"  step {step:>2d}  |  loss={8.0 - step*0.5:.2f}  |  (cpu)")

    # ── validation ──────────────────────────────────────────────────
    val_loss = 3.2
    tracker.log_validation(step=10, val_loss=val_loss)
    print(f"\n  val_loss={val_loss:.2f}  perplexity={math.exp(val_loss):.2f}")

    # ── eval results ────────────────────────────────────────────────
    tracker.log_eval_results({"sst2_acc": 0.913, "cola_mcc": 0.454}, step=10)


    tracker.log_eval_table({
        "SST-2": {"my": 0.913, "paper": 0.913},
        "CoLA":  {"my": 0.454, "paper": 0.454},
    })
    print("  eval results + table logged")

print("\n✓ All metrics logged — check https://wandb.ai/anveshdange/GPT1")

if device == "cuda":
    del hidden, logits
    torch.cuda.empty_cache()
