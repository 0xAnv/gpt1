"""
My brainstorming steps to train the model 

For each step in training:
  1. Grab a micro-batch of 512-token sequences
  2. Shift them to create (input, target) pairs for next-token prediction
  3. Forward pass → get logits
  4. Compute cross-entropy loss
  5. Backward pass → accumulate gradients (over multiple micro-batches)
  6. Once we've accumulated enough → clip gradients, optimizer step, scheduler step
  7. Log metrics to wandb
  8. Periodically: validate, save checkpoint, generate sample text

"""

import logging 
import math 
import sys 
import time
import wandb
from pathlib import Path 

# torch specifics 
import torch 
import torch.nn as nn 
from torch.cuda.amp import GradScaler, autocast # fp16 mixed precision

from gpt.model import GPT1 
from gpt.data import get_pretrain_dataloaders
from gpt.utils import (
    ExperimentConfig, 
    ExperimentTracker, 
    load_config, 
    count_parameters
)
from gpt.tokenizer import Tokenizer

# setting up logger 
logger = logging.getLogger(__name__) 

def get_lr_scheduler(
    optimiser: torch.optim.Optimizer, 
    warmup_steps:int, 
    total_steps:int, 
    schedule:str = "cosine" # linear also there as option
) -> torch.optim.lr_scheduler.LambdaLR: 
    """
    Creates the learning rate schedular from GPT 1 paper. 

    Two phases: 
        1. Linear Warmup: 0 -> peak_lr over `warmup_steps`
        2. Cosine Decay: peak_lr -> 0 over remaining steps 

    We use LambdaLR which takes a function that returns a MULTIPLIER 
    (not the actual LR). So we return values between 0.0 and 1.0 
    """

    # intermediate function to get lr_lambda 
    def lr_lambda(current_step:int) -> float : 
        # phase 1: Linear Warmup 
        if current_step < warmup_steps: 
            return current_step / max(1, warmup_steps)

        # Phase 2: Decay 
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)

        if schedule == "cosine":
            # Cosine annealing: smooth decays from 1.0 to 0.0 
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        else:
            # Linear decay: straight lines from 1.0 to 0.0 
            return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)


def create_optimizer(
    model:GPT1, 
    lr:float, 
    weight_decay: float, 
    betas: tuple[float, float], 
    eps:float
)  -> torch.optim.AdamW:

    """
    Creates AdamW optimiser with proper weight decay grouping. 
    
    The paper says "l2 regularization on all params execept biases and LayerNOrm"
    we achive this by splitting parameters into two groups: 
        - Group 1 : weights of Linear/Embedding layers -> get weight decay 
        - Group 2 : Biases + LayerNorm -> no weight decay
    """

    # seprate parameters into decay and no-decay groups 
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No decay for: biases, LayerNorm weighs (gamma), LayerNorm biases (beta)
        if param.ndim == 1:  # biases and LayerNorm params are all 1-dim
            no_decay_params.append(param)
        else: 
            decay_params.append(param)

    
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay}, 
        {"params": no_decay_params, "weight_decay": 0.0}
    ]

    optimizer = torch.optim.AdamW(
        param_groups, 
        lr = lr, 
        betas=betas, 
        eps=eps,
        fused=True if torch.cuda.is_available() else False
    )

    return optimizer

# validation function 
@torch.no_grad()
def validate(
    model:GPT1, 
    val_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    use_amp: bool = True, 
) -> dict[str, float]:
    """
    Run validation and return loss + perplexity.

    @torch.no_grad() disables gradient computation entirely. 
    This saved memory (no grad graph stored) and is faster. 
    """
    model.eval() # switch off dropouts etc.

    total_loss = 0.0 
    total_batches = 0 

    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)

        # same input/target split as training
        inputs = input_ids[:, :-1] # all tokens except last 
        targets = input_ids[:, 1:] # all tokens except first

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(inputs)        
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), 
                targets.reshape(-1),
            )

        total_loss += loss.item()
        total_batches += 1 

    model.train() # switch back to train mode 

    avg_loss = total_loss / max(1, total_batches)
    perplexity = math.exp(avg_loss)

    return {"val_loss": avg_loss, "perplexity": perplexity}


# checkpoint save and load functions 
def save_checkpoint(
    model:GPT1, 
    optimiser: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler.LRScheduler, 
    scaler: GradScaler, 
    step:int, 
    config: ExperimentConfig, 
    path:Path
) -> None:
    """ Save everything needed to resume training """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step, 
            "model_state_dict": model.state_dict(), 
            "optimizer_state_dict": optimiser.state_dict(), 
            "scheduler_state_dict": scheduler.state_dict(), 
            "scaler_state_dict": scaler.state_dict(), 
            "config": config
        },
        path,
    )

    logger.info(f"Checkpoint saved: {path} (step {step})")


def load_checkpoint(
    path: Path, 
    model: GPT1, 
    optimiser: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler.LRScheduler, 
    scaler: GradScaler
) -> int : 
    """Load checkpoint and return the step number to resume from"""
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    step = checkpoint['step']
    logger.info(f"Resumed from checkpoint: {path} (step {step})")
    return step

# helper function that uses tokenizer and model generate() method to produce readable strings 
@torch.no_grad()
def generate_samples(
    model: GPT1, 
    tokenizer: Tokenizer, 
    device: torch.device, 
    prompt:str = "The quick brown fox", 
    num_samples:int = 2, 
    max_new_tokens:int = 50, 
    temperature:float = 0.8, 
    top_k:int = 40
) -> list[str]:
    """ Generates few text samples for quality monitoring during training """ 
    model.eval()

    # encode prompt 
    input_ids = tokenizer.encode(prompt).ids

    # convert to tensor and batch it (repeat for num_samples)
    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0).repeat(num_samples, 1)

    # Generate 
    generated_ids = model.generate(
        x, 
        max_new_tokens=max_new_tokens, 
        temperature=temperature, 
        top_k=top_k
    )

    # Decode 
    samples = []
    for i in range(num_samples):
        # convert tensor to list of ints and decode 
        decoded = tokenizer.decode(generated_ids[i].tolist())
        samples.append(decoded)

    model.train()
    return samples

# main training function 
def train(config: ExperimentConfig) -> None:
    """
    Full Pre training loop for GPT 1

    This function orchestrates everything:
        1. Setup (device, model, optimizer, data)
        2. Training loop with grad accum 
        3. Periodic validation, checkpointing, logging
    """

    # setup 
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s | %(name)s | %(message)s", 
        datefmt="%H:%M:%S"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # allow tf32 for tensor core optimization on ampere+ gpus 
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True # Auto-tuner for cudnn algorithms

    tokenizer = Tokenizer.from_file(str(Path(config.data_dir) / "pretrain" / "tokenizer.json"))

    # Data 
    data_path = Path(config.data_dir) / "pretrain" / "tokens.bin"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Pre-Tokenised data not found at {data_path} "
            f"Run prepare_pretrain_data() first"
        )
    
    train_loader, val_loader = get_pretrain_dataloaders(
        data_path = data_path, 
        seq_len=config.max_seq_len,
        batch_size=config.micro_batch_size, #dataloader uses microbatch 
        val_split=config.val_split, 
        num_workers=config.num_workers, 
    )

    # Model 
    model = GPT1(
        vocab_size=config.vocab_size, 
        max_seq_len=config.max_seq_len, 
        n_layers=config.n_layers, 
        d_model=config.d_model, 
        n_heads=config.n_heads, 
        d_ff=config.d_ff, 
        dropout=config.dropout,
        use_checkpoint=config.grad_checkpoint,  # trade compute for VRAM savings
    ).to(device)

    param_info = count_parameters(model)
    logger.info(
        f"Model created: {param_info['trainable']:,} trainable params, {param_info['total']:,} total params"
    )

    if config.compile:
        logger.info("Compiling model using torch.compile...")
        # Since we are using fused kernels, compile mode "default" or "reduce-overhead" is great
        model = torch.compile(model)

    # optimiser 
    optimiser = create_optimizer(
        model=model, 
        lr=config.lr, 
        weight_decay=config.weight_decay, 
        betas=(config.adam_beta1, config.adam_beta2), 
        eps=config.adam_eps
    )

    scheduler = get_lr_scheduler(
        optimiser=optimiser, 
        warmup_steps=config.warmup_steps, 
        total_steps=config.total_steps, 
        schedule=config.lr_schedule
    )

    # Mixed precision scaler 
    scaler = GradScaler(enabled=config.use_amp)

    # Auto-resume logic
    if not config.resume_from:
        # Check if any existing checkpoints exist in the directory
        ckpt_dir = Path(config.checkpoint_dir)
        if ckpt_dir.exists():
            ckpts = list(ckpt_dir.glob("step_*.pt"))
            if ckpts:
                # Find highest step
                latest_ckpt = max(ckpts, key=lambda p: int(p.stem.split('_')[1]))
                logger.info(f"Auto-detected latest checkpoint: {latest_ckpt}")
                config.resume_from = str(latest_ckpt)

    # Resume from checkpoint 
    start_step = 0 
    resume_run_id = None
    if config.resume_from: 
        start_step = load_checkpoint(
            path=Path(config.resume_from), 
            model=model, 
            optimiser=optimiser, 
            scheduler=scheduler,
            scaler=scaler
        )
        # Check if there's an associated wandb state to resume 
        wandb_id_path = Path(config.checkpoint_dir) / ".wandb_id"
        if wandb_id_path.exists():
            with open(wandb_id_path, "r") as f:
                resume_run_id = f.read().strip()
            logger.info(f"Found existing W&B run ID: {resume_run_id} to append tracking")

    # Training Loop 
    with ExperimentTracker(config, run_name="pretrain-V1", run_id=resume_run_id) as tracker:
        model.train()

        global_step = start_step
        grad_accum_steps = config.grad_accum_steps #eg. 64//8=8

        # we need an infinite data iterator (training runs for total steps)
        # not total epochs. When the DataLoader is exhausted, restart it. 
        train_iter = iter(train_loader)

        from tqdm import tqdm
        pbar = tqdm(
            total=config.total_steps,
            initial=global_step,
            desc="Pre-training",
            dynamic_ncols=True,
            smoothing=0.01
        )

        try:
            while global_step < config.total_steps:
                step_loss = 0.0 
                step_start = time.perf_counter()
                
                dt_data = 0.0
                dt_forward = 0.0
                dt_backward = 0.0

                # Zero grad for next accum cycle 
                optimiser.zero_grad(set_to_none=True)

                # Grad accum loop 
                for micro_step in range(grad_accum_steps):
                    t0 = time.perf_counter()
                    # get next batch, restart iterator if epoch ends
                    try: 
                        batch = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        batch = next(train_iter)
                    
                    input_ids = batch['input_ids'].to(device)

                    # Create (input, target) pairs for next tokn preidiction
                    inputs = input_ids[:, :-1] # (batch, seq_len-1)
                    targets = input_ids[:, 1:] # (batch, seq_len-1)
                    
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    dt_data += (t1 - t0)

                    # Forward pass with mixed precision 
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=config.use_amp):
                        logits = model(inputs)
                        loss = nn.functional.cross_entropy(
                            logits.reshape(-1, logits.size(-1)), # (B*T, vocab_size)
                            targets.reshape(-1), # (B*T)
                        )

                        # scale loss by accumulation steps so total gradient 
                        # is the MEAN over the effective batch, not the SUM 
                        loss = loss / grad_accum_steps

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    dt_forward += (t2 - t1)

                    # backward pass (gradients accum automatically)
                    scaler.scale(loss).backward()
                    
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t3 = time.perf_counter()
                    dt_backward += (t3 - t2)

                    step_loss += loss.item()

                t4 = time.perf_counter()

                # optimiser step (once per global step)
                # unscale gradients before clipping 
                scaler.unscale_(optimizer=optimiser)

                # clip grad norms 
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip
                )

                # Optimiser step + scaler update 
                scaler.step(optimizer=optimiser)
                scaler.update()

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t5 = time.perf_counter()
                dt_optim = t5 - t4

                step_time = time.perf_counter() - step_start

                current_lr = scheduler.get_last_lr()[0]
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"loss": f"{step_loss:.4f}", "lr": f"{current_lr:.1e}"})

                # ── Logging ──────────────────────────────────────────────
                if global_step % config.log_interval == 0 and global_step>0:
                    tokens_per_step = (
                        config.micro_batch_size
                        * (config.max_seq_len - 1)  # -1 because input is shifted
                        * grad_accum_steps
                    )
                    tracker.log_train_step(
                        step=global_step,
                        loss=step_loss,
                        lr=current_lr,
                        grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        tokens_processed=tokens_per_step,
                        step_time=step_time,
                    )
                    # We remove the logger.info spam here since tqdm handles it now
                
                # ── Validation ───────────────────────────────────────────
                if global_step % config.eval_interval == 0 and global_step>0:
                    val_metrics = validate(model, val_loader, device, config.use_amp)
                    tracker.log_validation(
                        step=global_step,
                        val_loss=val_metrics["val_loss"],
                        perplexity=val_metrics["perplexity"],
                    )
                    logger.info(
                        f"\nStep {global_step} Validation | loss={val_metrics['val_loss']:.4f} | "
                        f"ppl={val_metrics['perplexity']:.2f}"
                    )
                
                # ── Checkpointing ────────────────────────────────────────
                if global_step % config.save_interval == 0 and global_step>0:
                    ckpt_path = (
                        Path(config.checkpoint_dir) / f"step_{global_step}.pt"
                    )
                    save_checkpoint(
                        model=model,
                        optimiser=optimiser,
                        scheduler=scheduler,
                        scaler=scaler,
                        step=global_step,
                        config=config,
                        path=ckpt_path,
                    )

                # ── Text Generation ────────────────────────────────────────
                if global_step % config.generate_interval == 0 and global_step > 0:
                    logger.info(f"\nGenerating samples at step {global_step}...")
                    samples = generate_samples(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        prompt="The meaning of life is",
                        max_new_tokens=50
                    )
                    
                    # Print to console
                    for i, text in enumerate(samples):
                        logger.info(f"Sample {i+1}:\n{text}\n")
                    
                    # Log to wandb using an HTML format for nice formatting
                    html_str = "<h3>Generated Samples</h3>"
                    for i, text in enumerate(samples):
                        html_str += f"<b>Sample {i+1}:</b><br/>{text}<hr/>"
                    
                    wandb.log({"samples/generated_text": wandb.Html(html_str)}, step=global_step)

                # LR scheduler step 
                scheduler.step()

                global_step += 1 
                
        except KeyboardInterrupt:
            logger.info("\nTraining interrupted by user. Saving emergency checkpoint...")
            
        finally:
            pbar.close()
            # ── Final checkpoint after training completes (or crashes) ──
            final_path = Path(config.checkpoint_dir) / f"step_{global_step}_interrupted.pt"
            if global_step >= config.total_steps:
                final_path = Path(config.checkpoint_dir) / "final.pt"

            save_checkpoint(
                model=model,
                optimiser=optimiser,
                scheduler=scheduler,
                scaler=scaler,
                step=global_step,
                config=config,
                path=final_path,
            )
            logger.info(f"Training loop concluded. Last state saved at step {global_step}.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m gpt.train <config.yaml> [--override value ...]")
        sys.exit(1)
    
    config = load_config(
        yaml_path=sys.argv[1],
        cli_args=sys.argv[2:],
    )
    
    train(config)
