"""NEURON-1 Training Loop.

Complete training infrastructure with:
  - Cosine LR schedule with warmup
  - Compound loss (CE + pred + contrast + compress + geo)
  - Gradient clipping
  - Checkpoint save/resume
  - Metric logging
  - Phase-aware curriculum support

Usage:
    python -m neuron1.train --data-dir data/tinystories --epochs 5
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neuron1.config import Neuron1Config
from neuron1.model import Neuron1
from neuron1.loss import Neuron1Loss, Neuron1WithHooks
from neuron1.data import TextDataset, SimpleTokenizer, load_tinystories
from neuron1.curriculum import CurriculumScheduler


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  COSINE LR SCHEDULE WITH WARMUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CosineWarmupScheduler:
    """Cosine annealing with linear warmup.

    Schedule from Section 4:
      Warmup: 1000 steps linear 0 → peak_lr
      Cosine: peak_lr → min_lr over remaining steps
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 1000,
        total_steps: int = 76000,
        peak_lr: float = 1e-3,
        min_lr: float = 1e-5,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.peak_lr * self.current_step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            progress = min(progress, 1.0)
            return self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TRAINING ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Trainer:
    """NEURON-1 training engine.

    Handles the complete training loop with logging, checkpointing,
    and curriculum phase tracking.
    """

    def __init__(
        self,
        model: Neuron1,
        config: Neuron1Config,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        output_dir: str = "checkpoints",
        peak_lr: float = 1e-3,
        min_lr: float = 1e-5,
        warmup_steps: int = 1000,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
        log_interval: int = 50,
        eval_interval: int = 500,
        save_interval: int = 1000,
        device: str = "cpu",
    ):
        self.config = config
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model with hooks for auxiliary losses
        self.model = model.to(self.device)
        self.hooked_model = Neuron1WithHooks(self.model)

        # Loss
        self.criterion = Neuron1Loss()

        # Optimizer (AdamW with weight decay)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=peak_lr,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
        )

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Schedule
        total_steps = len(train_loader) * 100  # estimate
        self.scheduler = CosineWarmupScheduler(
            self.optimizer, warmup_steps, total_steps, peak_lr, min_lr
        )

        # Hyperparams
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval

        # Curriculum scheduler (optional)
        self.curriculum = CurriculumScheduler()

        # Tracking
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.history = []

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []

        for batch_idx, (input_ids, targets) in enumerate(self.train_loader):
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            # Forward (through hooked model)
            logits, _, _ = self.hooked_model(input_ids)

            # Compound loss
            losses = self.criterion(logits, targets, self.hooked_model, input_ids)
            loss = losses["total"]

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )

            # Step
            self.optimizer.step()
            lr = self.scheduler.step()
            self.global_step += 1

            # Curriculum phase transition check
            phase_cfg = self.curriculum.step(self.global_step)
            if phase_cfg is not None:
                # Update loss lambdas
                self.criterion.lambda_pred = phase_cfg.lambda_pred
                self.criterion.lambda_collapse = getattr(phase_cfg, 'lambda_contrast', 0.05)
                self.criterion.lambda_compress = phase_cfg.lambda_compress
                # Handle freeze/unfreeze
                if phase_cfg.freeze_slow:
                    self.model.freeze_slow_layers()
                else:
                    self.model.unfreeze_slow_layers()

            # Log
            epoch_losses.append(losses["ce"].item())

            if self.global_step % self.log_interval == 0:
                avg_loss = sum(epoch_losses[-self.log_interval:]) / min(
                    len(epoch_losses), self.log_interval
                )
                self._log(epoch, batch_idx, losses, lr, grad_norm, avg_loss)

            # Eval
            if self.val_loader and self.global_step % self.eval_interval == 0:
                val_loss = self.evaluate()
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best.pt")
                self.model.train()

            # Save
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(f"step_{self.global_step}.pt")

        return sum(epoch_losses) / max(len(epoch_losses), 1)

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        n_batches = 0

        for input_ids, targets in self.val_loader:
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            logits, _, _ = self.model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
            total_loss += loss.item()
            n_batches += 1

        val_loss = total_loss / max(n_batches, 1)
        print(f"  [EVAL] step={self.global_step} val_loss={val_loss:.4f}")
        return val_loss

    def _log(self, epoch, batch_idx, losses, lr, grad_norm, avg_ce):
        """Print training metrics."""
        entry = {
            "step": self.global_step,
            "epoch": epoch,
            "ce": losses["ce"].item(),
            "pred": losses["pred"].item(),
            "collapse": losses["collapse"].item(),
            "compress": losses["compress"].item(),
            "total": losses["total"].item(),
            "lr": lr,
            "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
        }
        self.history.append(entry)

        print(
            f"  [{epoch}:{batch_idx:>5d}] step={self.global_step:>6d} | "
            f"CE={avg_ce:.4f} pred={entry['pred']:.4f} "
            f"clp={entry['collapse']:.4f} cmp={entry['compress']:.4f} | "
            f"lr={lr:.2e} grad={entry['grad_norm']:.2f}"
        )

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_step": self.scheduler.current_step,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": vars(self.config),
        }
        torch.save(checkpoint, path)
        print(f"  [SAVE] {path}")

    def load_checkpoint(self, path: str):
        """Resume from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.current_step = checkpoint["scheduler_step"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"  [LOAD] Resumed from {path} at step {self.global_step}")

    def save_history(self, filename: str = "training_history.json"):
        """Save training metrics history."""
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    @torch.no_grad()
    def generate(self, prompt_tokens: list[int], max_new: int = 50, temp: float = 0.8) -> list[int]:
        """Autoregressive generation using recurrent state.

        AUDIT FIX: previous version re-encoded the entire context at every
        step AND carried forward states — double-counting history and
        corrupting recurrent state. Fixed to encode prompt once, then
        generate one token at a time using state for history.
        """
        self.model.eval()
        tokens = list(prompt_tokens)
        fast_states = None
        slow_states = None

        # Encode full prompt ONCE
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        logits, fast_states, slow_states = self.model(input_ids, fast_states, slow_states)

        # Generate one token at a time using recurrent state
        for _ in range(max_new):
            next_logits = logits[0, -1] / temp
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)
            # Feed ONLY the new token — state carries history
            input_ids = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            logits, fast_states, slow_states = self.model(input_ids, fast_states, slow_states)

        return tokens


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    parser = argparse.ArgumentParser(description="Train NEURON-1")
    parser.add_argument("--data-dir", type=str, default="data/tinystories")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max-stories", type=int, default=-1)
    args = parser.parse_args()

    print("=" * 60)
    print("NEURON-1 TRAINING")
    print("=" * 60)

    # Config
    config = Neuron1Config(max_seq_len=args.seq_len)
    model = Neuron1(config)
    total_params = model.count_parameters()
    print(f"  Model parameters: {total_params:,}")
    print(f"  Device: {args.device}")

    # Tokenizer
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)

    # Data
    print(f"\n  Loading data from {args.data_dir}...")
    texts = load_tinystories(args.data_dir, args.max_stories)

    # Split 90/10
    split_idx = int(len(texts) * 0.9)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]

    train_dataset = TextDataset(train_texts, tokenizer, args.seq_len)
    val_dataset = TextDataset(val_texts, tokenizer, args.seq_len) if val_texts else None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0,
    ) if val_dataset else None

    print(f"  Train: {len(train_dataset):,} chunks")
    print(f"  Val:   {len(val_dataset):,} chunks" if val_dataset else "  Val: None")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Steps/epoch: {len(train_loader):,}")

    # Trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        peak_lr=args.lr,
        warmup_steps=args.warmup,
        device=args.device,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    print(f"\n{'=' * 60}")
    print("  TRAINING START")
    print(f"{'=' * 60}\n")

    for epoch in range(args.epochs):
        t0 = time.time()
        avg_loss = trainer.train_epoch(epoch)
        elapsed = time.time() - t0
        print(f"\n  Epoch {epoch} complete: avg_ce={avg_loss:.4f} time={elapsed:.1f}s\n")

    # Save final
    trainer.save_checkpoint("final.pt")
    trainer.save_history()

    # Generate a sample
    print(f"\n{'=' * 60}")
    print("  SAMPLE GENERATION")
    print(f"{'=' * 60}")
    prompt = "Once upon a time"
    prompt_tokens = tokenizer.encode(prompt)
    generated = trainer.generate(prompt_tokens, max_new=100)
    print(f"  Prompt: {prompt}")
    decoded = tokenizer.decode(generated)
    # Handle Windows console encoding limitations
    safe_output = decoded.encode("ascii", errors="replace").decode("ascii")
    print(f"  Output: {safe_output}")
    print()


if __name__ == "__main__":
    main()
