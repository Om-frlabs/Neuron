"""NEURON-1 Colab Training Script.

Optimized for Google Colab T4 GPU sessions:
  - Mixed-precision (AMP) training
  - Curriculum-aware training loop
  - Interleaved distillation phases
  - Checkpoint save/resume for session continuity
  - Automatic session time management
  - Weights & Biases logging (optional)

Usage (Colab cell):
    !python -m neuron1.train_colab \\
        --phase phase1_simple \\
        --data-dir /content/data \\
        --output-dir /content/drive/MyDrive/neuron1/checkpoints \\
        --resume /content/drive/MyDrive/neuron1/checkpoints/latest.pt
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from neuron1.config import Neuron1Config
from neuron1.model import Neuron1
from neuron1.loss import Neuron1Loss, Neuron1WithHooks
from neuron1.curriculum import CurriculumScheduler, DEFAULT_CURRICULUM
from neuron1.data import TextDataset, BinaryTokenDataset, SimpleTokenizer, load_tinystories
from neuron1.train import CosineWarmupScheduler


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  COLAB TRAINER (AMP + Curriculum + Distillation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ColabTrainer:
    """Production training engine for Colab T4 sessions."""

    def __init__(
        self,
        model: Neuron1,
        config: Neuron1Config,
        curriculum: CurriculumScheduler,
        output_dir: str = "checkpoints",
        peak_lr: float = 1e-3,
        min_lr: float = 1e-5,
        warmup_steps: int = 1000,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
        grad_accumulation: int = 1,
        log_interval: int = 50,
        eval_interval: int = 500,
        save_interval: int = 500,
        device: str = "cuda",
        use_amp: bool = True,
        max_session_minutes: int = 170,  # Colab T4 ~3h, leave buffer
    ):
        self.config = config
        self.curriculum = curriculum
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model
        self.model = model.to(self.device)
        self.hooked_model = Neuron1WithHooks(self.model)

        # Loss
        self.criterion = Neuron1Loss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=peak_lr, betas=(0.9, 0.95),
            weight_decay=weight_decay,
        )

        # Schedule
        self.scheduler = CosineWarmupScheduler(
            self.optimizer, warmup_steps, curriculum.total_steps,
            peak_lr, min_lr,
        )

        # AMP
        self.use_amp = use_amp and device != "cpu"
        self.scaler = GradScaler("cuda") if self.use_amp else None

        # Training state
        self.global_step = 0
        self.grad_clip = grad_clip
        self.grad_accumulation = grad_accumulation
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.max_session_minutes = max_session_minutes
        self.best_val_loss = float("inf")
        self.history = []
        self.session_start = time.time()
        self._slow_frozen = False

    def train_session(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        max_steps: int | None = None,
    ):
        """Run one Colab training session.

        Trains until max_steps or session time limit, whichever comes first.
        Saves checkpoint at the end for resume in next session.
        """
        self.session_start = time.time()
        max_steps = max_steps or self.curriculum.total_steps

        print(f"\n{'='*60}")
        print(f"  NEURON-1 COLAB SESSION")
        print(f"  Device: {self.device}")
        print(f"  AMP: {'ON' if self.use_amp else 'OFF'}")
        print(f"  Starting step: {self.global_step}")
        print(f"  Target steps: {max_steps}")
        print(f"  Session limit: {self.max_session_minutes}min")
        print(f"{'='*60}\n")
        print(self.curriculum.summary())
        print()

        self.model.train()
        accumulation_loss = 0.0
        accumulation_steps = 0

        while self.global_step < max_steps:
            # Check session time limit
            elapsed_min = (time.time() - self.session_start) / 60
            if elapsed_min > self.max_session_minutes:
                print(f"\n  [TIME] Session limit reached ({elapsed_min:.0f} min)")
                break

            for input_ids, targets in train_loader:
                if self.global_step >= max_steps:
                    break

                # Check time limit every 10 steps
                if self.global_step % 10 == 0:
                    elapsed_min = (time.time() - self.session_start) / 60
                    if elapsed_min > self.max_session_minutes:
                        break

                # Phase-aware configuration
                phase = self.curriculum.get_phase(self.global_step)

                # Handle freeze transitions
                if phase.freeze_slow and not self._slow_frozen:
                    print(f"  [FREEZE] Freezing slow layers at step {self.global_step}")
                    self.model.freeze_slow_layers()
                    self._slow_frozen = True
                elif not phase.freeze_slow and self._slow_frozen:
                    print(f"  [UNFREEZE] Unfreezing slow layers at step {self.global_step}")
                    self.model.unfreeze_slow_layers()
                    self._slow_frozen = False

                # Update loss coefficients
                loss_config = self.curriculum.get_loss_config(self.global_step)
                self.criterion.lambda_pred = loss_config["lambda_pred"]
                self.criterion.lambda_contrast = loss_config["lambda_contrast"]
                self.criterion.lambda_compress = loss_config["lambda_compress"]
                self.criterion.lambda_geo = loss_config["lambda_geo"]

                # Forward + backward
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)

                if self.use_amp:
                    with autocast("cuda"):
                        logits, _, _ = self.hooked_model(input_ids)
                        losses = self.criterion(logits, targets, self.hooked_model, input_ids)
                        loss = losses["total"] / self.grad_accumulation

                    self.scaler.scale(loss).backward()
                    accumulation_loss += loss.item()
                    accumulation_steps += 1

                    if accumulation_steps >= self.grad_accumulation:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()

                        lr = self.scheduler.step()
                        self.global_step += 1
                        accumulation_loss = 0.0
                        accumulation_steps = 0
                else:
                    logits, _, _ = self.hooked_model(input_ids)
                    losses = self.criterion(logits, targets, self.hooked_model, input_ids)
                    loss = losses["total"] / self.grad_accumulation

                    loss.backward()
                    accumulation_loss += loss.item()
                    accumulation_steps += 1

                    if accumulation_steps >= self.grad_accumulation:
                        grad_norm = nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        lr = self.scheduler.step()
                        self.global_step += 1
                        accumulation_loss = 0.0
                        accumulation_steps = 0

                # Logging
                if self.global_step % self.log_interval == 0 and self.global_step > 0:
                    elapsed = time.time() - self.session_start
                    tokens_per_sec = (
                        self.log_interval * phase.batch_size * phase.seq_len / elapsed
                    ) if elapsed > 0 else 0

                    entry = {
                        "step": self.global_step,
                        "phase": phase.name,
                        "ce": losses["ce"].item(),
                        "pred": losses["pred"].item(),
                        "total": losses["total"].item(),
                        "lr": lr if 'lr' in dir() else 0,
                        "tok_per_sec": tokens_per_sec,
                    }
                    self.history.append(entry)

                    print(
                        f"  [{phase.name[:8]:>8s}] step={self.global_step:>6d} | "
                        f"CE={entry['ce']:.4f} total={entry['total']:.4f} | "
                        f"lr={entry['lr']:.2e} | "
                        f"{tokens_per_sec:.0f} tok/s | "
                        f"{elapsed/60:.1f}min"
                    )

                # Eval
                if val_loader and self.global_step % self.eval_interval == 0 and self.global_step > 0:
                    val_loss = self._evaluate(val_loader)
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best.pt")

                # Save
                if self.global_step % self.save_interval == 0 and self.global_step > 0:
                    self.save_checkpoint("latest.pt")

        # Session end — always save
        self.save_checkpoint("latest.pt")
        self._save_history()

        elapsed = (time.time() - self.session_start) / 60
        print(f"\n  Session complete: step={self.global_step}, {elapsed:.1f} min")

    @torch.no_grad()
    def _evaluate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss, n = 0.0, 0
        for input_ids, targets in val_loader:
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            if self.use_amp:
                with autocast("cuda"):
                    logits, _, _ = self.model(input_ids)
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), targets.view(-1)
                    )
            else:
                logits, _, _ = self.model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )
            total_loss += loss.item()
            n += 1
            if n >= 20:
                break
        self.model.train()
        val_loss = total_loss / max(n, 1)
        print(f"  [EVAL] step={self.global_step} val_loss={val_loss:.4f}")
        return val_loss

    def save_checkpoint(self, filename: str):
        path = self.output_dir / filename
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict() if self.scaler else None,
            "scheduler_step": self.scheduler.current_step,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "curriculum_phase": self.curriculum.current_phase_idx,
            "slow_frozen": self._slow_frozen,
            "config": vars(self.config),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.scaler and checkpoint.get("scaler_state"):
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.scheduler.current_step = checkpoint["scheduler_step"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.curriculum.current_phase_idx = checkpoint.get("curriculum_phase", 0)
        self._slow_frozen = checkpoint.get("slow_frozen", False)
        if self._slow_frozen:
            self.model.freeze_slow_layers()
        print(f"  [LOAD] Resumed from step {self.global_step}, phase={self.curriculum.current_phase.name}")

    def _save_history(self):
        path = self.output_dir / "training_history.json"
        # Append to existing history
        existing = []
        if path.exists():
            with open(path) as f:
                existing = json.load(f)
        existing.extend(self.history)
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    parser = argparse.ArgumentParser(description="NEURON-1 Colab Training")
    parser.add_argument("--data-dir", type=str, default="data/tinystories")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-minutes", type=int, default=170)
    parser.add_argument("--max-stories", type=int, default=-1)
    args = parser.parse_args()

    # Config
    config = Neuron1Config(max_seq_len=args.seq_len)
    model = Neuron1(config)
    curriculum = CurriculumScheduler()

    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Device: {args.device}")
    print(f"  AMP: {'OFF' if args.no_amp else 'ON'}")

    # Data — MUST use pre-tokenized binary (RAM-safe)
    from neuron1.data import MixedBinaryDataset
    data_path = Path(args.data_dir)
    bin_path = data_path / "tokens.bin"

    # Check for multi-domain data (tokens_*.bin files)
    multi_bins = sorted(data_path.glob("tokens_*.bin"))

    if multi_bins:
        # Multi-domain mode: weighted mixture
        print(f"\n  Multi-domain mode: {len(multi_bins)} datasets found")
        bin_weights = {}
        for b in multi_bins:
            # Equal weights by default (curriculum weights are aspirational)
            bin_weights[str(b)] = 1.0
        if bin_path.exists():
            bin_weights[str(bin_path)] = 1.0  # Include base tokens.bin too
        full_dataset = MixedBinaryDataset(bin_weights, args.seq_len)
    elif bin_path.exists():
        # Single dataset mode
        print(f"  Using pre-tokenized data: {bin_path}")
        print(f"  File size: {bin_path.stat().st_size / 1e6:.1f} MB")
        full_dataset = BinaryTokenDataset(str(bin_path), args.seq_len)
    else:
        print(f"\n  ERROR: No tokenized data found in {data_path}")
        print(f"  Need either tokens.bin or tokens_*.bin files")
        sys.exit(1)

    n_train = int(len(full_dataset) * 0.95)
    train_dataset = torch.utils.data.Subset(full_dataset, range(n_train))
    val_dataset = torch.utils.data.Subset(full_dataset, range(n_train, len(full_dataset)))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0,
    )

    print(f"  Train: {len(train_dataset):,} chunks")
    print(f"  Val: {len(val_dataset):,} chunks")

    # Trainer
    trainer = ColabTrainer(
        model=model,
        config=config,
        curriculum=curriculum,
        output_dir=args.output_dir,
        peak_lr=args.lr,
        warmup_steps=args.warmup,
        grad_accumulation=args.grad_accum,
        device=args.device,
        use_amp=not args.no_amp,
        max_session_minutes=args.max_minutes,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train_session(
        train_loader=train_loader,
        val_loader=val_loader,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
