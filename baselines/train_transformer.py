"""Baseline Transformer Training Loop.

Mirrors neuron1/train.py but for the vanilla transformer baseline.
Same data pipeline, same schedule, same evaluation — only the model differs.

Usage:
    python -m baselines.train_transformer --data-dir data/tinystories --epochs 5 --device cpu
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baselines.transformer import BaselineTransformer
from neuron1.data import TextDataset, SimpleTokenizer, load_tinystories


class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps=1000, total_steps=76000,
                 peak_lr=1e-3, min_lr=1e-5):
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

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.peak_lr * self.current_step / self.warmup_steps
        progress = (self.current_step - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1)
        progress = min(progress, 1.0)
        return self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
            1 + math.cos(math.pi * progress))


class BaselineTrainer:
    def __init__(self, model, train_loader, val_loader=None,
                 output_dir="checkpoints/baseline", peak_lr=1e-3,
                 warmup_steps=500, grad_clip=1.0, device="cpu",
                 log_interval=50, eval_interval=500, save_interval=1000):
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.1)

        self.train_loader = train_loader
        self.val_loader = val_loader

        total_steps = len(train_loader) * 100
        self.scheduler = CosineWarmupScheduler(
            self.optimizer, warmup_steps, total_steps, peak_lr)

        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.history = []

    def train_epoch(self, epoch):
        self.model.train()
        epoch_losses = []

        for batch_idx, (input_ids, targets) in enumerate(self.train_loader):
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            logits = self.model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            lr = self.scheduler.step()
            self.global_step += 1

            epoch_losses.append(loss.item())

            if self.global_step % self.log_interval == 0:
                avg = sum(epoch_losses[-self.log_interval:]) / min(
                    len(epoch_losses), self.log_interval)
                gn = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                print(f"  [{epoch}:{batch_idx:>5d}] step={self.global_step:>6d} | "
                      f"CE={avg:.4f} | lr={lr:.2e} grad={gn:.2f}")
                self.history.append({
                    "step": self.global_step, "epoch": epoch,
                    "ce": loss.item(), "lr": lr, "grad_norm": gn})

            if self.val_loader and self.global_step % self.eval_interval == 0:
                val_loss = self.evaluate()
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best.pt")
                self.model.train()

            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(f"step_{self.global_step}.pt")

        return sum(epoch_losses) / max(len(epoch_losses), 1)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total, n = 0, 0
        for input_ids, targets in self.val_loader:
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            logits = self.model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
            total += loss.item()
            n += 1
        val_loss = total / max(n, 1)
        print(f"  [EVAL] step={self.global_step} val_loss={val_loss:.4f}")
        return val_loss

    def save_checkpoint(self, filename):
        path = self.output_dir / filename
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }, path)
        print(f"  [SAVE] {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  [LOAD] Resumed from {path} at step {self.global_step}")

    @torch.no_grad()
    def generate(self, prompt_tokens, max_new=50, temp=0.8):
        self.model.eval()
        tokens = list(prompt_tokens)
        for _ in range(max_new):
            ctx = tokens[-self.model.max_seq_len:]
            ids = torch.tensor([ctx], dtype=torch.long, device=self.device)
            logits = self.model(ids)
            next_logits = logits[0, -1] / temp
            probs = torch.softmax(next_logits, dim=-1)
            tokens.append(torch.multinomial(probs, 1).item())
        return tokens

    def save_history(self, filename="training_history.json"):
        with open(self.output_dir / filename, "w") as f:
            json.dump(self.history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train Baseline Transformer")
    parser.add_argument("--data-dir", type=str, default="data/tinystories")
    parser.add_argument("--output-dir", type=str, default="checkpoints/baseline")
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
    print("BASELINE TRANSFORMER TRAINING")
    print("=" * 60)

    model = BaselineTransformer(max_seq_len=args.seq_len)
    total_params = model.count_parameters()
    print(f"  Model parameters: {total_params:,}")
    print(f"  Device: {args.device}")

    tokenizer = SimpleTokenizer(vocab_size=4096)

    print(f"\n  Loading data from {args.data_dir}...")
    texts = load_tinystories(args.data_dir, args.max_stories)

    split_idx = int(len(texts) * 0.9)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]

    train_dataset = TextDataset(train_texts, tokenizer, args.seq_len)
    val_dataset = TextDataset(val_texts, tokenizer, args.seq_len) if val_texts else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0) if val_dataset else None

    print(f"  Train: {len(train_dataset):,} chunks")
    if val_dataset:
        print(f"  Val:   {len(val_dataset):,} chunks")

    trainer = BaselineTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        output_dir=args.output_dir, peak_lr=args.lr,
        warmup_steps=args.warmup, device=args.device)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    print(f"\n{'=' * 60}")
    print("  TRAINING START")
    print(f"{'=' * 60}\n")

    for epoch in range(args.epochs):
        t0 = time.time()
        avg_loss = trainer.train_epoch(epoch)
        elapsed = time.time() - t0
        print(f"\n  Epoch {epoch} complete: avg_ce={avg_loss:.4f} time={elapsed:.1f}s\n")

    trainer.save_checkpoint("final.pt")
    trainer.save_history()

    print(f"\n{'=' * 60}")
    print("  SAMPLE GENERATION")
    print(f"{'=' * 60}")
    prompt = "Once upon a time"
    prompt_tokens = tokenizer.encode(prompt)
    generated = trainer.generate(prompt_tokens, max_new=100)
    print(f"  Prompt: {prompt}")
    decoded = tokenizer.decode(generated)
    safe_output = decoded.encode("ascii", errors="replace").decode("ascii")
    print(f"  Output: {safe_output}")
    print()


if __name__ == "__main__":
    main()
