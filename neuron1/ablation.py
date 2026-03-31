"""NEURON-1 Ablation Study Framework.

Defines 7 ablation configurations from Section 6:
  1. No DeltaMemory → standard GatedLRU in fast layers
  2. No CompNorm → standard LayerNorm
  3. No Global Workspace → direct d=256 connection
  4. No Predictive Residual → standard skip connections
  5. No Temporal Stride → all stride=1
  6. No Dendritic Mixing → standard MLP
  7. Frozen vs Trainable Slow Layers (3 variants) — GO/NO-GO GATE

Run: python -m neuron1.ablation --steps 5000 --batch-size 32
"""
import argparse
import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neuron1.config import Neuron1Config
from neuron1.model import Neuron1
from neuron1.data import TextDataset, SimpleTokenizer, load_tinystories
from neuron1.train import CosineWarmupScheduler


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ABLATION CONFIGS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class AblationConfig:
    name: str
    description: str
    modify_model: callable  # fn(model) -> modified_model
    is_go_nogo: bool = False  # True for Blocker 3 gate


def _replace_compnorm_with_layernorm(model: Neuron1) -> Neuron1:
    """Replace all CompNorm instances with standard LayerNorm."""
    for name, module in model.named_modules():
        if module.__class__.__name__ == "CompNorm":
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model
            if parent_name:
                for part in parent_name.split("."):
                    parent = getattr(parent, part)
            setattr(parent, child_name, nn.LayerNorm(module.d_model))
    return model


class _PassthroughWorkspace(nn.Module):
    """Returns zeros — used where workspace output is added residually."""
    def forward(self, x):
        return torch.zeros_like(x)


class _PassthroughUpsample(nn.Module):
    """Returns x unchanged — accepts (x, skip, orig_len) signature."""
    def forward(self, x, skip_connection, orig_len):
        return x


def _remove_global_workspace(model: Neuron1) -> Neuron1:
    """Replace workspace with zeros (d=256 passthrough)."""
    model.workspace = _PassthroughWorkspace()
    return model


def _remove_predictive_residual(model: Neuron1) -> Neuron1:
    """Disable predictive residual by patching forward to skip the block.

    Zeroing weights is wrong: with x = error and predicted=0,
    error = prev_output - 0 = prev_output, so x = prev_output
    (adds full previous layer output, not standard skip).
    Instead, patch forward to skip the predictive block entirely.
    """
    for layer in model.fast_layers:
        # Save original forward
        original_forward = layer.forward

        def make_patched_forward(orig_fwd):
            def patched_forward(x, prev_layer_output=None, state=None):
                # Skip predictive residual — pass prev_layer_output=None
                return orig_fwd(x, prev_layer_output=None, state=state)
            return patched_forward

        layer.forward = make_patched_forward(original_forward)
    return model


def _remove_temporal_stride(model: Neuron1) -> Neuron1:
    """Set all fast layer strides to 1."""
    for layer in model.fast_layers:
        layer.stride = 1
    # Upsample becomes passthrough since no stride reduction
    model.upsample = _PassthroughUpsample()
    return model


def _replace_dendritic_with_mlp(model: Neuron1) -> Neuron1:
    """Replace DendriticMixer with a standard 2-layer MLP."""
    for layer in model.fast_layers:
        d = model.config.d_model
        layer.dendritic = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
    return model


def _replace_delta_with_lru(model: Neuron1) -> Neuron1:
    """Replace DeltaMemory in fast layers with GatedLRU."""
    from neuron1.layers import GatedLRU
    for layer in model.fast_layers:
        d = model.config.d_model
        d_s = model.config.d_state
        layer.delta_mem = GatedLRU(d, d_s)
    return model


def _identity(model: Neuron1) -> Neuron1:
    """No modification (baseline)."""
    return model


def _freeze_slow_at_init(model: Neuron1) -> Neuron1:
    """Freeze slow layers immediately (at random init)."""
    model.freeze_slow_layers()
    return model


def _freeze_slow_after_steps(model: Neuron1) -> Neuron1:
    """Marker — freeze slow layers after 2500 steps (handled in runner)."""
    model._freeze_at_step = 2500
    return model


ABLATION_CONFIGS = [
    AblationConfig("baseline", "Full NEURON-1 (no modification)", _identity),
    AblationConfig("no_delta_memory", "Replace DeltaMemory with GatedLRU in fast layers", _replace_delta_with_lru),
    AblationConfig("no_compnorm", "Replace CompNorm with LayerNorm", _replace_compnorm_with_layernorm),
    AblationConfig("no_workspace", "Remove Global Workspace bottleneck", _remove_global_workspace),
    AblationConfig("no_predictive", "Standard skip connections (zero predictor)", _remove_predictive_residual),
    AblationConfig("no_stride", "All layers stride=1 (no temporal hierarchy)", _remove_temporal_stride),
    AblationConfig("no_dendritic", "Standard MLP instead of DendriticMixer", _replace_dendritic_with_mlp),
    # GO/NO-GO GATE — Blocker 3
    AblationConfig("frozen_at_init", "Slow layers frozen at random init", _freeze_slow_at_init, is_go_nogo=True),
    AblationConfig("frozen_after_2500", "Slow layers frozen after 2500 steps", _freeze_slow_after_steps, is_go_nogo=True),
    # baseline with fully trainable slow layers is already the "baseline" config
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ABLATION RUNNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AblationRunner:
    """Runs all ablation experiments and compares results."""

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Neuron1Config,
        max_steps: int = 5000,
        lr: float = 1e-3,
        output_dir: str = "ablation_results",
        device: str = "cpu",
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.max_steps = max_steps
        self.lr = lr
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.results = {}

    def run_single(self, ablation: AblationConfig) -> dict:
        """Run a single ablation experiment."""
        print(f"\n{'='*60}")
        print(f"  ABLATION: {ablation.name}")
        print(f"  {ablation.description}")
        print(f"{'='*60}\n")

        # Fresh model
        model = Neuron1(self.config).to(self.device)
        model = ablation.modify_model(model)

        # Count params
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.lr, betas=(0.9, 0.95), weight_decay=0.1,
        )
        scheduler = CosineWarmupScheduler(
            optimizer, warmup_steps=200, total_steps=self.max_steps,
            peak_lr=self.lr,
        )

        # Training loop
        model.train()
        step = 0
        train_losses = []
        val_losses = []
        log_interval = max(self.max_steps // 20, 1)

        t_start = time.time()

        while step < self.max_steps:
            for input_ids, targets in self.train_loader:
                if step >= self.max_steps:
                    break

                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)

                # Handle deferred freezing
                if hasattr(model, '_freeze_at_step') and step == model._freeze_at_step:
                    print(f"  [FREEZE] Freezing slow layers at step {step}")
                    model.freeze_slow_layers()
                    # Rebuild optimizer with only trainable params
                    optimizer = torch.optim.AdamW(
                        [p for p in model.parameters() if p.requires_grad],
                        lr=self.lr, betas=(0.9, 0.95), weight_decay=0.1,
                    )
                    scheduler = CosineWarmupScheduler(
                        optimizer, warmup_steps=0,
                        total_steps=self.max_steps - step,
                        peak_lr=scheduler.get_lr(),
                    )

                # Forward
                logits, _, _ = model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                step += 1

                train_losses.append(loss.item())

                if step % log_interval == 0:
                    avg = sum(train_losses[-log_interval:]) / min(len(train_losses), log_interval)
                    val_loss = self._evaluate(model)
                    val_losses.append({"step": step, "val_loss": val_loss})
                    elapsed = time.time() - t_start
                    print(
                        f"  step={step:>5d}/{self.max_steps} | "
                        f"train={avg:.4f} val={val_loss:.4f} | "
                        f"lr={scheduler.get_lr():.2e} | {elapsed:.1f}s"
                    )

        # Final evaluation
        final_val = self._evaluate(model)
        elapsed = time.time() - t_start

        result = {
            "name": ablation.name,
            "description": ablation.description,
            "is_go_nogo": ablation.is_go_nogo,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "final_val_loss": final_val,
            "final_train_loss": sum(train_losses[-100:]) / min(len(train_losses), 100),
            "val_history": val_losses,
            "total_time_s": elapsed,
            "steps": step,
        }

        # Save checkpoint
        ckpt_path = self.output_dir / f"{ablation.name}.pt"
        torch.save(model.state_dict(), ckpt_path)

        self.results[ablation.name] = result
        return result

    @torch.no_grad()
    def _evaluate(self, model: nn.Module) -> float:
        """Quick validation loss."""
        model.eval()
        total_loss, n = 0, 0
        for input_ids, targets in self.val_loader:
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            logits, _, _ = model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
            total_loss += loss.item()
            n += 1
            if n >= 10:  # cap eval batches for speed
                break
        model.train()
        return total_loss / max(n, 1)

    def run_all(self):
        """Run all ablation experiments."""
        for ablation in ABLATION_CONFIGS:
            self.run_single(ablation)

        self._print_summary()
        self._check_go_nogo()
        self._save_results()

    def _print_summary(self):
        """Print comparative summary table."""
        print(f"\n{'='*80}")
        print("  ABLATION RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"  {'Name':<25s} {'Val Loss':>10s} {'Delta':>8s} {'Params':>10s} {'Time':>8s}")
        print(f"  {'-'*25} {'-'*10} {'-'*8} {'-'*10} {'-'*8}")

        baseline = self.results.get("baseline", {}).get("final_val_loss", 0)

        for name, r in self.results.items():
            delta = r["final_val_loss"] - baseline
            delta_pct = (delta / baseline * 100) if baseline > 0 else 0
            sign = "+" if delta >= 0 else ""
            marker = " **" if r.get("is_go_nogo") else ""
            print(
                f"  {name:<25s} {r['final_val_loss']:>10.4f} "
                f"{sign}{delta_pct:>6.1f}% {r['trainable_params']:>10,} "
                f"{r['total_time_s']:>7.1f}s{marker}"
            )

    def _check_go_nogo(self):
        """Check the frozen layers go/no-go gate."""
        print(f"\n{'='*80}")
        print("  BLOCKER 3 — FROZEN LAYERS GO/NO-GO GATE")
        print(f"{'='*80}")

        baseline_loss = self.results.get("baseline", {}).get("final_val_loss")
        if baseline_loss is None:
            print("  [ERROR] Baseline not found. Cannot evaluate go/no-go.")
            return

        threshold = 0.10  # 10% degradation threshold

        for name in ["frozen_at_init", "frozen_after_2500"]:
            if name not in self.results:
                continue
            r = self.results[name]
            degradation = (r["final_val_loss"] - baseline_loss) / baseline_loss

            status = "GO" if degradation <= threshold else "NO-GO"
            print(f"\n  {name}:")
            print(f"    Baseline val loss: {baseline_loss:.4f}")
            print(f"    Ablation val loss: {r['final_val_loss']:.4f}")
            print(f"    Degradation: {degradation*100:.1f}%")
            print(f"    Threshold: {threshold*100:.0f}%")
            print(f"    VERDICT: *** {status} ***")

            if status == "NO-GO":
                print(f"    RECOMMENDATION: Abandon full freeze. Use gradual freeze instead.")
                print(f"    See Section 3, Hypothesis C for fallback strategy.")

    def _save_results(self):
        """Save results to JSON."""
        path = self.output_dir / "ablation_results.json"
        # Make JSON-serializable
        serializable = {}
        for name, r in self.results.items():
            serializable[name] = {k: v for k, v in r.items()}
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n  Results saved to {path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    parser = argparse.ArgumentParser(description="NEURON-1 Ablation Studies")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="data/tinystories")
    parser.add_argument("--max-stories", type=int, default=-1)
    parser.add_argument("--output-dir", type=str, default="ablation_results")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only this ablation (comma-separated names)")
    args = parser.parse_args()

    config = Neuron1Config(max_seq_len=args.seq_len)
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)

    # Load data
    texts = load_tinystories(args.data_dir, args.max_stories)
    split_idx = int(len(texts) * 0.9)

    train_dataset = TextDataset(texts[:split_idx], tokenizer, args.seq_len)
    val_dataset = TextDataset(texts[split_idx:], tokenizer, args.seq_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"  Data: {len(train_dataset)} train / {len(val_dataset)} val chunks")
    print(f"  Steps per ablation: {args.steps}")
    print(f"  Ablations to run: {len(ABLATION_CONFIGS)}")

    runner = AblationRunner(
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        max_steps=args.steps,
        lr=args.lr,
        output_dir=args.output_dir,
        device=args.device,
    )

    if args.only:
        names = [n.strip() for n in args.only.split(",")]
        configs = [c for c in ABLATION_CONFIGS if c.name in names]
        for ab in configs:
            runner.run_single(ab)
        runner._print_summary()
        runner._check_go_nogo()
        runner._save_results()
    else:
        runner.run_all()


if __name__ == "__main__":
    main()
