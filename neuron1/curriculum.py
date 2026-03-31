"""NEURON-1 Curriculum Learning Scheduler.

5-phase curriculum from Section 4 (Training Strategy):
  Phase 1: Simple text (TinyStories, short sequences)
  Phase 2: Longer context (paragraph-level, increased seq_len)
  Phase 3: Multi-domain mixing (Wikipedia, code, QA)
  Phase 4: Reasoning traces (CoT, step-by-step)
  Phase 5: Distillation interleave (teacher-student KL)

Each phase defines:
  - Data mixtures (dataset weights)
  - Sequence length schedule
  - Loss coefficient schedule
  - Frozen layer schedule
"""
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class PhaseConfig:
    """Configuration for a single curriculum phase."""
    name: str
    start_step: int
    end_step: int
    seq_len: int
    batch_size: int
    # Data mixture weights (must sum to ~1.0)
    data_weights: dict[str, float] = field(default_factory=dict)
    # Loss coefficient overrides
    lambda_pred: float = 0.1
    lambda_contrast: float = 0.05
    lambda_compress: float = 0.01
    lambda_geo: float = 0.01
    # Distillation
    distill_alpha: float = 0.0   # 0 = no distillation
    distill_temp: float = 4.0
    # Layer freezing
    freeze_slow: bool = False
    # Learning rate factor (multiplied by base LR)
    lr_factor: float = 1.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DEFAULT 10B-TOKEN CURRICULUM (35-40 Colab sessions)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEFAULT_CURRICULUM = [
    PhaseConfig(
        name="phase1_simple",
        start_step=0,
        end_step=15000,      # ~2B tokens
        seq_len=128,
        batch_size=64,
        data_weights={"tinystories": 0.7, "shakespeare": 0.2, "simple_wiki": 0.1},
        lambda_pred=0.05,    # Low pred loss early (let model learn basics)
        lambda_contrast=0.02,
        freeze_slow=False,
    ),
    PhaseConfig(
        name="phase2_context",
        start_step=15000,
        end_step=30000,      # ~2B tokens
        seq_len=256,
        batch_size=32,
        data_weights={"tinystories": 0.3, "wiki": 0.4, "books": 0.2, "code": 0.1},
        lambda_pred=0.1,     # Full pred loss
        lambda_contrast=0.05,
        freeze_slow=False,
    ),
    PhaseConfig(
        name="phase3_multidomain",
        start_step=30000,
        end_step=50000,      # ~2.5B tokens
        seq_len=512,
        batch_size=16,
        data_weights={"wiki": 0.3, "books": 0.2, "code": 0.2, "qa": 0.2, "math": 0.1},
        lambda_pred=0.1,
        lambda_contrast=0.05,
        lambda_compress=0.02,
        freeze_slow=False,
    ),
    PhaseConfig(
        name="phase4_reasoning",
        start_step=50000,
        end_step=65000,      # ~1.5B tokens
        seq_len=512,
        batch_size=16,
        data_weights={"cot_traces": 0.4, "qa": 0.3, "code": 0.2, "math": 0.1},
        lambda_pred=0.15,    # Increase predictive coding for reasoning
        lambda_contrast=0.05,
        lambda_compress=0.02,
        lambda_geo=0.02,
        freeze_slow=True,    # Start freezing slow layers
        lr_factor=0.5,       # Reduce LR for fine-tuning
    ),
    PhaseConfig(
        name="phase5_distill",
        start_step=65000,
        end_step=76000,      # ~1.5B tokens
        seq_len=512,
        batch_size=16,
        data_weights={"cot_traces": 0.3, "teacher_cache": 0.4, "qa": 0.2, "code": 0.1},
        distill_alpha=0.7,   # Heavy distillation
        distill_temp=4.0,
        lambda_pred=0.1,
        lambda_contrast=0.03,
        freeze_slow=True,
        lr_factor=0.3,
    ),
]


class CurriculumScheduler:
    """Manages curriculum phase transitions during training.

    Tracks the current phase, handles transitions, and provides
    the active configuration for each training step.
    """

    def __init__(self, phases: list[PhaseConfig] | None = None):
        self.phases = phases or DEFAULT_CURRICULUM
        self.current_phase_idx = 0
        self._phase_log = []

    @property
    def current_phase(self) -> PhaseConfig:
        return self.phases[self.current_phase_idx]

    @property
    def total_steps(self) -> int:
        return self.phases[-1].end_step

    def get_phase(self, step: int) -> PhaseConfig:
        """Get the active phase for a given step."""
        for i, phase in enumerate(self.phases):
            if phase.start_step <= step < phase.end_step:
                if i != self.current_phase_idx:
                    self._on_phase_change(i, step)
                return phase

        # Past all phases — return last
        return self.phases[-1]

    def _on_phase_change(self, new_idx: int, step: int):
        """Handle phase transition."""
        old = self.phases[self.current_phase_idx]
        new = self.phases[new_idx]

        self._phase_log.append({
            "step": step,
            "from": old.name,
            "to": new.name,
        })

        print(f"\n{'='*60}")
        print(f"  CURRICULUM PHASE TRANSITION at step {step}")
        print(f"  {old.name} -> {new.name}")
        print(f"  seq_len: {old.seq_len} -> {new.seq_len}")
        print(f"  batch_size: {old.batch_size} -> {new.batch_size}")
        print(f"  distill_alpha: {old.distill_alpha} -> {new.distill_alpha}")
        print(f"  freeze_slow: {old.freeze_slow} -> {new.freeze_slow}")
        print(f"{'='*60}\n")

        self.current_phase_idx = new_idx

    def get_data_weights(self, step: int) -> dict[str, float]:
        """Get data mixture weights for current step."""
        return self.get_phase(step).data_weights

    def get_loss_config(self, step: int) -> dict[str, float]:
        """Get loss coefficients for current step."""
        phase = self.get_phase(step)
        return {
            "lambda_pred": phase.lambda_pred,
            "lambda_contrast": phase.lambda_contrast,
            "lambda_compress": phase.lambda_compress,
            "lambda_geo": phase.lambda_geo,
        }

    def should_freeze_slow(self, step: int) -> bool:
        """Whether slow layers should be frozen at this step."""
        return self.get_phase(step).freeze_slow

    def should_distill(self, step: int) -> bool:
        """Whether to use distillation loss at this step."""
        return self.get_phase(step).distill_alpha > 0

    def summary(self) -> str:
        """Print curriculum summary."""
        lines = ["CURRICULUM SCHEDULE:", "=" * 60]
        total_tokens = 0
        for p in self.phases:
            steps = p.end_step - p.start_step
            est_tokens = steps * p.batch_size * p.seq_len
            total_tokens += est_tokens
            lines.append(
                f"  {p.name:22s} | steps {p.start_step:>6d}-{p.end_step:>6d} | "
                f"seq={p.seq_len:>4d} B={p.batch_size:>3d} | "
                f"~{est_tokens/1e9:.1f}B tok | "
                f"distill={p.distill_alpha:.1f} freeze={p.freeze_slow}"
            )
        lines.append(f"  {'TOTAL':22s} | {self.total_steps:>6d} steps | ~{total_tokens/1e9:.1f}B tokens")
        return "\n".join(lines)
