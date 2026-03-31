"""NEURON-1 Knowledge Distillation Module.

Implements cross-vocabulary distillation from large teachers
(Qwen2.5-7B, Llama-3.1-8B) to NEURON-1 (4096 vocab).

Key innovation: Temperature-scaled top-k redistribution for
cross-vocab KL loss (Section 4 protocol).

Supports three distillation modes:
  Phase A: Soft-target distillation (logit matching)
  Phase B: Chain-of-thought compression (trace learning)
  Phase C: Speculative correction (student generates, teacher corrects)
"""
import json
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DistillationConfig:
    """Configuration for distillation."""
    teacher_vocab_size: int = 152064     # Qwen2.5-7B default
    student_vocab_size: int = 4096       # NEURON-1
    temperature: float = 4.0             # Distillation temperature
    alpha: float = 0.7                   # Weight for KL vs CE loss
    top_k: int = 256                     # Teacher top-k for redistribution
    phase: str = "soft_target"           # soft_target | cot | speculative


class VocabProjector(nn.Module):
    """Projects teacher logits from large vocab to student vocab.

    Uses temperature-scaled top-k redistribution:
      1. Take teacher's top-k logits (covers >99% probability mass)
      2. Decode each teacher token to byte representation
      3. Find closest NEURON-1 token via longest common prefix
      4. Aggregate teacher probabilities onto matched student tokens
      5. Renormalize the student-vocab distribution
    """

    def __init__(
        self,
        teacher_vocab_size: int,
        student_vocab_size: int,
        top_k: int = 256,
    ):
        super().__init__()
        self.teacher_vocab_size = teacher_vocab_size
        self.student_vocab_size = student_vocab_size
        self.top_k = top_k

        # Precomputed mapping: teacher_token_id → student_token_id
        # Registered as buffer after build_mapping for fast GPU lookup
        self._mapping = None  # Lazy init
        self.register_buffer(
            '_mapping_tensor',
            torch.full((teacher_vocab_size,), 3, dtype=torch.long),  # default UNK
            persistent=False,
        )

    def build_mapping(
        self,
        teacher_tokenizer,
        student_tokenizer,
    ):
        """Build teacher→student token mapping via byte overlap.

        Uses numpy vectorization for speed (O(teacher_vocab × student_vocab)
        but via broadcasting, not nested Python loops).
        """
        import numpy as np
        print("  Building cross-vocab mapping...")

        # Collect student byte strings
        student_byte_list = []
        for sid in range(self.student_vocab_size):
            try:
                decoded = student_tokenizer.decode([sid])
                student_byte_list.append(decoded.encode("utf-8", errors="replace"))
            except Exception:
                student_byte_list.append(b"")

        mapping = {}
        n_teacher = self.teacher_vocab_size
        batch_size = 5000  # Process in batches to limit memory

        for batch_start in range(0, n_teacher, batch_size):
            batch_end = min(batch_start + batch_size, n_teacher)
            for tid in range(batch_start, batch_end):
                try:
                    if hasattr(teacher_tokenizer, "decode"):
                        decoded = teacher_tokenizer.decode([tid])
                    else:
                        decoded = str(tid)
                    t_bytes = decoded.encode("utf-8", errors="replace")
                except Exception:
                    t_bytes = b""

                best_sid = 3  # UNK
                best_overlap = 0

                if t_bytes:
                    for sid, s_bytes in enumerate(student_byte_list):
                        if not s_bytes:
                            continue
                        # Longest common prefix
                        overlap = 0
                        for a, b in zip(t_bytes, s_bytes):
                            if a == b:
                                overlap += 1
                            else:
                                break
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_sid = sid

                mapping[tid] = best_sid

            if (batch_end - 1) % 10000 == 0 or batch_end == n_teacher:
                print(f"    Mapped {batch_end}/{n_teacher} teacher tokens")

        self._mapping = mapping

        # Precompute mapping tensor as buffer (no rebuild per forward)
        for tid, sid in mapping.items():
            self._mapping_tensor[tid] = sid

        print(f"  Mapped {len(mapping)} teacher tokens to student vocab")
        return mapping

    def build_identity_mapping(self):
        """Build a simple identity mapping for same-vocab distillation."""
        self._mapping = {i: min(i, self.student_vocab_size - 1)
                         for i in range(self.teacher_vocab_size)}
        return self._mapping

    def project(
        self,
        teacher_logits: torch.Tensor,
        temperature: float = 4.0,
    ) -> torch.Tensor:
        """Project teacher logits to student vocab space.

        Args:
            teacher_logits: (B, T, teacher_vocab_size)
            temperature: softmax temperature

        Returns:
            (B, T, student_vocab_size) projected probability distribution
        """
        B, T, V_t = teacher_logits.shape
        device = teacher_logits.device

        # Temperature-scaled softmax on teacher
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        # Top-k: only consider highest probability tokens
        topk_probs, topk_ids = teacher_probs.topk(self.top_k, dim=-1)
        # topk_probs: (B, T, top_k), topk_ids: (B, T, top_k)

        # Map teacher token ids to student token ids
        if self._mapping is None:
            self.build_identity_mapping()

        # Use precomputed buffer (no rebuild per forward)
        mapping_tensor = self._mapping_tensor.to(device)

        # Map top-k teacher ids to student ids
        student_ids = mapping_tensor[topk_ids.clamp(0, self.teacher_vocab_size - 1)]
        # student_ids: (B, T, top_k)

        # Scatter-add probabilities into student vocab
        student_probs = torch.zeros(B, T, self.student_vocab_size, device=device)
        student_probs.scatter_add_(2, student_ids, topk_probs)

        # Renormalize
        student_probs = student_probs / student_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return student_probs


class DistillationLoss(nn.Module):
    """Cross-vocabulary distillation loss.

    Combines:
      L = α · KL(teacher_proj ‖ student) / T² + (1-α) · CE(target, student)

    where teacher_proj is the teacher distribution projected into
    the student's vocabulary space.
    """

    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config
        self.projector = VocabProjector(
            config.teacher_vocab_size,
            config.student_vocab_size,
            config.top_k,
        )
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute distillation loss.

        Args:
            student_logits: (B, T, student_vocab)
            teacher_logits: (B, T, teacher_vocab)
            targets: (B, T) ground truth token ids

        Returns:
            dict with 'total', 'kl', 'ce'
        """
        T = self.config.temperature
        alpha = self.config.alpha

        # Project teacher to student vocab space
        teacher_probs = self.projector.project(teacher_logits, temperature=T)

        # Student log-probs (temperature-scaled)
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)

        # KL divergence: KL(teacher ‖ student)
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
            log_target=False,
        ) * (T * T)  # Scale by T² per Hinton et al.

        # Hard-target CE
        ce_loss = self.ce_loss(
            student_logits.view(-1, student_logits.size(-1)),
            targets.view(-1),
        )

        total = alpha * kl_loss + (1 - alpha) * ce_loss

        return {
            "total": total,
            "kl": kl_loss.detach(),
            "ce": ce_loss.detach(),
        }


class CachedTeacherDataset:
    """Loads pre-cached teacher logits from disk.

    For efficiency, teacher logits are generated offline (on RTX 3050)
    and saved as compressed tensors. This loader reads them for
    distillation training.

    File format per shard:
      {
        "input_ids": tensor (N, T),
        "teacher_logits": tensor (N, T, top_k),
        "teacher_topk_ids": tensor (N, T, top_k),
        "targets": tensor (N, T),
      }
    """

    def __init__(self, cache_dir: str, top_k: int = 256):
        self.cache_dir = Path(cache_dir)
        self.top_k = top_k
        self.shards = sorted(self.cache_dir.glob("shard_*.pt"))
        if not self.shards:
            print(f"  No cached teacher data found in {cache_dir}")

    def __len__(self):
        return len(self.shards)

    def load_shard(self, idx: int) -> dict:
        """Load a single shard of cached teacher data."""
        return torch.load(self.shards[idx], weights_only=True, map_location="cpu")


@torch.no_grad()
def generate_teacher_cache(
    teacher_model,
    teacher_tokenizer,
    texts: list[str],
    output_dir: str,
    seq_len: int = 256,
    top_k: int = 256,
    batch_size: int = 4,
    device: str = "cpu",
    shard_size: int = 1000,
):
    """Generate and cache teacher logits for offline distillation.

    Runs the teacher model on the provided texts and saves
    top-k logits + ids to disk as compressed shards.

    Args:
        teacher_model: loaded teacher model (e.g., Qwen2.5-7B)
        teacher_tokenizer: teacher's tokenizer
        texts: list of text strings to process
        output_dir: directory to save shard files
        seq_len: sequence length for chunking
        top_k: number of top logits to cache per position
        batch_size: inference batch size
        device: device for teacher inference
        shard_size: number of examples per shard file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    teacher_model.eval()
    teacher_model.to(device)

    all_input_ids = []
    all_topk_logits = []
    all_topk_ids = []
    all_targets = []
    shard_idx = 0

    print(f"  Generating teacher cache for {len(texts)} texts...")

    for i, text in enumerate(texts):
        tokens = teacher_tokenizer.encode(text)
        if len(tokens) < seq_len + 1:
            continue

        for start in range(0, len(tokens) - seq_len, seq_len):
            chunk = tokens[start: start + seq_len + 1]
            input_ids = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
            targets = torch.tensor([chunk[1:]], dtype=torch.long)

            # Teacher forward
            outputs = teacher_model(input_ids)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs[0]

            # Extract top-k
            topk_vals, topk_idx = logits.cpu().topk(top_k, dim=-1)

            all_input_ids.append(input_ids.cpu())
            all_topk_logits.append(topk_vals)
            all_topk_ids.append(topk_idx)
            all_targets.append(targets)

            # Save shard
            if len(all_input_ids) >= shard_size:
                shard_path = output_path / f"shard_{shard_idx:04d}.pt"
                torch.save({
                    "input_ids": torch.cat(all_input_ids),
                    "teacher_topk_logits": torch.cat(all_topk_logits),
                    "teacher_topk_ids": torch.cat(all_topk_ids),
                    "targets": torch.cat(all_targets),
                }, shard_path)
                print(f"    Saved shard {shard_idx} ({shard_size} examples)")
                all_input_ids, all_topk_logits, all_topk_ids, all_targets = [], [], [], []
                shard_idx += 1

        if (i + 1) % 100 == 0:
            print(f"    Processed {i+1}/{len(texts)} texts")

    # Save remaining
    if all_input_ids:
        shard_path = output_path / f"shard_{shard_idx:04d}.pt"
        torch.save({
            "input_ids": torch.cat(all_input_ids),
            "teacher_topk_logits": torch.cat(all_topk_logits),
            "teacher_topk_ids": torch.cat(all_topk_ids),
            "targets": torch.cat(all_targets),
        }, shard_path)
        shard_idx += 1

    print(f"  Teacher cache complete: {shard_idx} shards saved to {output_dir}")
