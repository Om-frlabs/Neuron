"""NEURON-1 Compound Training Loss — AUDIT-HARDENED.

AUDIT FIXES:
  1. L_pred: direction unified with forward pass (prev predicts curr)
  2. L_contrast: DISABLED — adjacent BPE tokens have no semantic guarantee.
     Replaced with collapse-prevention regularizer (variance target).
  3. L_compress: replaced fake-VAE KL with variance regularizer that
     keeps per-feature workspace activation variance near 1.0.
  4. L_geo: removed entirely (was already dead code returning 0.0).
  5. Neuron1WithHooks: fixed __getattr__ proxy — overrides state_dict()
     and load_state_dict() to delegate to inner model, preventing
     silent key namespace mismatches.

Loss function:
  L_total = L_CE + λ₁·L_pred + λ₂·L_collapse + λ₃·L_compress
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Neuron1Loss(nn.Module):
    """Compound loss for NEURON-1 training.

    Args:
        lambda_pred:     Weight for predictive coding loss (default: 0.1)
        lambda_collapse: Weight for representation collapse prevention (default: 0.05)
        lambda_compress: Weight for workspace compression loss (default: 0.01)
        label_smoothing: Label smoothing factor for CE (default: 0.0)
    """

    def __init__(
        self,
        lambda_pred: float = 0.1,
        lambda_collapse: float = 0.05,
        lambda_compress: float = 0.01,
        lambda_moe: float = 0.01,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.lambda_pred = lambda_pred
        self.lambda_collapse = lambda_collapse
        self.lambda_compress = lambda_compress
        self.lambda_moe = lambda_moe

        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing, ignore_index=-100
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module,
        input_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute compound loss.

        Returns dict with 'total', 'ce', 'pred', 'collapse', 'compress'.
        """
        B, T, V = logits.shape

        # ── L_CE: Cross-Entropy ──
        l_ce = self.ce_loss(logits.view(-1, V), targets.view(-1))

        # ── L_pred: Predictive Coding Loss ──
        l_pred = self._predictive_loss(model)

        # ── L_collapse: Representation Collapse Prevention ──
        l_collapse = self._collapse_loss(model)

        # ── L_compress: Workspace Variance Regularizer ──
        l_compress = self._compression_loss(model)

        # ── L_moe: Mixture of Experts Load Balancing Loss ──
        l_moe = self._moe_loss(model)

        total = (
            l_ce
            + self.lambda_pred * l_pred
            + self.lambda_collapse * l_collapse
            + self.lambda_compress * l_compress
            + self.lambda_moe * l_moe
        )

        return {
            "total": total,
            "ce": l_ce.detach(),
            "pred": l_pred.detach(),
            "collapse": l_collapse.detach(),
            "compress": l_compress.detach(),
            "moe": l_moe.detach() if isinstance(l_moe, torch.Tensor) else l_moe,
        }

    def _moe_loss(self, model: nn.Module) -> torch.Tensor:
        if hasattr(model, 'moe_loss') and model.moe_loss is not None:
            return model.moe_loss
        return torch.tensor(0.0, device=next(model.parameters()).device)

    def _predictive_loss(self, model: nn.Module) -> torch.Tensor:
        """Predictive coding loss: minimize per-layer prediction errors.

        AUDIT FIX: direction unified with forward pass.
        Layer i-1's output predicts layer i's output.
        L_pred = Σ_l ||curr_l - predictor_l(prev_l)||² / ||curr_l||²
        """
        if not hasattr(model, '_fast_layer_outputs') or not model._fast_layer_outputs:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        outputs = model._fast_layer_outputs
        n_compared = 0

        for i in range(1, len(outputs)):
            curr = outputs[i]
            prev = outputs[i - 1]

            # Handle sequence length mismatch from temporal striding
            if curr.shape[1] != prev.shape[1]:
                if curr.shape[1] < prev.shape[1]:
                    ratio = (prev.shape[1] + curr.shape[1] - 1) // max(curr.shape[1], 1)
                    curr = curr.repeat_interleave(ratio, dim=1)
                    curr = curr[:, :prev.shape[1]]
                else:
                    ratio = (curr.shape[1] + prev.shape[1] - 1) // max(prev.shape[1], 1)
                    prev = prev.repeat_interleave(ratio, dim=1)
                    prev = prev[:, :curr.shape[1]]

            # AUDIT FIX: prev predicts curr (unified with forward pass)
            pred = model.fast_layers[i].predictor(prev)
            error = curr - pred
            norm = curr.norm() + 1e-8
            loss = loss + (error.norm() / norm) ** 2
            n_compared += 1

        return loss / max(n_compared, 1)

    def _collapse_loss(self, model: nn.Module) -> torch.Tensor:
        """Prevent representation collapse in workspace.

        AUDIT FIX: replaces broken contrastive loss. Penalizes when
        per-feature variance deviates from 1.0 — encourages the workspace
        to use all available dimensions without collapsing to a subspace.
        """
        if not hasattr(model, '_workspace_z') or model._workspace_z is None:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        z = model._workspace_z  # (B, T, d_bottleneck)
        # Per-feature variance across batch and time
        z_flat = z.reshape(-1, z.shape[-1])  # (B*T, d_bot)
        var_per_feature = z_flat.var(dim=0)  # (d_bot,)
        # Encourage each feature to have variance ≈ 1.0
        return (var_per_feature - 1.0).pow(2).mean()

    def _compression_loss(self, model: nn.Module) -> torch.Tensor:
        """Workspace compression: keep activations at moderate scale.

        AUDIT FIX: replaced fake-VAE KL with variance regularizer.
        Encourages per-token workspace vectors to have variance near 1.0,
        preventing both collapse (too small) and explosion (too large).
        """
        if not hasattr(model, '_workspace_z') or model._workspace_z is None:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        z = model._workspace_z  # (B, T, d_bottleneck)
        # Per-token variance across features
        var_per_token = z.var(dim=-1)  # (B, T)
        return (var_per_token - 1.0).pow(2).mean()


class Neuron1WithHooks(nn.Module):
    """Wrapper that captures intermediate activations for auxiliary losses.

    AUDIT FIX: overrides state_dict() and load_state_dict() to delegate
    to the inner model, preventing silent key namespace mismatches.
    Previously, saving through the wrapper produced keys like
    'model.fast_layers.0...' while loading without wrapper expected
    'fast_layers.0...' — a silent checkpoint corruption bug.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._fast_layer_outputs = []
        self._workspace_z = None

        # Register hooks
        self._hooks = []
        for i, layer in enumerate(model.fast_layers):
            hook = layer.register_forward_hook(self._make_fast_hook(i))
            self._hooks.append(hook)

        # Hook on workspace norm to capture bottleneck activations
        if hasattr(model, 'workspace'):
            hook = model.workspace.norm.register_forward_hook(self._workspace_hook)
            self._hooks.append(hook)

    def _make_fast_hook(self, layer_idx):
        def hook(module, input, output):
            x = output[0]
            if layer_idx >= len(self._fast_layer_outputs):
                self._fast_layer_outputs.append(x)
            else:
                self._fast_layer_outputs[layer_idx] = x
        return hook

    def _workspace_hook(self, module, input, output):
        self._workspace_z = output

    def forward(self, *args, **kwargs):
        self._fast_layer_outputs = []
        self._workspace_z = None
        return self.model(*args, **kwargs)

    # AUDIT FIX: delegate state_dict to inner model
    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
