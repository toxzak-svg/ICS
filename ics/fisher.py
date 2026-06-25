"""Activation Fisher Information per channel.

For each nn.Linear layer in a model, computes F[i] = E[(dL/dy_i)^2] where
y_i is the i-th output channel of that layer. The expectation is taken
over a small calibration dataset.

The Fisher vector is the canonical sort key for ICS: it is shared between
a layer's output channels and the next layer's input channels (since
they are the same activation tensor), so there is no "two-sided" conflict
on the activation side. The remaining joint optimization is on the
weight-magnitude side (see ics.permutation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import torch
import torch.nn as nn


@dataclass
class FisherStats:
    """Per-channel Fisher Information for a single layer.

    Attributes:
        name: dotted module path of the layer (e.g. "model.layers.0.self_attn.q_proj")
        fisher: tensor of shape [out_features], F[i] = E[(dL/dy_i)^2]
        n_samples: number of (batch, seq-position) pairs averaged over
    """

    name: str
    fisher: torch.Tensor
    n_samples: int = 0


def _register_fisher_hooks(
    model: nn.Module,
    layer_filter: Callable[[str, nn.Module], bool] | None = None,
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    """Register forward + backward hooks on target layers.

    Returns (activations, grad_buffers) where:
        - activations[name] is the captured output y of layer `name`
          (shape [..., out_features]) and will receive a populated .grad
          after backward().
        - grad_buffers is a list of references we will accumulate (dL/dy)^2 into.
    """
    activations: dict[str, torch.Tensor] = {}
    fisher_accumulators: dict[str, torch.Tensor] = {}
    n_samples: dict[str, int] = {}
    handles = []

    def make_forward_hook(name: str):
        def hook(module, inputs, output):
            # output is [..., out_features]
            if isinstance(output, tuple):
                y = output[0]
            else:
                y = output
            if not y.requires_grad:
                y.requires_grad_(True)
            y.retain_grad()
            activations[name] = y
            if name not in fisher_accumulators:
                fisher_accumulators[name] = torch.zeros(
                    y.shape[-1], dtype=torch.float32, device=y.device
                )
                n_samples[name] = 0
        return hook

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if layer_filter is not None and not layer_filter(name, module):
            continue
        handles.append(module.register_forward_hook(make_forward_hook(name)))

    return activations, fisher_accumulators, n_samples, handles


@torch.no_grad()
def _accumulate_fisher(
    fisher_accumulators: dict[str, torch.Tensor],
    n_samples: dict[str, int],
    activations: dict[str, torch.Tensor],
) -> None:
    """After backward(), fold (dL/dy)^2 into the per-channel accumulator.

    y has shape [..., out_features]. Reduce over all leading dims.
    """
    for name, y in activations.items():
        if y.grad is None:
            continue
        # y.grad has same shape as y
        g2 = y.grad.detach().pow(2).to(torch.float32)
        # reduce over all but the channel dim
        flat = g2.reshape(-1, g2.shape[-1])
        fisher_accumulators[name] += flat.sum(dim=0)
        n_samples[name] += flat.shape[0]


def compute_fisher(
    model: nn.Module,
    tokenizer,
    calibration_texts: Sequence[str],
    layer_filter: Callable[[str, nn.Module], bool] | None = None,
    max_length: int = 512,
    device: str | torch.device = "cuda",
    show_progress: bool = True,
    loss_mode: str = "cross_entropy",
) -> dict[str, FisherStats]:
    """Compute per-channel activation Fisher Information for each Linear layer.

    Args:
        model: a causal LM in eval mode (we set it to train mode internally
            so gradients flow). 4-bit / 8-bit loaded models work as long as
            the underlying Linear supports backward (BitsAndBytes Linear4bit
            does in recent versions).
        tokenizer: an HF tokenizer. Must produce input_ids + attention_mask.
        calibration_texts: iterable of calibration strings. ~64-256 strings
            is usually enough for stable Fisher estimates.
        layer_filter: optional predicate (name, module) -> bool to restrict
            which layers get Fisher computed. Defaults to all nn.Linear.
        max_length: max sequence length for tokenization.
        device: device to run forward+backward on.
        show_progress: print a dot per sample.

    Returns:
        Dict mapping layer name -> FisherStats.
    """
    was_training = model.training
    model.train()
    model.to(device)
    param_requires_grad = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    activations, fisher_accumulators, n_samples, handles = _register_fisher_hooks(
        model, layer_filter
    )

    results: dict[str, FisherStats] = {}
    try:
        for i, text in enumerate(calibration_texts):
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            if input_ids.shape[-1] < 2:
                raise ValueError(
                    "Fisher calibration samples must tokenize to at least 2 tokens; "
                    f"sample {i} produced length {input_ids.shape[-1]}"
                )

            # Causal LM forward
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            logits = out.logits  # [B, T, V]

            if loss_mode == "cross_entropy":
                # Next-token cross-entropy loss (shifted)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)).float(),
                    shift_labels.view(-1),
                    reduction="sum",
                )
            elif loss_mode == "last_logit_mean":
                shift_logits = None
                shift_labels = None
                loss = logits[:, -1, :].float().mean()
            else:
                raise ValueError(f"unknown Fisher loss_mode: {loss_mode}")

            # Backward
            model.zero_grad(set_to_none=True)
            loss.backward()

            _accumulate_fisher(fisher_accumulators, n_samples, activations)
            activations.clear()

            if show_progress and (i + 1) % 8 == 0:
                print(f"  fisher: {i + 1}/{len(calibration_texts)}", flush=True)

            # Detach loss graph
            del loss, out, logits
    finally:
        for h in handles:
            h.remove()
        for p, requires_grad in zip(model.parameters(), param_requires_grad):
            p.requires_grad_(requires_grad)
        model.zero_grad(set_to_none=True)
        if not was_training:
            model.eval()

    for name, acc in fisher_accumulators.items():
        denom = max(n_samples[name], 1)
        results[name] = FisherStats(
            name=name,
            fisher=(acc / denom).to(torch.float32),
            n_samples=n_samples[name],
        )

    return results


def fisher_to_per_channel(
    fisher: dict[str, FisherStats],
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Convert FisherStats to a plain tensor dict with a small floor for stability."""
    out = {}
    for name, fs in fisher.items():
        f = fs.fisher.clone()
        f = f / (f.max() + eps)  # normalize to [0, 1]
        out[name] = f
    return out
