"""Minimal GPTQ quantizer.

Reference: Frantar et al., "GPTQ: Accurate Post-Training Quantization for
Generative Pre-trained Transformers" (arXiv:2210.17323).

This is a from-scratch implementation, not a wrapper around auto-gptq.
Kept deliberately small (~150 LoC) so the integration with ICS is
transparent. Supports:
  - per-group symmetric INT4 (group_size default 128)
  - Hessian-based column ordering
  - blockwise processing to bound memory
  - standard 1% diagonal damping for numerical stability

The quantizer consumes a (out_features, in_features) weight matrix and
the (in_features, in_features) input Hessian, and produces:
  - Q:    quantized weight (same shape as W)
  - scales: (n_groups,) float32 per-group scales
  - zeros:  (n_groups,) int32 per-group zero points (always 0 for symmetric)
  - perm:   (in_features,) int64 column permutation used by GPTQ

The caller is responsible for applying `perm` to the original weight
columns before storing Q (or for storing perm and un-permuting at
dequant time).
"""
from __future__ import annotations

import torch


def compute_layer_hessian(
    model: torch.nn.Module,
    calibration_inputs: list[dict[str, torch.Tensor]],
    layer_filter: callable | None = None,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Compute the input Hessian H = 2 * sum x x^T for each Linear layer.

    For each calibration sample, run a forward pass with hooks that capture
    the input to each target Linear layer. Sum the outer products over
    all tokens across all samples, divide by total token count, multiply
    by 2 (so the resulting L2 reconstruction loss matches GPTQ's derivation
    where the loss is ||Wx - Qx||^2 = ||err . x||^2 and Hessian = 2 X X^T).

    Args:
        model: an nn.Module with a forward(input_ids=...) method.
        calibration_inputs: list of dicts with at least 'input_ids' key.
            Each is a batch (e.g., {'input_ids': tensor[1, T]}).
        layer_filter: optional (name, module) -> bool to restrict which
            Linear layers get Hessians computed. Defaults to all nn.Linear.
        device: device to run forward on.

    Returns:
        Dict mapping layer name -> (in_features, in_features) Hessian on CPU.
    """
    was_training = model.training
    model.eval()

    # Hooks: capture the input to each target layer
    captured: dict[str, list[torch.Tensor]] = {}
    handles = []

    def make_hook(name: str):
        def hook(module, inputs, output):
            x = inputs[0]
            # x shape: (B, T, in_features) for HF causal LM linear layers
            # Flatten to (B*T, in_features)
            x = x.detach().reshape(-1, x.shape[-1]).to(torch.float32)
            captured.setdefault(name, []).append(x)
        return hook

    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if layer_filter is not None and not layer_filter(name, mod):
            continue
        handles.append(mod.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            for batch in calibration_inputs:
                batch = {k: v.to(device) for k, v in batch.items()}
                model(**batch, use_cache=False)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    # Build Hessian per layer: H = 2 * X^T X / (B*T)
    hessians: dict[str, torch.Tensor] = {}
    for name, xs in captured.items():
        X = torch.cat(xs, dim=0)  # (N, in_features)
        n = X.shape[0]
        H = 2.0 * (X.T @ X) / max(n, 1)
        H = H.cpu()
        # Dead-column cleanup: zero diag -> swap to 1 to keep Cholesky happy
        dead = torch.diag(H) == 0
        if dead.any():
            H[dead, dead] = 1.0
        hessians[name] = H
    return hessians


def gptq_quantize(
    W: torch.Tensor,
    H: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    percdamp: float = 0.01,
    blocksize: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a weight matrix using GPTQ.

    Args:
        W: (out_features, in_features) weight matrix, will be modified.
        H: (in_features, in_features) input Hessian (must be PD).
        bits: target bit-width. 4 -> symmetric INT4 in [-8, 7].
        group_size: number of columns that share a scale.
        percdamp: relative damping (fraction of mean H diagonal).
        blocksize: GPTQ processes this many columns at a time (memory).

    Returns:
        Q:      (out_features, in_features) int8 quantized weight.
        scales: (in_features // group_size,) float32 per-group scales.
        zeros:  (in_features // group_size,) int32 per-group zeros (0 for symmetric).
        perm:   (in_features,) int64 column permutation. To recover the
                original weight from Q, apply: W_recovered[:, perm] = dequant(Q).
    """
    assert W.dim() == 2, "W must be 2D"
    out_features, in_features = W.shape
    assert H.shape == (in_features, in_features), f"H shape mismatch: {H.shape}"

    device = W.device
    W = W.detach().to(torch.float32).clone()
    H = H.detach().to(torch.float32).clone().to(device)

    # 1. Dead columns (zero Hessian diag) -> set W column to 0
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    # 2. Damping: add percdamp * mean(diag(H)) to the diagonal
    damp = percdamp * torch.mean(torch.diag(H))
    diag_idx = torch.arange(in_features, device=device)
    H[diag_idx, diag_idx] += damp

    # 3. Cholesky-based inverse
    try:
        L = torch.linalg.cholesky(H)
    except Exception:
        # Retry with progressively larger damping
        ok = False
        for i in range(100):
            extra = (i + 1) * 0.001 * torch.mean(torch.diag(H))
            H_retry = H.clone()
            H_retry[diag_idx, diag_idx] += extra
            try:
                L = torch.linalg.cholesky(H_retry)
                ok = True
                break
            except Exception:
                continue
        if not ok:
            raise RuntimeError("GPTQ: Cholesky failed even with damping")

    H_inv = torch.cholesky_inverse(L)
    H_inv_chol = torch.linalg.cholesky(H_inv, upper=True)

    # 4. Permute columns by descending Hessian diagonal (most-important first)
    perm = torch.argsort(torch.diag(H), descending=True).to(torch.int64)
    W = W[:, perm].contiguous()
    # Note: H is no longer needed in the original layout; we permuted W already.

    # 5. Blockwise GPTQ
    Q = torch.zeros_like(W)
    # Pre-compute per-group scales/zeros
    n_groups = (in_features + group_size - 1) // group_size
    scales = torch.ones(n_groups, dtype=torch.float32, device=device)
    zeros = torch.zeros(n_groups, dtype=torch.int32, device=device)

    qmax = (1 << (bits - 1)) - 1  # e.g., 7 for 4-bit signed
    qmin = -(1 << (bits - 1))     # e.g., -8 for 4-bit signed

    for i1 in range(0, in_features, blocksize):
        i2 = min(i1 + blocksize, in_features)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = H_inv_chol[i1:i2, i1:i2]

        for j in range(count):
            col = i1 + j
            w = W1[:, j]
            d = Hinv1[j, j]

            # Compute scale for this group (or reuse the previous)
            g = col // group_size
            if col % group_size == 0:
                # Start of a new group: compute scale from the next group_size columns
                g_end = min((g + 1) * group_size, in_features)
                group_w = W[:, g * group_size:g_end]  # already permuted
                absmax = group_w.abs().amax()
                if absmax < 1e-12:
                    scales[g] = 1.0
                    zeros[g] = 0
                else:
                    scales[g] = float(absmax / qmax)
                    zeros[g] = 0  # symmetric

            # Quantize w using the current group's scale
            scale = scales[g]
            q_w = torch.clamp(torch.round(w / scale), qmin, qmax)
            Q1[:, j] = q_w

            # Compute error and propagate to remaining columns in this block
            err = (w - q_w * scale) / d
            W1[:, j:] -= err.unsqueeze(1) * Hinv1[j, j:].unsqueeze(0)
            Err1[:, j] = err

        Q[:, i1:i2] = Q1

        # Propagate block error to remaining blocks
        if i2 < in_features:
            W[:, i2:] -= Err1 @ H_inv_chol[i1:i2, i2:]

    # Convert Q to int8 for storage (values are in [qmin, qmax], fit in int8)
    Q_int8 = Q.clamp(qmin, qmax).round().to(torch.int8)
    return Q_int8, scales.cpu(), zeros.cpu(), perm.cpu()


def dequantize_gptq(
    Q: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    perm: torch.Tensor,
    group_size: int = 128,
    in_features: int | None = None,
) -> torch.Tensor:
    """Reconstruct the weight from GPTQ quantized output.

    The output is in the GPTQ-processed column order; to get the original
    weight, un-permute: W_original[:, perm] = dequant(Q).

    Args:
        Q:      (out_features, in_features) int8 quantized weight.
        scales: (in_features // group_size,) float32 per-group scales.
        zeros:  (in_features // group_size,) int32 per-group zeros.
        perm:   (in_features,) int64 column permutation from gptq_quantize.
        group_size: number of columns per group.
        in_features: total number of columns (defaults to Q.shape[1]).

    Returns:
        W: (out_features, in_features) float32 dequantized weight, in
        the original column order.
    """
    out_features, q_cols = Q.shape
    n_features = in_features or q_cols
    assert q_cols == n_features, f"Q has {q_cols} cols but expected {n_features}"

    n_groups = (n_features + group_size - 1) // group_size
    W = torch.zeros((out_features, n_features), dtype=torch.float32, device=Q.device)
    for g in range(n_groups):
        start = g * group_size
        end = min((g + 1) * group_size, n_features)
        scale = float(scales[g].item())
        zero = int(zeros[g].item())
        W[:, start:end] = (Q[:, start:end].float() - zero) * scale

    # Un-permute: the stored Q has columns in GPTQ's processing order;
    # the original weight is recovered by W_original[:, perm] = W.
    W_orig = torch.zeros_like(W)
    W_orig[:, perm.long()] = W
    return W_orig
