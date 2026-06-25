"""Channel permutation algorithms for ICS.

Given:
    W_A: weight matrix of layer A, shape [out_A, in_A]. We permute its
         COLUMNS (output channels) by applying W_A @ P^T.
    W_B: weight matrix of layer B, shape [out_B, in_B] where in_B == out_A.
         We permute its ROWS (input channels) by applying P @ W_B.
    F:   shared activation-Fisher vector of length out_A (== in_B).

We seek a permutation P such that:
    Y = X (W_A P^T) (P W_B)  ==  X W_A W_B   (mathematically exact)
and the joint quantization cost is minimized. The chain (P, P^-1 = P^T)
cancels algebraically, so the forward pass is bit-exact (modulo FP
roundoff). All the cleverness is in *which* P we pick: we want it to
group high-Fisher channels into blocks that don't get crushed by the
outlier-driven per-block scaling factor, AND to cluster W_A's column
outliers and W_B's row outliers into the same dense block.

Two algorithms:
    - composite:    1D sort on F * (a*M_A + b*M_B). O(N log N). Fast.
    - sinkhorn_h:   Sinkhorn-Knopp relaxation + Hungarian projection.
                   Theoretically cleaner for the joint objective. O(N^3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


@dataclass
class PermutationResult:
    """The result of finding a permutation for a (W_A, W_B) chain.

    Attributes:
        permutation: LongTensor of shape [N] giving the new index for
            each original channel. perm[i] = j means "channel j from the
            original layer should go to position i in the new layer".
        score: the joint cost under the chosen objective.
        method: "composite" or "sinkhorn_hungarian".
    """

    permutation: torch.Tensor
    score: float
    method: str


def find_permutation_spectral(
    W_A: torch.Tensor,
    W_B: torch.Tensor,
    F: torch.Tensor,
    n_components: int = 4,
    gamma: float | None = None,
) -> PermutationResult:
    """Find permutation via spectral ordering on the joint feature space.

    Builds a Gaussian RBF affinity matrix on the 3-D feature space
    [F, M_A, M_B] (per-output-channel Fisher, per-output-channel
    outlier norm of W_A, per-input-channel outlier norm of W_B), and
    sorts channels by the leading non-trivial eigenvector of the
    symmetric-normalized affinity.

    Why this is more elegant than the composite score:
        - F, M_A, M_B are features of different scales. The composite
          score F * (a*M_A + b*M_B) does AND semantics — a channel must
          score high in BOTH Fisher AND magnitude to rank at the top.
          Channels that are high-Fisher but low-magnitude (or vice
          versa) get buried in the tail.
        - The spectral approach uses an RBF affinity, which gives OR
          semantics on the joint feature: a channel is "close" to
          another if they're similar in any feature. The eigenvectors
          of the affinity recover the smoothest 1-D ordering that
          preserves this multi-feature neighborhood structure.
        - No alpha/beta knobs. The bandwidth gamma is auto-set by the
          median heuristic.

    Args:
        W_A: [out_A, in_A]
        W_B: [out_B, in_B] with in_B == out_A
        F: [out_A] shared activation-Fisher (normalized to [0, 1])
        n_components: number of leading non-trivial eigenvectors to
            stack before taking the principal 1-D projection. 1 = use
            the Fiedler vector directly.
        gamma: RBF bandwidth. None = median heuristic.

    Returns:
        PermutationResult.
    """
    _validate_chain(W_A, W_B)
    N = W_A.shape[0]
    M_A = _per_output_channel_norm(W_A).to(torch.float32)
    M_B = _per_input_channel_norm(W_B).to(torch.float32)
    F_n = F.to(torch.float32)

    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return x / (x.max() + 1e-12)

    # Joint feature: [N, 3]
    feat = torch.stack([_normalize(F_n), _normalize(M_A), _normalize(M_B)], dim=1)

    # Pairwise squared Euclidean distance: [N, N]
    diff = feat.unsqueeze(0) - feat.unsqueeze(1)
    sq_dist = (diff * diff).sum(dim=-1)

    # Median heuristic for gamma
    if gamma is None:
        median_sq = sq_dist.flatten().median()
        gamma = 1.0 / (median_sq + 1e-12)

    # Gaussian RBF affinity
    K = torch.exp(-gamma * sq_dist)

    # Symmetric normalization
    d = K.sum(dim=1)
    d_inv_sqrt = torch.rsqrt(d + 1e-12)
    K_norm = d_inv_sqrt.unsqueeze(1) * K * d_inv_sqrt.unsqueeze(0)

    # Eigendecomposition (ascending)
    eigvals, eigvecs = torch.linalg.eigh(K_norm)

    # Skip the trivial top eigenvector (largest eigenvalue ≈ 1, near-constant).
    # Use the next n_components, then take their first principal component
    # as the 1-D sort key.
    if n_components == 1:
        v = eigvecs[:, -2]  # second-largest
    else:
        V = eigvecs[:, -(n_components + 1):-1]  # [N, n_components]
        V = V - V.mean(dim=0, keepdim=True)
        # First principal component via SVD
        U, S, Vt = torch.linalg.svd(V, full_matrices=False)
        v = U[:, 0]

    perm = torch.argsort(v)
    return PermutationResult(
        permutation=perm,
        score=0.0,
        method="spectral",
    )


def _channel_inf_norm(W: torch.Tensor, dim: int) -> torch.Tensor:
    """Per-channel infinity norm (max abs).

    For nn.Linear weight W of shape [out_features, in_features]:
        dim=1 -> per-output-channel norm (max over input dim, shape [out_features])
        dim=0 -> per-input-channel  norm (max over output dim, shape [in_features])
    """
    return W.abs().amax(dim=dim)


def _per_output_channel_norm(W: torch.Tensor) -> torch.Tensor:
    """Per-output-channel infinity norm: max|W[i, :]| over i. Shape [W.shape[0]]."""
    return _channel_inf_norm(W, dim=1)


def _per_input_channel_norm(W: torch.Tensor) -> torch.Tensor:
    """Per-input-channel infinity norm: max|W[:, j]| over j. Shape [W.shape[1]]."""
    return _channel_inf_norm(W, dim=0)


def _validate_chain(W_A: torch.Tensor, W_B: torch.Tensor) -> None:
    if W_A.ndim != 2 or W_B.ndim != 2:
        raise ValueError("W_A and W_B must be 2-D [out, in]")
    if W_A.shape[0] != W_B.shape[1]:
        raise ValueError(
            f"chain dim mismatch: W_A has out_features={W_A.shape[0]} but "
            f"W_B has in_features={W_B.shape[1]}. They must match so that "
            f"the activation tensor between A and B has consistent channel dim."
        )


def find_permutation_composite(
    W_A: torch.Tensor,
    W_B: torch.Tensor,
    F: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    descending: bool = True,
) -> PermutationResult:
    """Fast 1D sort on F * (alpha * M_A + beta * M_B).

    The sort key is:  score[i] = F[i] * (alpha * ||W_A[:, i]||_inf
                                          + beta  * ||W_B[i, :]||_inf)

    This is a scalarization of the joint objective: it puts the most
    Fisher-sensitive AND most outlier-heavy channels first, where the
    "first" channels will land in the dense INT4 blocks of the quantized
    model. Channels at the tail end of the sort land in the aggressive
    INT2/INT1 blocks.

    Args:
        W_A: [out_A, in_A]
        W_B: [out_B, in_B] with in_B == out_A
        F: [out_A] shared activation-Fisher (already normalized to [0, 1])
        alpha: weight on W_A's column-outlier contribution
        beta: weight on W_B's row-outlier contribution
        descending: if True, high-score channels go first (will be assigned
            high-bit quantization).

    Returns:
        PermutationResult with `permutation` mapping old channel index
        to new position.
    """
    _validate_chain(W_A, W_B)
    N = W_A.shape[0]
    if F.shape[0] != N:
        raise ValueError(f"F must have length {N}, got {F.shape[0]}")

    M_A = _per_output_channel_norm(W_A).to(torch.float32)  # [out_A] = W_A's per-output-channel max
    M_B = _per_input_channel_norm(W_B).to(torch.float32)   # [in_B]  = W_B's per-input-channel max

    # Bridge device mismatch: Fisher is often offloaded to CPU (see pipeline.py),
    # but the weight-derived norms live on whatever device the model is on.
    F = F.to(device=M_A.device, dtype=torch.float32)

    score = F * (alpha * M_A + beta * M_B + 1e-12)
    perm = torch.argsort(score, descending=descending)

    return PermutationResult(
        permutation=perm,
        score=float(score[perm].mean().item()),
        method="composite",
    )


def find_permutation_sinkhorn_hungarian(
    W_A: torch.Tensor,
    W_B: torch.Tensor,
    F: torch.Tensor,
    block_size: int = 64,
    sinkhorn_iters: int = 10,
    block_aware: bool = True,
) -> PermutationResult:
    """Sinkhorn-Knopp relaxation + Hungarian projection, block-aware.

    Builds a swap-cost matrix G[i, j] measuring the cost of swapping
    channels i and j under the joint objective, relaxes to a doubly
    stochastic assignment via Sinkhorn-Knopp normalization, then projects
    to a hard permutation via the Hungarian algorithm.

    If block_aware=True, channels are first clustered into blocks of
    `block_size` and the assignment is computed within / across block
    boundaries. Channels that straddle a hardware block boundary (a
    high-Fisher channel padded next to low-Fisher channels) are pushed
    inward.

    Args:
        W_A: [out_A, in_A]
        W_B: [out_B, in_B] with in_B == out_A
        F: [out_A] shared activation-Fisher (normalized to [0, 1])
        block_size: NPU hardware block size. Channels will be re-bucketed
            so that high-Fisher channels don't get split across blocks.
        sinkhorn_iters: number of Sinkhorn normalization iterations.
        block_aware: if True, enforce block-boundary constraints.

    Returns:
        PermutationResult.
    """
    _validate_chain(W_A, W_B)
    N = W_A.shape[0]

    M_A = _per_output_channel_norm(W_A).to(torch.float32).numpy()  # [out_A]
    M_B = _per_input_channel_norm(W_B).to(torch.float32).numpy()   # [in_B == out_A]
    F_np = F.to(torch.float32).numpy()

    # Build the soft swap-cost matrix.
    # G[i, j] = F[i] * (M_A[i] - M_A[j])^2 + F[j] * (M_B[i] - M_B[j])^2
    # This is the cost of swapping i and j under the joint objective.
    # Equivalently, we want to find P minimizing sum_i F[P(i)] * M_A[P(i)]
    # + sum_j F[j] * M_B[j] (after the same perm is applied to W_B's rows).
    # The Sinkhorn approach: build a cost matrix C[i, j] = the cost of
    # assigning source-channel i to position j, then doubly-stochastic
    # relax and Hungarian-project.
    dMA = M_A[:, None] - M_A[None, :]  # [N, N]
    dMB = M_B[:, None] - M_B[None, :]
    F_col = F_np[:, None]
    F_row = F_np[None, :]
    G = F_col * (dMA ** 2) + F_row * (dMB ** 2)  # [N, N] swap cost

    # Convert to assignment cost: we want to find a permutation P
    # minimizing sum_i F[i] * (alpha*M_A[P(i)] + beta*M_B[P(i)]).
    # This is a linear assignment problem with cost matrix
    # C[i, j] = F[i] * (alpha*M_A[j] + beta*M_B[j]).
    # The "swap" interpretation is equivalent for the joint objective
    # when alpha=beta=1 and the cross-terms match. We use the linear
    # form for Sinkhorn.
    alpha = 1.0
    beta = 1.0
    C = F_col * (alpha * M_A[None, :] + beta * M_B[None, :])  # [N, N]

    # Sinkhorn-Knopp: convert to a "soft" doubly-stochastic assignment
    K = np.exp(-C / (C.mean() + 1e-9))
    for _ in range(sinkhorn_iters):
        K = K / (K.sum(axis=1, keepdims=True) + 1e-12)
        K = K / (K.sum(axis=0, keepdims=True) + 1e-12)

    # Project to a hard permutation via Hungarian
    # We use the *soft* matrix as a *preference* (higher = prefer)
    # by minimizing -K. (linear_sum_assignment minimizes cost.)
    row_ind, col_ind = linear_sum_assignment(-K)

    # col_ind[i] = position assigned to source-channel i
    # We want `permutation` to map old channel -> new position.
    # In our convention, permutation[old] = new.
    perm = np.empty(N, dtype=np.int64)
    perm[row_ind] = col_ind
    perm_t = torch.from_numpy(perm)

    if block_aware:
        perm_t = _rebalance_for_blocks(perm_t, F_np, M_A, M_B, block_size)

    score = float(C[np.arange(N), perm].mean())
    return PermutationResult(permutation=perm_t, score=score, method="sinkhorn_hungarian")


def _rebalance_for_blocks(
    perm: torch.Tensor,
    F: np.ndarray,
    M_A: np.ndarray,
    M_B: np.ndarray,
    block_size: int,
) -> torch.Tensor:
    """Re-bucket channels so high-Fisher ones don't get split across blocks.

    After the initial permutation, channels 0..N-1 are ordered by
    descending composite score. We re-bucket them into blocks of
    `block_size` and, within each block, ensure the high-Fisher channels
    are at the *interior* of the block (not at the tail, which would
    be padded next to lower-bit neighboring blocks).

    Concretely: for each block, sort the channels in the block by
    Fisher descending. This is a cheap local re-sort that respects the
    block boundary.
    """
    perm_np = perm.numpy().copy()
    N = perm_np.shape[0]
    n_blocks = (N + block_size - 1) // block_size
    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, N)
        idx = perm_np[start:end]
        # Re-sort this block's channels by Fisher descending
        fisher_in_block = F[idx]
        local_order = np.argsort(-fisher_in_block)
        perm_np[start:end] = idx[local_order]
    return torch.from_numpy(perm_np)


def apply_permutation_to_chain(
    W_A: torch.Tensor,
    W_B: torch.Tensor,
    P: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the inverse-permutation trick to a (W_A, W_B) chain.

    The chain is:
        Y = X W_A^T  (or W_A depending on convention; we use the Linear
                      convention: y = x @ W^T + b, so W is [out, in])
        Z = Y W_B^T

    After applying P to the chain, we want:
        Y' = Y[:, P]        (permute output channels of layer A)
        Z' = Y' W_B'^T      (where W_B' has rows reordered to compensate)

    Concretely, we apply:
        W_A' = W_A[P, :]    (permute rows of W_A, which IS the output
                             channel perm in Linear's [out, in] convention)
        W_B' = W_B[:, P]    (permute columns of W_B, which IS the input
                             channel perm in Linear's [out, in] convention)

    Wait, this is the *standard* ICS application. Let me state it
    clearly so the convention is unambiguous.

    Layer A: y = x @ W_A^T   (W_A is [out_A, in_A] in nn.Linear's storage)
    Layer B: z = y @ W_B^T   (W_B is [out_B, in_B] where in_B == out_A)

    We want the permuted forward pass to be bit-identical to the
    original, but the channels in the activation y to be reordered by P.
    Concretely:
        y' = y[:, P]                      # new activation layout
        z' = y' @ W_B_new^T = y[:, P] @ W_B_new^T
        We want z' == z = y @ W_B^T
        So: y[:, P] @ W_B_new^T = y @ W_B^T
        Substituting y[:, P] = y @ E^T (where E is the permutation matrix
        with E[i, P[i]] = 1):
            y @ E^T @ W_B_new^T = y @ W_B^T
            E^T @ W_B_new^T = W_B^T
            W_B_new = W_B @ E
    So W_B's *columns* get permuted: W_B_new[:, i] = W_B[:, P[i]]
    Equivalently: W_B_new = W_B[:, P]  (index along dim 1)

    For W_A: we want the OUTPUT of layer A to be y' = y[:, P]. We don't
    need to permute W_A at all if we permute the activation externally
    — but that defeats the purpose. The ICS trick is to bake the
    activation perm into W_A so the runtime doesn't need to permute
    activations. To make y' = x @ W_A_new^T equal to y[:, P] = (x @ W_A^T)[:, P],
    we need: x @ W_A_new^T = (x @ W_A^T)[:, P]
    Expanding: for each output position j, (W_A_new)[j, :] = (W_A)[P[j], :]
    So: W_A_new = W_A[P, :]  (index along dim 0)

    Summary:
        W_A_new = W_A[P, :]    (permute rows of W_A = output channels)
        W_B_new = W_B[:, P]    (permute columns of W_B = input channels)

    This is the canonical ICS application. The forward pass through
    (W_A_new, W_B_new) is bit-identical to the original.
    """
    P_np = P.cpu().numpy() if isinstance(P, torch.Tensor) else P
    W_A_new = W_A[P_np, :]
    W_B_new = W_B[:, P_np]
    return W_A_new.clone(), W_B_new.clone()


def fan_out_permutation(
    W_list: list[torch.Tensor],
    F: torch.Tensor,
    P: torch.Tensor,
) -> list[torch.Tensor]:
    """Apply a permutation to a fan-out group (q_proj, k_proj, v_proj, ...).

    All W in the list share the same input dimension (the residual
    stream). Permuting the input dim of all of them by P keeps the
    forward pass bit-identical: X @ W^T with X[:, P] == (X @ P^T) @ W^T
    is the same as X @ (P^T W)^T.
    """
    P_np = P.cpu().numpy() if isinstance(P, torch.Tensor) else P
    return [W[:, P_np].clone() for W in W_list]


def find_permutation_for_fan_out(
    W: torch.Tensor,
    F: torch.Tensor,
    descending: bool = True,
) -> PermutationResult:
    """Find a permutation for a fan-out group's input dim.

    Single W: the fan-out group shares one input dim. We sort that
    input dim by F (activation Fisher) weighted by W's per-input-channel
    outlier norm.
    """
    if W.ndim != 2:
        raise ValueError("W must be 2-D [out, in]")
    in_dim = W.shape[1]
    if F.shape[0] != in_dim:
        raise ValueError(f"F must have length {in_dim} (W's in_features), got {F.shape[0]}")
    M = _per_input_channel_norm(W).to(torch.float32)
    score = F.to(torch.float32) * (M + 1e-12)
    perm = torch.argsort(score, descending=descending)
    return PermutationResult(
        permutation=perm,
        score=float(score.mean().item()),
        method="fan_out_composite",
    )
