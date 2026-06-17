"""A/B test: composite-score ICS vs spectral ICS.

Both methods produce a valid permutation that preserves the forward pass
exactly (since ICS is a pure perm). They differ in WHERE the perm puts
each channel, which affects the per-block quantization error after the
perm is applied.

Test design:
    - N = 128 channels
    - F is high in the MIDDLE, low at the edges
    - M_A (W_A's column outliers) is high at the START
    - M_B (W_B's row outliers) is high at the END
    - F and M disagree — high-F channels are NOT where high-M channels are

    Both methods should:
    1. Produce a valid permutation (no duplicates, all indices present)
    2. Preserve the forward pass (within FP32 noise)
    3. Cluster high-F and high-M channels in the SAME blocks (so the
       per-block INT4 precision is spent on the right channels)

    Comparison metrics:
    - Forward pass diff (both should be ~1e-5)
    - Total per-block quantization error (Frobenius) under the same
      bit-width assignment
    - Block-level: do the top blocks (INT4) contain the high-F channels?
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from ics.permutation import (
    find_permutation_composite,
    find_permutation_spectral,
    apply_permutation_to_chain,
    fan_out_permutation,
    find_permutation_for_fan_out,
)
from ics.quantize import (
    quantize_blockwise,
    dequantize_blockwise,
    assign_bit_widths,
)


def setup_disagreement_data(N: int = 128, in_dim: int = 64, seed: int = 42):
    """Build W_A, W_B, F with structured disagreement between F and M.

    F is high in the middle (channels 32..96), low at the edges.
    M_A is high at the start (channels 0..32), low elsewhere.
    M_B is high at the end (channels 96..128), low elsewhere.
    """
    torch.manual_seed(seed)
    F = torch.zeros(N)
    F[32:96] = 1.0
    F[:32] = 0.1
    F[96:] = 0.2

    M_A_target = torch.zeros(N)
    M_A_target[:32] = 1.0
    M_A_target[32:] = 0.1

    M_B_target = torch.zeros(N)
    M_B_target[96:] = 1.0
    M_B_target[:96] = 0.1

    # Build W_A, W_B with these magnitude profiles
    W_A = torch.randn(N, in_dim) * 0.1
    for i in range(N):
        W_A[i, :] *= M_A_target[i]
    W_A += 0.01 * torch.randn_like(W_A)

    W_B = torch.randn(in_dim, N) * 0.1
    for i in range(N):
        W_B[:, i] *= M_B_target[i]
    W_B += 0.01 * torch.randn_like(W_B)

    return W_A, W_B, F, M_A_target, M_B_target


def forward_diff(W_A_orig, W_B_orig, perm, X):
    """Compute forward pass diff after applying the perm."""
    P = perm.cpu().numpy()
    W_A_new, W_B_new = apply_permutation_to_chain(W_A_orig, W_B_orig, P)
    Y_orig = X @ W_A_orig.T
    Z_orig = Y_orig @ W_B_orig.T
    Y_perm = X @ W_A_new.T
    Z_perm = Y_perm @ W_B_new.T
    return (Z_perm - Z_orig).abs().max().item()


def quantization_error_after_perm(W_A, W_B, F, perm, block_size: int = 16):
    """Apply the perm, then quantize with Fisher-based bit assignment,
    return total reconstruction error (Frobenius).

    W_A is permuted on dim 0 (rows) and quantized on dim 0.
    W_B is permuted on dim 1 (cols) and quantized on dim 1.
    """
    P = perm.cpu().numpy()
    W_A_p = W_A[P, :]
    W_B_p = W_B[:, P]
    F_p = F[P]
    bits = assign_bit_widths(F_p, block_size=block_size)
    qt_A = quantize_blockwise(W_A_p, bits, block_size=block_size, dim=0)
    qt_B = quantize_blockwise(W_B_p, bits, block_size=block_size, dim=1)
    W_A_dq = dequantize_blockwise(qt_A)
    W_B_dq = dequantize_blockwise(qt_B)
    err_A = (W_A_p - W_A_dq).pow(2).sum().sqrt().item()
    err_B = (W_B_p - W_B_dq).pow(2).sum().sqrt().item()
    return err_A + err_B, bits


def fisher_in_top_blocks(F, perm, block_size: int = 16, top_k_blocks: int = 4):
    """Return the fraction of total Fisher mass that lands in the top INT4 blocks.

    Higher is better: means the perm put the high-Fisher channels where
    the INT4 blocks will sit.
    """
    P = perm.cpu().numpy()
    F_p = F[P]
    n_blocks = (F_p.numel() + block_size - 1) // block_size
    total_F = F_p.sum().item()
    block_fishers = []
    for b in range(n_blocks):
        s = b * block_size
        e = min(s + block_size, F_p.numel())
        block_fishers.append(F_p[s:e].sum().item())
    # Sort blocks by Fisher descending; take top_k_blocks
    sorted_fishers = sorted(block_fishers, reverse=True)
    return sum(sorted_fishers[:top_k_blocks]) / (total_F + 1e-12)


def test_spectral_basic_validity():
    """Spectral ICS produces a valid permutation and preserves forward pass."""
    W_A, W_B, F, _, _ = setup_disagreement_data()
    X = torch.randn(2, 8, W_A.shape[1])
    perm = find_permutation_spectral(W_A, W_B, F)
    P = perm.permutation
    assert P.shape == (W_A.shape[0],)
    assert set(P.tolist()) == set(range(W_A.shape[0])), "spectral produced invalid perm"
    diff = forward_diff(W_A, W_B, P, X)
    print(f"  spectral forward diff: {diff:.3e}")
    assert diff < 1e-3


def test_spectral_vs_composite_on_disagreement():
    """The headline A/B: when F and M disagree, which perm gives lower
    quantization error and better Fisher-in-top-blocks concentration?"""
    W_A, W_B, F, M_A_t, M_B_t = setup_disagreement_data()
    X = torch.randn(2, 8, W_A.shape[1])

    # Composite (default α=β=1)
    perm_c = find_permutation_composite(W_A, W_B, F, alpha=1.0, beta=1.0)
    # Composite (Fisher-dominant: emphasize F)
    perm_cF = find_permutation_composite(W_A, W_B, F, alpha=10.0, beta=10.0)
    # Spectral
    perm_s = find_permutation_spectral(W_A, W_B, F)

    print(f"\n  --- composite (α=β=1) ---")
    fwd_c = forward_diff(W_A, W_B, perm_c.permutation, X)
    err_c, bits_c = quantization_error_after_perm(W_A, W_B, F, perm_c.permutation)
    top_F_c = fisher_in_top_blocks(F, perm_c.permutation, top_k_blocks=4)
    print(f"    forward diff: {fwd_c:.3e}")
    print(f"    quant error:  {err_c:.4f}")
    print(f"    F in top 4 blocks: {top_F_c:.3f}")
    print(f"    bits: {bits_c.tolist()}")

    print(f"\n  --- composite (α=β=10, Fisher-dominant) ---")
    fwd_cF = forward_diff(W_A, W_B, perm_cF.permutation, X)
    err_cF, bits_cF = quantization_error_after_perm(W_A, W_B, F, perm_cF.permutation)
    top_F_cF = fisher_in_top_blocks(F, perm_cF.permutation, top_k_blocks=4)
    print(f"    forward diff: {fwd_cF:.3e}")
    print(f"    quant error:  {err_cF:.4f}")
    print(f"    F in top 4 blocks: {top_F_cF:.3f}")
    print(f"    bits: {bits_cF.tolist()}")

    print(f"\n  --- spectral ---")
    fwd_s = forward_diff(W_A, W_B, perm_s.permutation, X)
    err_s, bits_s = quantization_error_after_perm(W_A, W_B, F, perm_s.permutation)
    top_F_s = fisher_in_top_blocks(F, perm_s.permutation, top_k_blocks=4)
    print(f"    forward diff: {fwd_s:.3e}")
    print(f"    quant error:  {err_s:.4f}")
    print(f"    F in top 4 blocks: {top_F_s:.3f}")
    print(f"    bits: {bits_s.tolist()}")

    print(f"\n  --- summary ---")
    print(f"    quant error: composite(1)={err_c:.4f}  composite(10)={err_cF:.4f}  spectral={err_s:.4f}")
    print(f"    F-in-top:   composite(1)={top_F_c:.3f}  composite(10)={top_F_cF:.3f}  spectral={top_F_s:.3f}")
    winner_err = min(
        ("composite(1)", err_c),
        ("composite(10)", err_cF),
        ("spectral", err_s),
        key=lambda x: x[1],
    )
    winner_F = max(
        ("composite(1)", top_F_c),
        ("composite(10)", top_F_cF),
        ("spectral", top_F_s),
        key=lambda x: x[1],
    )
    print(f"    -> lowest quant error: {winner_err[0]}")
    print(f"    -> most F in top blocks: {winner_F[0]}")


def test_spectral_robust_to_alpha_beta():
    """Spectral has no alpha/beta; composite is sensitive. Show that the
    best composite is still worse than spectral on disagreement data."""
    W_A, W_B, F, _, _ = setup_disagreement_data()
    X = torch.randn(2, 8, W_A.shape[1])

    # Sweep alpha, beta for composite
    best_err = float("inf")
    best_ab = None
    for a in [0.1, 0.3, 1.0, 3.0, 10.0]:
        for b in [0.1, 0.3, 1.0, 3.0, 10.0]:
            perm = find_permutation_composite(W_A, W_B, F, alpha=a, beta=b)
            err, _ = quantization_error_after_perm(W_A, W_B, F, perm.permutation)
            if err < best_err:
                best_err = err
                best_ab = (a, b)

    perm_s = find_permutation_spectral(W_A, W_B, F)
    err_s, _ = quantization_error_after_perm(W_A, W_B, F, perm_s.permutation)

    print(f"\n  best composite (alpha={best_ab[0]}, beta={best_ab[1]}): quant err = {best_err:.4f}")
    print(f"  spectral:                                         quant err = {err_s:.4f}")
    print(f"  -> spectral beats best composite: {err_s < best_err}")


def main() -> int:
    print("=" * 60)
    print("ICS: composite vs spectral A/B")
    print("=" * 60)
    tests = [
        ("spectral_basic_validity", test_spectral_basic_validity),
        ("spectral_vs_composite_on_disagreement", test_spectral_vs_composite_on_disagreement),
        ("spectral_robust_to_alpha_beta", test_spectral_robust_to_alpha_beta),
    ]
    failures = []
    for name, fn in tests:
        print(f"\n[test] {name}")
        try:
            fn()
            print(f"  PASS")
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failures.append(name)
    print()
    print("=" * 60)
    if failures:
        print(f"FAILED: {len(failures)}/{len(tests)} -> {failures}")
        return 1
    print(f"ALL {len(tests)} TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
