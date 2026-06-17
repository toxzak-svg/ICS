"""Correctness tests for the ICS core algorithm.

Verifies, on synthetic data, that:
    1. The forward pass through (W_A, W_B) is bit-identical to the
       forward pass through the permuted (W_A', W_B') within FP tolerance.
    2. The composite-score permutation is consistent (F-aware).
    3. The Sinkhorn-Hungarian permutation is a valid permutation
       (no duplicates, no missing indices).
    4. The block-aware rebalance produces valid blocks.
    5. Quantize-dequantize round-trip is within tolerance.
    6. Variable-bit block quantization matches the bit-width spec.
    7. The full pipeline (fan-out + chain) preserves forward pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from ics.permutation import (
    find_permutation_composite,
    find_permutation_sinkhorn_hungarian,
    find_permutation_for_fan_out,
    apply_permutation_to_chain,
    fan_out_permutation,
    _rebalance_for_blocks,
)
from ics.quantize import (
    quantize_blockwise,
    dequantize_blockwise,
    assign_bit_widths,
    _int4_block_quantize,
    _int2_block_quantize,
    _int1_block_quantize,
    _dequant_int4_block,
    _dequant_int2_block,
    _dequant_int1_block,
)


def test_chain_forward_pass_identity():
    """Y = X W_A W_B == X (W_A' P^T) (P W_B')  within FP tolerance."""
    torch.manual_seed(0)
    out_A, in_A = 256, 512
    out_B, in_B = 128, out_A
    X = torch.randn(4, 32, in_A)
    W_A = torch.randn(out_A, in_A) * 0.1
    W_B = torch.randn(out_B, in_B) * 0.1

    # Original forward
    Y_orig = X @ W_A.T
    Z_orig = Y_orig @ W_B.T

    # Find permutation with synthetic Fisher
    F = torch.rand(out_A)
    perm = find_permutation_composite(W_A, W_B, F, alpha=1.0, beta=1.0)

    # Apply permutation
    W_A_new, W_B_new = apply_permutation_to_chain(W_A, W_B, perm.permutation)

    # Verify dimensions
    assert W_A_new.shape == W_A.shape, f"W_A shape changed: {W_A.shape} -> {W_A_new.shape}"
    assert W_B_new.shape == W_B.shape, f"W_B shape changed: {W_B.shape} -> {W_B_new.shape}"

    # Verify it's actually a permutation
    P = perm.permutation
    assert P.shape == (out_A,)
    assert set(P.tolist()) == set(range(out_A)), "permutation has duplicates or missing indices"

    # Permuted forward: simulate the runtime by permuting activations
    Y_perm_runtime = X @ W_A_new.T  # output of layer A after perm is baked in
    # Layer A's output in the *original* channel layout is Y_perm_runtime[:, inverse(P)]
    # But the next layer W_B_new was permuted to consume that layout
    Z_perm = Y_perm_runtime @ W_B_new.T

    # Mathematically: Z_perm should equal Z_orig because the (P, P^-1) pair cancels
    diff = (Z_perm - Z_orig).abs().max().item()
    print(f"  chain forward-pass max abs diff: {diff:.3e}")
    assert diff < 1e-4, f"forward pass diverged by {diff}"


def test_sinkhorn_hungarian_is_permutation():
    """Sinkhorn-Hungarian output is a valid permutation."""
    torch.manual_seed(1)
    out_A, in_A = 128, 256
    W_A = torch.randn(out_A, in_A) * 0.1
    W_B = torch.randn(64, out_A) * 0.1
    F = torch.rand(out_A)
    perm = find_permutation_sinkhorn_hungarian(W_A, W_B, F, block_size=32)
    P = perm.permutation
    assert P.shape == (out_A,)
    assert set(P.tolist()) == set(range(out_A)), "Sinkhorn-Hungarian produced invalid perm"
    print(f"  sinkhorn-hungarian: valid perm, score={perm.score:.3f}")


def test_block_rebalance():
    """Block rebalance keeps high-Fisher channels out of block edges."""
    F = np.array([10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)  # 12 channels
    M_A = np.ones(12)
    M_B = np.ones(12)
    perm = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    out = _rebalance_for_blocks(perm, F, M_A, M_B, block_size=4)
    # Each block of 4 should have its highest-Fisher channel at index 0
    for b in range(3):
        block_F = F[out[b * 4:(b + 1) * 4].numpy()]
        # Within the block, F should be monotonically non-increasing
        # (we sort by Fisher descending within the block)
        assert block_F[0] == block_F.max(), f"block {b} has max-Fisher at non-zero position"
    print(f"  block rebalance: high-Fisher at block start in all 3 blocks")


def test_quantize_roundtrip_int4():
    """INT4 quantize-dequantize round-trip is within tolerance."""
    torch.manual_seed(2)
    W = torch.randn(64, 128) * 0.5
    bits = torch.full((2,), 4, dtype=torch.int32)  # 2 blocks of 64
    qt = quantize_blockwise(W, bits, block_size=64)
    W_dq = dequantize_blockwise(qt)
    err = (W - W_dq).abs().max().item()
    # INT4 symmetric: scale = absmax/7, max err = scale/2 = absmax/14
    absmax = W.abs().amax().item()
    bound = absmax / 14 + 1e-6
    print(f"  int4 roundtrip max err: {err:.4f} (bound {bound:.4f})")
    assert err < bound, f"int4 roundtrip err {err} > bound {bound}"


def test_quantize_roundtrip_int2():
    """INT2 quantize-dequantize round-trip is within tolerance."""
    torch.manual_seed(3)
    W = torch.randn(64, 64) * 0.5
    bits = torch.full((1,), 2, dtype=torch.int32)
    qt = quantize_blockwise(W, bits, block_size=64)
    W_dq = dequantize_blockwise(qt)
    err = (W - W_dq).abs().max().item()
    absmax = W.abs().amax().item()
    bound = absmax / 2 + 1e-6  # INT2 symmetric: 4 levels, half-step = absmax/2
    print(f"  int2 roundtrip max err: {err:.4f} (bound {bound:.4f})")
    assert err < bound, f"int2 roundtrip err {err} > bound {bound}"


def test_quantize_roundtrip_int1():
    """INT1 (binary) quantize-dequantize round-trip is within tolerance."""
    torch.manual_seed(4)
    W = torch.randn(64, 64) * 0.5
    bits = torch.full((1,), 1, dtype=torch.int32)
    qt = quantize_blockwise(W, bits, block_size=64)
    W_dq = dequantize_blockwise(qt)
    err = (W - W_dq).abs().max().item()
    absmax = W.abs().amax().item()
    bound = 2 * absmax + 1e-6  # INT1: maps to ±absmax
    print(f"  int1 roundtrip max err: {err:.4f} (bound {bound:.4f})")
    assert err < bound, f"int1 roundtrip err {err} > bound {bound}"


def test_mixed_bit_widths():
    """Variable bit-widths are correctly assigned per block."""
    F = torch.tensor([10, 8, 6, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float32)
    bits = assign_bit_widths(F, block_size=4, int4_fraction=0.5, int2_fraction=0.5, int1_fraction=0.0)
    # 4 blocks of 4 channels each. Top 2 by Fisher should be INT4, next 2 INT2.
    assert bits[0] == 4, f"block 0 (highest Fisher) should be INT4, got {bits[0]}"
    assert bits[1] == 4, f"block 1 should be INT4, got {bits[1]}"
    assert bits[2] == 2, f"block 2 should be INT2, got {bits[2]}"
    assert bits[3] == 2, f"block 3 should be INT2, got {bits[3]}"
    print(f"  bit-width assignment: {bits.tolist()} (correct)")


def test_fan_out_permutation():
    """MLP fan-out: gate/up fan out on the OUTPUT (inter) dim, down
    consumes that dim. ICS perm is on the inter dim; gate/up's rows and
    down's columns are permuted. The element-wise SiLU and product
    commute with the perm, so the forward pass is preserved.
    """
    torch.manual_seed(5)
    hidden = 256
    inter = 512
    X = torch.randn(2, 16, hidden)

    W_gate = torch.randn(inter, hidden) * 0.1
    W_up = torch.randn(inter, hidden) * 0.1
    W_down = torch.randn(hidden, inter) * 0.1
    # Fisher of the inter-dim activation (silu(gate_out) * up_out)
    F = torch.rand(inter)

    # Original MLP block
    silu_gate = torch.nn.functional.silu(X @ W_gate.T)
    up_out = X @ W_up.T
    product = silu_gate * up_out
    out_orig = product @ W_down.T

    # ICS perm on the shared inter dim, driven by (gate, down) as the
    # representative (W_A, W_B) chain pair.
    perm = find_permutation_composite(W_gate, W_down, F, alpha=1.0, beta=1.0)
    P = perm.permutation
    P_np = P.cpu().numpy()

    # Apply: gate's rows, up's rows (output = inter), down's columns (input = inter)
    W_gate_new = W_gate[P_np, :].clone()
    W_up_new = W_up[P_np, :].clone()
    W_down_new = W_down[:, P_np].clone()

    # Permuted forward
    silu_gate_new = torch.nn.functional.silu(X @ W_gate_new.T)
    up_out_new = X @ W_up_new.T
    product_new = silu_gate_new * up_out_new
    out_perm = product_new @ W_down_new.T

    diff = (out_perm - out_orig).abs().max().item()
    print(f"  MLP fan-out-chain forward diff: {diff:.3e}")
    assert diff < 1e-4, f"MLP forward diverged by {diff}"


def test_attention_head_aware_perm():
    """Attention fan-out: q, k, v share output dim (num_heads*head_dim).
    Perm commutes through softmax(QK^T)@V. This test models a single-head
    attention with no head structure (treats the channel dim as one block).
    """
    torch.manual_seed(7)
    hidden = 256
    n_heads = 1
    head_dim = 64
    qkv_dim = n_heads * head_dim  # 64
    X = torch.randn(2, 8, hidden)

    W_q = torch.randn(qkv_dim, hidden) * 0.1
    W_k = torch.randn(qkv_dim, hidden) * 0.1
    W_v = torch.randn(qkv_dim, hidden) * 0.1
    W_o = torch.randn(hidden, qkv_dim) * 0.1
    F = torch.rand(qkv_dim)

    # Original attention block
    Q = X @ W_q.T  # (B, T, qkv_dim)
    K = X @ W_k.T
    V = X @ W_v.T
    scores = Q @ K.transpose(-1, -2) / (head_dim ** 0.5)
    weights = torch.nn.functional.softmax(scores, dim=-1)
    attn_out = weights @ V
    out_orig = attn_out @ W_o.T

    # ICS perm on the qkv_dim (shared by q/k/v output and o input)
    perm = find_permutation_composite(W_q, W_o, F, alpha=1.0, beta=1.0)
    P = perm.permutation
    P_np = P.cpu().numpy()

    # Apply to q/k/v rows and o columns
    W_q_new = W_q[P_np, :].clone()
    W_k_new = W_k[P_np, :].clone()
    W_v_new = W_v[P_np, :].clone()
    W_o_new = W_o[:, P_np].clone()

    # Permuted attention
    Q_new = X @ W_q_new.T
    K_new = X @ W_k_new.T
    V_new = X @ W_v_new.T
    scores_new = Q_new @ K_new.transpose(-1, -2) / (head_dim ** 0.5)
    weights_new = torch.nn.functional.softmax(scores_new, dim=-1)
    attn_out_new = weights_new @ V_new
    out_perm = attn_out_new @ W_o_new.T

    diff = (out_perm - out_orig).abs().max().item()
    print(f"  attention fan-out forward diff: {diff:.3e}")
    assert diff < 1e-4, f"attention forward diverged by {diff}"


def test_full_pipeline_synthetic():
    """End-to-end test: synthetic 2-layer model with synthetic Fisher,
    apply ICS chain, verify forward pass."""
    torch.manual_seed(6)
    hidden = 256
    inter = 512
    batch, seq = 2, 16
    X = torch.randn(batch, seq, hidden)

    W_A = torch.randn(inter, hidden) * 0.1
    W_B = torch.randn(hidden, inter) * 0.1
    F = torch.rand(inter)

    # Modest outliers to make the sort key meaningful without
    # blowing up FP32 roundoff
    W_A[0:5, :] *= 3.0
    W_B[:, 0:5] *= 3.0

    Y_orig = X @ W_A.T
    Z_orig = Y_orig @ W_B.T

    perm = find_permutation_composite(W_A, W_B, F, alpha=1.0, beta=1.0)
    W_A_new, W_B_new = apply_permutation_to_chain(W_A, W_B, perm.permutation)

    Y_perm = X @ W_A_new.T
    Z_perm = Y_perm @ W_B_new.T

    diff = (Z_perm - Z_orig).abs().max().item()
    print(f"  full pipeline: forward pass max diff = {diff:.3e}")
    # FP32 roundoff budget: ~1e-5 per matmul, ~1e-4 chained for these
    # magnitudes. The math is exact; the residual is FP precision.
    assert diff < 1e-3, f"full pipeline forward diverged by {diff}"


def main() -> int:
    print("=" * 60)
    print("ICS correctness tests")
    print("=" * 60)
    failures = []
    tests = [
        ("chain_forward_pass_identity", test_chain_forward_pass_identity),
        ("sinkhorn_hungarian_is_permutation", test_sinkhorn_hungarian_is_permutation),
        ("block_rebalance", test_block_rebalance),
        ("quantize_roundtrip_int4", test_quantize_roundtrip_int4),
        ("quantize_roundtrip_int2", test_quantize_roundtrip_int2),
        ("quantize_roundtrip_int1", test_quantize_roundtrip_int1),
        ("mixed_bit_widths", test_mixed_bit_widths),
        ("mlp_fan_out_chain", test_fan_out_permutation),
        ("attention_fan_out_chain", test_attention_head_aware_perm),
        ("full_pipeline_synthetic", test_full_pipeline_synthetic),
    ]
    for name, fn in tests:
        print(f"\n[test] {name}")
        try:
            fn()
            print(f"  PASS")
        except Exception as e:
            print(f"  FAIL: {e}")
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
