"""GPTQ roundtrip tests.

These tests verify that `gptq_quantize` (in `ics/gptq.py`) actually does
what GPTQ should do: produce a quantized weight that, when dequantized,
beats plain Round-To-Nearest (RTN) on the *weighted* reconstruction loss
||(W - W_dq) X^T||_F^2, where H = X^T X is the input Hessian.

GPTQ's promise is NOT "lower unweighted l2_rel than RTN" -- on a normal
distribution with no Hessian structure, both methods produce similar
quantization noise. GPTQ's promise IS "lower weighted reconstruction
loss when H has structure" -- i.e., when there is a meaningful
calibration signal.

History (2026-07-02): the previous version of this test asserted
l2_rel < 0.10, which was the wrong contract -- INT4 on randn * 0.02
naturally produces l2_rel ~0.18 because of quantization noise, not
because of a bug. The bug was that GPTQ was not permuting H alongside
W (missing `H = H[perm][:, perm]` from the reference IST-DASLab/gptq).
The fix landed in ics/gptq.py; this test now verifies the corrected
behavior.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import ics.export as export_module
from ics.export import dequantized_state_dict, load_ics_model, save_ics_model
from ics.gptq import gptq_quantize, dequantize_gptq
from ics.pipeline import ICSConfig, ICSResult
from ics.quantize import QuantizedTensor, quantize_blockwise, dequantize_blockwise


def _l2_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).norm() / b.norm()


def _weighted_recon_loss(W: torch.Tensor, W_dq: torch.Tensor, H: torch.Tensor) -> float:
    """||(W - W_dq) X^T||_F^2 where H = X^T X. Equivalently: trace(E^T E H)."""
    E = W - W_dq
    return (E @ H).pow(2).sum().item()


def _make_rtn_baseline(W: torch.Tensor, group_size: int = 128, bits: int = 4) -> torch.Tensor:
    """Plain RTN with per-group INT4, no GPTQ error propagation."""
    in_features = W.shape[1]
    n_groups = in_features // group_size
    bit_tensor = torch.full((n_groups,), bits, dtype=torch.int32)
    qt = quantize_blockwise(W, bit_tensor, block_size=group_size, dim=-1)
    return dequantize_blockwise(qt)


def _make_structured_h(in_features: int, n_salient: int = 128, salient_scale: float = 5.0, seed: int = 0) -> torch.Tensor:
    """Build a Hessian H = X^T X / n where X has some 'salient' input dimensions."""
    g = torch.Generator().manual_seed(seed)
    n_samples = 64
    X = torch.randn(n_samples, in_features, generator=g)
    salient_dims = torch.randperm(in_features, generator=g)[:n_salient]
    X[:, salient_dims] *= salient_scale
    return X.T @ X / n_samples


def _pack_gptq_block_major(Q: torch.Tensor, group_size: int) -> torch.Tensor:
    pieces = []
    for start in range(0, Q.shape[1], group_size):
        pieces.append(Q[:, start:start + group_size].contiguous().reshape(-1))
    return torch.cat(pieces).to(torch.int8)


def test_gptq_beats_rtn_on_weighted_loss_with_structured_h():
    """GPTQ must beat RTN on the actual GPTQ objective when H has structure.

    The unweighted l2_rel can be slightly worse (GPTQ trades off non-salient
    error for salient error), but the weighted loss must be lower.
    """
    torch.manual_seed(0)
    out_features, in_features = 1024, 1024
    W = torch.randn(out_features, in_features) * 0.02
    H = _make_structured_h(in_features, n_salient=128, salient_scale=5.0)

    W_dq_rtn = _make_rtn_baseline(W)
    Q, scales, zeros, perm = gptq_quantize(W.clone(), H.clone(), bits=4, group_size=128, blocksize=128)
    W_dq_gptq = dequantize_gptq(Q, scales, zeros, perm, group_size=128)

    err_rtn = _l2_rel(W_dq_rtn, W)
    err_gptq = _l2_rel(W_dq_gptq, W)
    loss_rtn = _weighted_recon_loss(W, W_dq_rtn, H)
    loss_gptq = _weighted_recon_loss(W, W_dq_gptq, H)

    print(
        f"  l2_rel:  RTN={err_rtn:.4f}  GPTQ={err_gptq:.4f}  "
        f"(GPTQ may be higher -- it optimizes weighted loss, not unweighted)\n"
        f"  recon:   RTN={loss_rtn:.2f}  GPTQ={loss_gptq:.2f}  "
        f"ratio GPTQ/RTN = {loss_gptq/loss_rtn:.3f}"
    )
    assert loss_gptq < loss_rtn, (
        f"GPTQ weighted recon loss {loss_gptq:.2f} >= RTN {loss_rtn:.2f}; "
        f"GPTQ is failing to use H information."
    )
    # Generous bound: GPTQ should be at least 2x better when H has salient structure
    assert loss_gptq < 0.5 * loss_rtn, (
        f"GPTQ improvement too small: {loss_gptq:.2f} vs RTN {loss_rtn:.2f} "
        f"(ratio {loss_gptq/loss_rtn:.3f}); expected at least 2x better."
    )


def test_gptq_matches_rtn_with_identity_h():
    """With H=I (no structure), GPTQ should reduce to RTN-like behavior.

    H_inv=I means H_inv_chol=I, so the error propagation `W1[:, j:] -=
    err * Hinv1[j, j:]` is multiplying by 0 -- no propagation happens.
    GPTQ output should match RTN exactly.
    """
    torch.manual_seed(1)
    out_features, in_features = 1024, 1024
    W = torch.randn(out_features, in_features) * 0.02
    H = torch.eye(in_features)

    W_dq_rtn = _make_rtn_baseline(W)
    Q, scales, zeros, perm = gptq_quantize(W.clone(), H.clone(), bits=4, group_size=128, blocksize=128)
    W_dq_gptq = dequantize_gptq(Q, scales, zeros, perm, group_size=128)

    err_rtn = _l2_rel(W_dq_rtn, W)
    err_gptq = _l2_rel(W_dq_gptq, W)
    diff = (W_dq_gptq - W_dq_rtn).abs().max().item()
    print(f"  l2_rel:  RTN={err_rtn:.4f}  GPTQ={err_gptq:.4f}  max_diff={diff:.6f}")
    # For H=I, GPTQ should be exactly RTN (no propagation). Allow tiny FP noise.
    assert diff < 1e-3, f"H=I: GPTQ should reduce to RTN, max diff {diff}"


def test_gptq_perm_is_valid():
    """The perm output by gptq_quantize is a valid column permutation."""
    torch.manual_seed(2)
    W = torch.randn(64, 256) * 0.1
    H = torch.eye(256)
    _, _, _, perm = gptq_quantize(W.clone(), H.clone(), bits=4, group_size=128)
    assert perm.shape == (256,), f"perm shape {perm.shape} != (256,)"
    assert set(perm.tolist()) == set(range(256)), (
        "perm has duplicates or missing indices"
    )
    print(f"  perm: valid 256-element permutation, first 8 entries = {perm[:8].tolist()}")


def test_gptq_perm_sorts_by_descending_h_diag():
    """The perm should sort columns by descending H diagonal (actorder).

    For a diagonal H with entries [0.1, 0.5, 0.3, 0.9, 0.7], the perm
    should be [3, 4, 1, 2, 0] (indices in original order, sorted by
    H_diag descending).
    """
    torch.manual_seed(3)
    in_features = 5
    H = torch.diag(torch.tensor([0.1, 0.5, 0.3, 0.9, 0.7]))
    W = torch.randn(8, in_features) * 0.1
    _, _, _, perm = gptq_quantize(W.clone(), H.clone(), bits=4, group_size=in_features)
    # perm[0] should be the index of the largest H diag (3 = 0.9)
    # perm[-1] should be the index of the smallest H diag (0 = 0.1)
    assert perm[0].item() == 3, f"perm[0]={perm[0]}, expected 3 (largest H diag 0.9)"
    assert perm[-1].item() == 0, f"perm[-1]={perm[-1]}, expected 0 (smallest H diag 0.1)"
    print(f"  perm for diagonal H [0.1,0.5,0.3,0.9,0.7]: {perm.tolist()} (correct order)")


def test_gptq_scales_zeros_shape():
    """scales and zeros have the right shape (one per group)."""
    torch.manual_seed(4)
    in_features = 512
    W = torch.randn(32, in_features) * 0.1
    H = torch.eye(in_features)
    Q, scales, zeros, perm = gptq_quantize(
        W.clone(), H.clone(), bits=4, group_size=128
    )
    expected_groups = in_features // 128  # = 4
    assert scales.shape == (expected_groups,), (
        f"scales shape {scales.shape} != ({expected_groups},)"
    )
    assert zeros.shape == (expected_groups,), (
        f"zeros shape {zeros.shape} != ({expected_groups},)"
    )
    assert (zeros == 0).all(), "symmetric INT4 should have all-zero zero-points"
    assert (scales > 0).all(), "scales should all be positive"
    print(
        f"  scales shape {tuple(scales.shape)}, all positive, range "
        f"[{scales.min().item():.5f}, {scales.max().item():.5f}]"
    )


def test_gptq_recovery_via_perm():
    """dequantize_gptq output reconstructs W in original column order."""
    torch.manual_seed(5)
    W = torch.randn(128, 256) * 0.05
    H = torch.eye(256)

    Q, scales, zeros, perm = gptq_quantize(W.clone(), H.clone(), bits=4, group_size=128)
    W_dq = dequantize_gptq(Q, scales, zeros, perm, group_size=128)

    # If dequant is correct, W_dq should match W (modulo quantization noise).
    err = _l2_rel(W_dq, W).item()
    print(f"  W_dq vs W: l2_rel = {err:.4f}")
    # Same floor as RTN on this data: ~0.18.
    assert err < 0.30, f"recovery failed: l2_rel {err:.4f} > 0.30 (quantization broken?)"


def test_saved_gptq_tensor_dequantizes_from_block_layout():
    """Saved GPTQ qdata uses block-row-major layout and dequantizes via export."""
    group_size = 3
    Q = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6],
            [-1, -2, -3, -4, -5, -6],
        ],
        dtype=torch.int8,
    )
    qdata = _pack_gptq_block_major(Q, group_size)
    scales = torch.tensor([0.5, 0.25], dtype=torch.float32)
    zeros = torch.zeros(2, dtype=torch.int32)
    bits = torch.full((2,), 4, dtype=torch.int32)
    qt = QuantizedTensor(
        qdata=qdata,
        scales=scales,
        zeros=zeros,
        bits=bits,
        block_size=group_size,
        original_shape=tuple(Q.shape),
        quant_dim=1,
        method="gptq_per_group",
    )
    gptq_perm = torch.tensor([2, 0, 5, 1, 4, 3], dtype=torch.long)
    result = ICSResult(
        perms={},
        quant={"layer": qt},
        bit_widths={"layer": bits},
        fisher={},
        config=ICSConfig(quant_method="gptq", gptq_group_size=group_size),
        layer_perms={"layer": gptq_perm},
    )

    with tempfile.TemporaryDirectory() as tmp:
        out = save_ics_model(result, tmp)
        loaded = load_ics_model(out)
        state = dequantized_state_dict(loaded)

    raw_expected = torch.zeros_like(Q, dtype=torch.float32)
    raw_expected[:, :group_size] = Q[:, :group_size].float() * scales[0]
    raw_expected[:, group_size:] = Q[:, group_size:].float() * scales[1]
    expected = torch.zeros_like(raw_expected)
    expected[:, gptq_perm] = raw_expected

    assert loaded["layers"]["layer"].qdata.tolist() == qdata.tolist(), (
        "saved GPTQ qdata should preserve the block-row-major buffer layout"
    )
    assert torch.allclose(state["layer.weight"], expected), (
        "saved GPTQ tensor did not dequantize through the export path"
    )
    assert not hasattr(export_module, "dequantize_gptq_per_group"), (
        "stale row-major-only GPTQ dequant helper should not be exported"
    )


def main() -> int:
    print("=" * 60)
    print("GPTQ roundtrip tests (ics/gptq.py)")
    print("=" * 60)
    failures = []
    tests = [
        ("gptq_beats_rtn_on_weighted_loss", test_gptq_beats_rtn_on_weighted_loss_with_structured_h),
        ("gptq_matches_rtn_with_identity_h", test_gptq_matches_rtn_with_identity_h),
        ("gptq_perm_is_valid", test_gptq_perm_is_valid),
        ("gptq_perm_sorts_by_descending_h_diag", test_gptq_perm_sorts_by_descending_h_diag),
        ("gptq_scales_zeros_shape", test_gptq_scales_zeros_shape),
        ("gptq_recovery_via_perm", test_gptq_recovery_via_perm),
        ("saved_gptq_tensor_dequantizes_from_block_layout", test_saved_gptq_tensor_dequantizes_from_block_layout),
    ]
    for name, fn in tests:
        print(f"\n[test] {name}")
        try:
            fn()
            print(f"  PASS")
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")
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
