"""Tests for GQA (Grouped Query Attention) chain discovery.

Arbitrary channel permutations are not attention-head safe for GQA. Until the
pipeline has a head-group-preserving permutation, GQA attention chains must not
be selected for ICS. MLP chains remain safe because their intermediate
dimension is consumed by elementwise operations and a following linear layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from ics.permutation import find_permutation_composite
from ics.pipeline import (
    _derive_gqa_sub_perm,
    apply_chain,
    discover_chains,
)


class GQAAttention(nn.Module):
    """GQA: 8 Q heads, 2 KV heads, head_dim=4 => q(32) > k/v(8)."""

    def __init__(self):
        super().__init__()
        head_dim = 4
        n_q_heads = 8
        n_kv_heads = 2
        self.q_proj = nn.Linear(32, n_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(32, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(32, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_q_heads * head_dim, 32, bias=False)


class TinyMLP(nn.Module):
    def __init__(self, hidden=32, inter=64):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)


class GQABlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = GQAAttention()
        self.mlp = TinyMLP()


class GQAModel(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([GQABlock() for _ in range(n_layers)])


def _fwd_block(block: GQABlock, x: torch.Tensor) -> torch.Tensor:
    q = block.self_attn.q_proj(x)
    k = block.self_attn.k_proj(x)
    v = block.self_attn.v_proj(x)
    n_heads = 8
    head_dim = 4
    q = q.view(1, 4, n_heads, head_dim).transpose(1, 2)
    k = k.view(1, 4, 2, head_dim).transpose(1, 2)
    v = v.view(1, 4, 2, head_dim).transpose(1, 2)
    k = k.repeat_interleave(4, dim=1)
    v = v.repeat_interleave(4, dim=1)
    scores = q @ k.transpose(-1, -2) / (head_dim ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    attn = (weights @ v).transpose(1, 2).reshape(1, 4, 32)
    x = block.self_attn.o_proj(attn)
    gate = block.mlp.gate_proj(x)
    up = block.mlp.up_proj(x)
    hidden = torch.nn.functional.silu(gate) * up
    return block.mlp.down_proj(hidden)


def test_gqa_discover_chains_skips_attention_and_keeps_mlp():
    """GQA attention is not selected without a head-safe permutation."""
    model = GQAModel(n_layers=2)
    chains = discover_chains(model, skip_modules=("embed", "norm", "lm_head"))

    assert len(chains) == 2, f"expected 2 MLP chains, got {len(chains)}"
    print(f"  chains found: {len(chains)}")

    attn_chains = [c for c in chains if "self_attn" in c.members[0]]
    assert len(attn_chains) == 0, f"expected no GQA attention chains, got {len(attn_chains)}"

    mlp_chains = [c for c in chains if ".mlp." in c.members[0]]
    assert len(mlp_chains) == 2
    for chain in mlp_chains:
        assert chain.kind == "fan_out_chain"
        assert [m.split(".")[-1] for m in chain.members] == ["gate_proj", "up_proj", "down_proj"]
        assert chain.perm_targets == [0, 0, 1]
        assert chain.shared_dim_size == 64
        assert not chain.gqa_sub_perm_members

    print("  GQA attention skipped; MLP chains kept: PASS")


def test_derive_gqa_sub_perm_is_strict_permutation():
    """The legacy helper still returns a strict permutation."""
    P = torch.arange(32)
    sub = _derive_gqa_sub_perm(P, k_dim=8, gqa_ratio=4)
    assert sub.tolist() == list(range(8))
    print(f"  identity case: sub = {sub.tolist()}")

    P_swap = torch.tensor([
        4, 5, 6, 7, 0, 1, 2, 3,
        12, 13, 14, 15, 8, 9, 10, 11,
        20, 21, 22, 23, 16, 17, 18, 19,
        28, 29, 30, 31, 24, 25, 26, 27,
    ])
    sub_swap = _derive_gqa_sub_perm(P_swap, k_dim=8, gqa_ratio=4)
    assert sub_swap.tolist() == [1, 0, 3, 2, 5, 4, 7, 6]
    print(f"  swap-groups case: sub = {sub_swap.tolist()}")

    P_collide = torch.tensor([
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 1, 2, 3, 4, 5, 6, 7,
    ])
    sub_collide = _derive_gqa_sub_perm(P_collide, k_dim=8, gqa_ratio=4)
    assert sorted(sub_collide.tolist()) == list(range(8))
    print(f"  adversarial collision case: sub = {sub_collide.tolist()}")


def test_gqa_pipeline_applies_only_mlp_and_preserves_forward():
    """With GQA attention skipped, applying discovered chains stays exact."""
    torch.manual_seed(0)
    model = GQAModel(n_layers=1)
    chains = discover_chains(model, skip_modules=("embed", "norm", "lm_head"))
    assert all("self_attn" not in chain.members[0] for chain in chains)

    x = torch.randn(1, 4, 32)
    block = model.layers[0]
    out_orig = _fwd_block(block, x)

    fisher_norm = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            fisher_norm[name] = torch.rand(mod.out_features)

    for chain in chains:
        F = fisher_norm[chain.fisher_source]
        W_A_raw = model
        for part in chain.members[0].split("."):
            W_A_raw = getattr(W_A_raw, part)
        W_B_raw = model
        for part in chain.members[-1].split("."):
            W_B_raw = getattr(W_B_raw, part)
        W_A = W_A_raw.weight.detach().float()
        W_B = W_B_raw.weight.detach().float()
        if W_A.shape[0] > chain.shared_dim_size:
            W_A = W_A[:chain.shared_dim_size]
        if W_B.shape[1] > chain.shared_dim_size:
            W_B = W_B[:, :chain.shared_dim_size]
        perm = find_permutation_composite(W_A, W_B, F)
        apply_chain(model, chain, perm, F)

    out_perm = _fwd_block(block, x)
    assert torch.isfinite(out_perm).all(), "output contains non-finite values"
    diff = (out_perm - out_orig).abs().max().item()
    assert diff < 1e-6, f"MLP-only GQA block forward diverged by {diff:.3e}"
    print(f"  GQA block with MLP-only ICS forward max abs diff: {diff:.3e}")


def main() -> int:
    print("=" * 60)
    print("GQA ICS pipeline tests (attention skipped)")
    print("=" * 60)
    tests = [
        ("GQA discover_chains skips attention", test_gqa_discover_chains_skips_attention_and_keeps_mlp),
        ("_derive_gqa_sub_perm is strict", test_derive_gqa_sub_perm_is_strict_permutation),
        ("GQA MLP-only apply_chain preserves forward", test_gqa_pipeline_applies_only_mlp_and_preserves_forward),
    ]
    failures = []
    for name, fn in tests:
        print(f"\n[test] {name}")
        try:
            fn()
            print("  PASS")
        except Exception as exc:
            print(f"  FAIL: {exc}")
            import traceback
            traceback.print_exc()
            failures.append(name)
    print()
    if failures:
        print(f"FAILED: {len(failures)}/{len(tests)} -> {failures}")
        return 1
    print(f"ALL {len(tests)} TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
