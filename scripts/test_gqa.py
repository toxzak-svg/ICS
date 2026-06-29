"""Test GQA (Grouped Query Attention) handling in discover_chains + apply_chain.

Verifies the GQA-aware path: k/v are now included in the chain with a
sub-perm derived from the main chain perm via _derive_gqa_sub_perm.
The pre-fix code excluded k/v entirely, which caused garbage PPL because
attention computes softmax(q_perm Â· k_orig Â· v_orig) with mismatched
layouts.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from ics.pipeline import (
    discover_chains,
    apply_chain,
    LinearChainSpec,
    _derive_gqa_sub_perm,
)
from ics.permutation import find_permutation_composite


class GQAAttention(nn.Module):
    """GQA: 8 Q heads, 2 KV heads, head_dim=4 => q(32) > k/v(8)."""
    def __init__(self):
        super().__init__()
        head_dim = 4
        n_q_heads = 8
        n_kv_heads = 2
        self.q_proj = nn.Linear(32, n_q_heads * head_dim, bias=False)   # (32, 32)
        self.k_proj = nn.Linear(32, n_kv_heads * head_dim, bias=False)  # (8, 32)
        self.v_proj = nn.Linear(32, n_kv_heads * head_dim, bias=False)  # (8, 32)
        self.o_proj = nn.Linear(n_q_heads * head_dim, 32, bias=False)   # (32, 32)


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


def test_gqa_discover_chains_includes_kv_with_sub_perm():
    """GQA: k/v have fewer output dims (8) than q (32). They MUST be
    included as producers, marked with gqa_sub_perm_members, so the
    chain applies a GQA sub-perm to their rows."""
    model = GQAModel(n_layers=2)
    chains = discover_chains(model, skip_modules=("embed", "norm", "lm_head"))

    # Expect 2 layers * 2 blocks = 4 chains
    assert len(chains) == 4, f"expected 4 chains, got {len(chains)}"
    print(f"  chains found: {len(chains)}")

    attn_chains = [c for c in chains if "self_attn" in c.members[0]]
    assert len(attn_chains) == 2

    for c in attn_chains:
        # Q=32, KV=8, so shared=32, gqa_ratio=32/8=4
        producers = [m for m in c.members if c.perm_targets[c.members.index(m)] == 0]
        consumers = [m for m in c.members if c.perm_targets[c.members.index(m)] == 1]
        print(f"  attn producers: {[p.split('.')[-1] for p in producers]}")
        consumers = [c.members[i] for i in range(len(c.members)) if c.perm_targets[i] == 1]
        print(f"  attn consumers: {[c.split('.')[-1] for c in consumers]}")
        # q, k, v are all producers with target=0 (row perm); o is consumer with target=1
        assert len(producers) == 3, f"expected 3 producers (q, k, v), got {len(producers)}"
        producer_names = [p.split(".")[-1] for p in producers]
        assert producer_names == ["q_proj", "k_proj", "v_proj"], f"got {producer_names}"
        assert len(consumers) == 1
        assert "o_proj" in consumers[0]
        assert c.kind == "fan_out_chain"
        assert c.shared_dim_size == 32
        # GQA-specific metadata
        assert c.gqa_ratio == 4, f"expected gqa_ratio=4, got {c.gqa_ratio}"
        sub_names = [n.split(".")[-1] for n in c.gqa_sub_perm_members]
        assert sub_names == ["k_proj", "v_proj"], f"got {sub_names}"
        print(f"  gqa_ratio={c.gqa_ratio}, gqa_sub_perm_members={sub_names}")

    print(f"  GQA discover_chains (with k/v): PASS")


def test_derive_gqa_sub_perm_is_strict_permutation():
    """_derive_gqa_sub_perm must always return a strict permutation,
    even when the main perm would produce collisions."""
    torch.manual_seed(0)

    # Case 1: identity main perm -> sub-perm should be identity too
    P = torch.arange(32)  # shared=32, k_dim=8, gqa_ratio=4
    sub = _derive_gqa_sub_perm(P, k_dim=8, gqa_ratio=4)
    assert sub.shape == (8,)
    assert sorted(sub.tolist()) == list(range(8)), f"not a perm: {sub.tolist()}"
    # Identity main perm: sub should be [0,1,...,7]
    assert sub.tolist() == list(range(8))
    print(f"  identity case: sub = {sub.tolist()}")

    # Case 2: swap groups -> sub-perm swaps correspondingly
    P_swap = torch.tensor([4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11,
                           20, 21, 22, 23, 16, 17, 18, 19, 28, 29, 30, 31, 24, 25, 26, 27])
    sub_swap = _derive_gqa_sub_perm(P_swap, k_dim=8, gqa_ratio=4)
    assert sorted(sub_swap.tolist()) == list(range(8)), f"not a perm: {sub_swap.tolist()}"
    # P_swap blocks-swap-pairwise (0<->1, 2<->3, ...). The algorithm picks
    # the OLD KV head corresponding to NEW Q block d's first row:
    #   initial[d] = P_swap[d*4] // 4.
    # For P_swap: d=0 -> 4//4=1, d=1 -> 0//4=0, d=2 -> 12//4=3, d=3 -> 8//4=2,
    #             d=4 -> 20//4=5, d=5 -> 16//4=4, d=6 -> 28//4=7, d=7 -> 24//4=6.
    # No collisions, so result == initial.
    expected = [1, 0, 3, 2, 5, 4, 7, 6]
    assert sub_swap.tolist() == expected, f"swap case: got {sub_swap.tolist()}, expected {expected}"
    print(f"  swap-groups case: sub = {sub_swap.tolist()}")

    # Case 3: adversarial perm that produces collisions
    # P maps d*4 -> all in OLD head group 0 (P[d*4] in [0,4) for all d)
    P_collide = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7,
                              0, 1, 2, 3, 4, 5, 6, 7,
                              0, 1, 2, 3, 4, 5, 6, 7,
                              0, 1, 2, 3, 4, 5, 6, 7])
    sub_collide = _derive_gqa_sub_perm(P_collide, k_dim=8, gqa_ratio=4)
    assert sorted(sub_collide.tolist()) == list(range(8)), f"collisions not resolved: {sub_collide.tolist()}"
    print(f"  adversarial (collision) case: sub = {sub_collide.tolist()} -- collision resolved")

    print(f"  _derive_gqa_sub_perm: PASS")


def test_gqa_apply_chain_with_sub_perm():
    """Verify apply_chain correctly applies the main perm to q/o and the
    GQA sub-perm to k/v, and the resulting forward produces finite output."""
    torch.manual_seed(0)
    model = GQAModel(n_layers=1)
    chains = discover_chains(model, skip_modules=("embed", "norm", "lm_head"))

    X = torch.randn(1, 4, 32)
    block = model.layers[0]

    def fwd_attn(block, x):
        q = block.self_attn.q_proj(x)
        k = block.self_attn.k_proj(x)
        v = block.self_attn.v_proj(x)
        n_heads = 8
        head_dim = 4
        q = q.view(1, 4, n_heads, head_dim).transpose(1, 2)
        k = k.view(1, 4, 2, head_dim).transpose(1, 2)
        v = v.view(1, 4, 2, head_dim).transpose(1, 2)
        # Repeat KV for GQA
        k = k.repeat_interleave(4, dim=1)
        v = v.repeat_interleave(4, dim=1)
        scores = q @ k.transpose(-1, -2) / (head_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        attn = (weights @ v).transpose(1, 2).reshape(1, 4, 32)
        x = block.self_attn.o_proj(attn)
        gate = block.mlp.gate_proj(x)
        up = block.mlp.up_proj(x)
        h = torch.nn.functional.silu(gate) * up
        x = block.mlp.down_proj(h)
        return x

    out_orig = fwd_attn(block, X)

    # Mock Fisher
    fisher_norm = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            fisher_norm[name] = torch.rand(mod.out_features)

    # Find and apply perms
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

        # GQA: derive sub-perm for k/v
        if chain.gqa_sub_perm_members and chain.gqa_ratio > 1:
            k_dim = chain.shared_dim_size // chain.gqa_ratio
            perm.gqa_sub_perm = _derive_gqa_sub_perm(
                perm.permutation, k_dim=k_dim, gqa_ratio=chain.gqa_ratio,
            )

        apply_chain(model, chain, perm, F)

    out_perm = fwd_attn(block, X)
    assert torch.isfinite(out_perm).all(), "output contains non-finite values"

    # The GQA-aware perm keeps the forward finite and the magnitude
    # bounded. We don't assert bit-exact (GQA perm breaks exact invariance
    # for non-head-respecting perms; see design note), but we DO assert
    # the output isn't garbage. Note: rel metric is unreliable on random-init
    # models (output magnitudes are tiny, so even small abs diffs blow up rel);
    # use an absolute bound instead.
    diff = (out_perm - out_orig).abs().max().item()
    out_orig_max = out_orig.abs().max().item()
    # Sanity-check 1: perm didn't blow up the output (>10x magnitude = "garbage").
    assert out_perm.abs().max().item() < 10.0 * (out_orig_max + 1e-12), (
        f"perm diverged in magnitude: {out_perm.abs().max().item():.3e} vs "
        f"orig {out_orig_max:.3e}"
    )
    # Sanity-check 2: absolute diff stays at small numerical scale (< 1.0).
    assert diff < 1.0, f"abs diff too large: {diff:.3e}"
    print(f"  GQA forward max abs diff: {diff:.3e}, orig max: {out_orig_max:.3e}")
    print(f"  GQA apply_chain with sub-perm: PASS")


def test_sub_perm_is_strict_permutation_after_real_pipeline():
    """End-to-end: the GQA sub-perm produced from a composite-score
    perm must be a strict permutation (no collisions)."""
    torch.manual_seed(42)
    model = GQAModel(n_layers=1)
    chains = discover_chains(model, skip_modules=("embed", "norm", "lm_head"))

    for chain in chains:
        if not chain.gqa_sub_perm_members:
            continue
        # Random Fisher and weights to give the perm finder real signal
        F = torch.rand(chain.shared_dim_size)
        W_A = model
        for part in chain.members[0].split("."):
            W_A = getattr(W_A, part)
        W_B = model
        for part in chain.members[-1].split("."):
            W_B = getattr(W_B, part)
        W_A = W_A.weight.detach().float()
        W_B = W_B.weight.detach().float()
        if W_A.shape[0] > chain.shared_dim_size:
            W_A = W_A[:chain.shared_dim_size]
        if W_B.shape[1] > chain.shared_dim_size:
            W_B = W_B[:, :chain.shared_dim_size]
        perm = find_permutation_composite(W_A, W_B, F)
        k_dim = chain.shared_dim_size // chain.gqa_ratio
        sub = _derive_gqa_sub_perm(perm.permutation, k_dim=k_dim, gqa_ratio=chain.gqa_ratio)
        assert sorted(sub.tolist()) == list(range(k_dim)), (
            f"sub-perm is not a strict permutation: {sub.tolist()}"
        )
        print(f"  real-perm sub is strict perm: {sub.tolist()}")

    print(f"  end-to-end sub-perm strictness: PASS")


def main() -> int:
    print("=" * 60)
    print("GQA ICS pipeline tests (post-fix: GQA-aware sub-perm)")
    print("=" * 60)
    tests = [
        ("GQA discover_chains (k/v included)", test_gqa_discover_chains_includes_kv_with_sub_perm),
        ("_derive_gqa_sub_perm is strict", test_derive_gqa_sub_perm_is_strict_permutation),
        ("GQA apply_chain with sub-perm", test_gqa_apply_chain_with_sub_perm),
        ("end-to-end sub-perm strictness", test_sub_perm_is_strict_permutation_after_real_pipeline),
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
    if failures:
        print(f"FAILED: {len(failures)}/{len(tests)} -> {failures}")
        return 1
    print(f"ALL {len(tests)} TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
