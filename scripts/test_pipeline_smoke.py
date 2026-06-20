"""Pipeline smoke test: build a tiny model with self_attn + mlp blocks,
mock the Fisher, and run discover_chains + apply_chain end-to-end.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from ics.pipeline import discover_chains, apply_chain, LinearChainSpec, select_chains
from ics.permutation import find_permutation_composite, PermutationResult
from ics.fisher import FisherStats, compute_fisher


class TinyAttention(nn.Module):
    def __init__(self, hidden=64, n_heads=4):
        super().__init__()
        head_dim = hidden // n_heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)


class TinyMLP(nn.Module):
    def __init__(self, hidden=64, inter=128):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention()
        self.mlp = TinyMLP()


class TinyModel(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([TinyBlock() for _ in range(n_layers)])


class TinyTokenizer:
    def __call__(self, text, return_tensors="pt", truncation=True, max_length=512):
        ids = [ord(ch) % 16 for ch in text][:max_length]
        if len(ids) < 2:
            ids = ids + [1] * (2 - len(ids))
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class EmptyTokenizer:
    def __call__(self, text, return_tensors="pt", truncation=True, max_length=512):
        input_ids = torch.empty((1, 0), dtype=torch.long)
        attention_mask = torch.empty((1, 0), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class TinyCausalLM(nn.Module):
    def __init__(self, hidden=8, vocab=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.lm_head = nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        x = self.embed(input_ids)
        x = torch.tanh(self.proj(x))
        logits = self.lm_head(x)
        return type("TinyOutput", (), {"logits": logits})


def test_discover_chains():
    """discover_chains finds the right chains with the right targets."""
    model = TinyModel(n_layers=2)
    chains = discover_chains(model, skip_modules=("embed", "norm", "lm_head"))

    # Expect 2 layers * 2 blocks (attn + mlp) = 4 chains
    assert len(chains) == 4, f"expected 4 chains, got {len(chains)}"

    attn_chains = [c for c in chains if "self_attn" in c.members[0]]
    mlp_chains = [c for c in chains if ".mlp." in c.members[0]]
    assert len(attn_chains) == 2
    assert len(mlp_chains) == 2

    # All attn chains: 4 members (q/k/v/o), perm_targets [0, 0, 0, 1]
    for c in attn_chains:
        assert c.kind == "fan_out_chain"
        assert len(c.members) == 4
        assert c.perm_targets == [0, 0, 0, 1], f"got {c.perm_targets}"
        assert c.shared_dim_size == 64
        print(f"  attn chain: {c.members[0].split('.self_attn.')[1] if False else c.members[-1].split('.')[-1]} block, {len(c.members)} members, targets {c.perm_targets}")

    # All mlp chains: 3 members (gate/up/down), perm_targets [0, 0, 1]
    for c in mlp_chains:
        assert c.kind == "fan_out_chain"
        assert len(c.members) == 3
        assert c.perm_targets == [0, 0, 1], f"got {c.perm_targets}"
        assert c.shared_dim_size == 128
        print(f"  mlp chain: {c.members[-1].split('.')[-1]} block, {len(c.members)} members, targets {c.perm_targets}")

    print(f"  discover_chains: {len(chains)} chains discovered correctly")


def test_apply_chain_preserves_forward():
    """Apply a found permutation to a chain and verify forward pass is preserved."""
    torch.manual_seed(0)
    model = TinyModel(n_layers=1)
    chains = discover_chains(model, skip_modules=("embed", "norm", "lm_head"))

    # Forward a dummy input through the first block (attn + mlp) and capture
    X = torch.randn(2, 8, 64)

    def fwd_block(block, x):
        # Self-attn
        q = block.self_attn.q_proj(x)
        k = block.self_attn.k_proj(x)
        v = block.self_attn.v_proj(x)
        # simple attention
        scores = q @ k.transpose(-1, -2) / 8.0
        weights = torch.softmax(scores, dim=-1)
        attn = weights @ v
        x = block.self_attn.o_proj(attn)
        # MLP
        gate = block.mlp.gate_proj(x)
        up = block.mlp.up_proj(x)
        h = torch.nn.functional.silu(gate) * up
        x = block.mlp.down_proj(h)
        return x

    block = model.layers[0]
    out_orig = fwd_block(block, X)

    # Mock Fisher for each linear (full dotted paths matching chain.fisher_source)
    fisher_norm = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            fisher_norm[name] = torch.rand(mod.out_features)

    # Find a perm for the attn chain
    for chain in chains:
        F = fisher_norm[chain.fisher_source]
        W_A = chain.members[0]  # dotted
        W_B = chain.members[-1]
        w_a = model
        for part in W_A.split("."):
            w_a = getattr(w_a, part)
        w_b = model
        for part in W_B.split("."):
            w_b = getattr(w_b, part)
        w_a = w_a.weight.detach().float()
        w_b = w_b.weight.detach().float()
        perm = find_permutation_composite(w_a, w_b, F)
        apply_chain(model, chain, perm, F)

    out_perm = fwd_block(block, X)
    diff = (out_perm - out_orig).abs().max().item()
    print(f"  end-to-end forward diff: {diff:.3e}")
    # FP32 budget for 2 matmul chains: ~1e-5
    assert diff < 1e-3, f"forward diverged by {diff}"


def test_compute_fisher_accumulates_nonzero_gradients():
    """compute_fisher captures gradients from the real loss graph."""
    torch.manual_seed(0)
    model = TinyCausalLM()
    fisher = compute_fisher(
        model,
        TinyTokenizer(),
        ["abcde", "fghij"],
        layer_filter=lambda name, mod: name == "proj",
        max_length=8,
        device="cpu",
        show_progress=False,
    )

    assert "proj" in fisher
    assert fisher["proj"].n_samples > 0
    total = fisher["proj"].fisher.sum().item()
    print(f"  fisher sum: {total:.3e}")
    assert total > 0.0, "expected nonzero Fisher from retained output gradients"


def test_select_chains_limits_and_filters_by_member_name():
    model = TinyModel(n_layers=2)
    chains = discover_chains(model, skip_modules=("embed", "norm", "lm_head"))

    selected = select_chains(chains, name_filters=("mlp",), max_chains=1)

    assert len(selected) == 1
    assert selected[0].kind == "fan_out_chain"
    assert all(".mlp." in member for member in selected[0].members)


def test_compute_fisher_rejects_empty_tokenization():
    model = TinyCausalLM()
    try:
        compute_fisher(
            model,
            EmptyTokenizer(),
            ["abcde"],
            layer_filter=lambda name, mod: name == "proj",
            device="cpu",
            show_progress=False,
        )
    except ValueError as exc:
        assert "at least 2 tokens" in str(exc)
    else:
        raise AssertionError("expected compute_fisher to reject empty tokenization")


def test_compute_fisher_supports_last_logit_mean_loss():
    torch.manual_seed(0)
    model = TinyCausalLM()
    fisher = compute_fisher(
        model,
        TinyTokenizer(),
        ["abcde"],
        layer_filter=lambda name, mod: name == "proj",
        max_length=8,
        device="cpu",
        show_progress=False,
        loss_mode="last_logit_mean",
    )

    assert fisher["proj"].n_samples > 0
    assert fisher["proj"].fisher.sum().item() > 0.0


def test_compute_fisher_does_not_accumulate_parameter_gradients():
    model = TinyCausalLM()
    before = {name: param.requires_grad for name, param in model.named_parameters()}

    compute_fisher(
        model,
        TinyTokenizer(),
        ["abcde"],
        layer_filter=lambda name, mod: name == "proj",
        max_length=8,
        device="cpu",
        show_progress=False,
        loss_mode="last_logit_mean",
    )

    after = {name: param.requires_grad for name, param in model.named_parameters()}
    assert after == before
    assert all(param.grad is None for param in model.parameters())


def main() -> int:
    print("=" * 60)
    print("ICS pipeline smoke tests")
    print("=" * 60)
    tests = [
        ("discover_chains", test_discover_chains),
        ("apply_chain_preserves_forward", test_apply_chain_preserves_forward),
        ("compute_fisher_accumulates_nonzero_gradients", test_compute_fisher_accumulates_nonzero_gradients),
        ("select_chains_limits_and_filters_by_member_name", test_select_chains_limits_and_filters_by_member_name),
        ("compute_fisher_rejects_empty_tokenization", test_compute_fisher_rejects_empty_tokenization),
        ("compute_fisher_supports_last_logit_mean_loss", test_compute_fisher_supports_last_logit_mean_loss),
        ("compute_fisher_does_not_accumulate_parameter_gradients", test_compute_fisher_does_not_accumulate_parameter_gradients),
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
