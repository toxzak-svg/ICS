"""End-to-end ICS pipeline.

Discovers linear layers in an HF causal LM, computes Fisher, finds
permutations for sequential chains and fan-out groups, applies them,
and quantizes the result.

The pipeline is designed to work on a 4-bit-loaded model (BitsAndBytes
Linear4bit) so it can run on a T4. Activation Fisher is computed with
the model in train() mode to enable gradients; backward through
Linear4bit is supported in BitsAndBytes >= 0.43.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn

from ics.fisher import compute_fisher, FisherStats
from ics.permutation import (
    apply_permutation_to_chain,
    fan_out_permutation,
    find_permutation_composite,
    find_permutation_sinkhorn_hungarian,
    PermutationResult,
)
from ics.quantize import (
    assign_bit_widths,
    dequantize_blockwise,
    quantize_blockwise,
    QuantizedTensor,
)


@dataclass
class ICSConfig:
    """Configuration for the ICS pipeline.

    Attributes:
        block_size: hardware NPU block size (default 64).
        int4_fraction: fraction of blocks assigned INT4 (top-Fisher).
        int2_fraction: fraction assigned INT2 (next).
        int1_fraction: fraction assigned INT1 (tail).
        alpha: weight on W_A's column-outlier contribution in composite
            sort.
        beta: weight on W_B's row-outlier contribution in composite sort.
        method: "composite" (fast) or "sinkhorn_hungarian" (clean).
        calibration_texts: list of strings for Fisher computation.
        max_calibration_length: max tokens per calibration sample.
        skip_modules: dotted-path substrings; modules matching any
            substring are skipped (e.g. "lm_head", "embed_tokens").
    """

    block_size: int = 64
    int4_fraction: float = 0.5
    int2_fraction: float = 0.4
    int1_fraction: float = 0.1
    alpha: float = 1.0
    beta: float = 1.0
    method: str = "composite"
    calibration_texts: Sequence[str] = field(default_factory=list)
    max_calibration_length: int = 256
    skip_modules: tuple[str, ...] = (
        "lm_head",
        "embed_tokens",
        "embed_out",
        "norm",
        "rotary_emb",
    )


@dataclass
class LinearChainSpec:
    """A description of a linear chain or fan-out group to ICS-ize.

    For a "chain" (W_A → W_B sequential), the perm is on the shared
    channel dim: rows of A, columns of B.

    For a "fan_out_chain" (e.g., gate_proj + up_proj producing
    intermediate, down_proj consuming it), the perm is on the shared
    inter dim: rows of producers, columns of consumer.

    For a "solo" layer (e.g., o_proj, down_proj consumed individually),
    the perm is on the appropriate input or output dim.

    Attributes:
        kind: "chain", "fan_out_chain", or "solo"
        members: dotted module paths of the layers involved
        perm_targets: for each member, 0 = permute rows (output dim),
            1 = permute columns (input dim)
        fisher_source: layer whose Fisher vector drives the sort
        shared_dim_size: length of the perm (i.e., the shared channel dim)
    """

    kind: str
    members: list[str]
    perm_targets: list[int]
    fisher_source: str
    shared_dim_size: int = 0


@dataclass
class ICSResult:
    """The output of the ICS pipeline for a single model.

    Attributes:
        perms: dict[chain_name -> PermutationResult]
        quant: dict[layer_name -> QuantizedTensor]
        bit_widths: dict[layer_name -> Tensor[block -> bits]]
        fisher: dict[layer_name -> FisherStats]
        config: ICSConfig
    """

    perms: dict[str, PermutationResult]
    quant: dict[str, QuantizedTensor]
    bit_widths: dict[str, torch.Tensor]
    fisher: dict[str, FisherStats]
    config: ICSConfig


def discover_chains(model: nn.Module, skip_modules: tuple[str, ...]) -> list[LinearChainSpec]:
    """Discover the linear chains in a HF causal LM.

    The canonical ICS pattern: in each transformer block, find the
    "producers" (q/k/v for self_attn, gate/up for mlp) and the
    "consumer" (o_proj for self_attn, down_proj for mlp). The shared
    perm dim is the producers' output dim = consumer's input dim.

    ICS target: this shared dim. Producers get their rows permuted,
    the consumer gets its columns permuted. Element-wise ops (SiLU,
    *) and softmax commute with the perm.
    """
    name_to_module: dict[str, nn.Module] = dict(model.named_modules())
    chains: list[LinearChainSpec] = []

    # Group linear layers by their parent block (the .self_attn / .mlp)
    by_attn_block: dict[str, list[str]] = {}
    by_mlp_block: dict[str, list[str]] = {}

    for name, mod in name_to_module.items():
        if not isinstance(mod, nn.Linear):
            continue
        if any(s in name for s in skip_modules):
            continue
        parts = name.split(".")
        if "self_attn" in parts:
            idx = parts.index("self_attn")
            block = ".".join(parts[:idx + 1])
            by_attn_block.setdefault(block, []).append(name)
        elif "mlp" in parts:
            idx = parts.index("mlp")
            block = ".".join(parts[:idx + 1])
            by_mlp_block.setdefault(block, []).append(name)

    # Attention block: q/k/v fan out on output dim, o_proj consumes.
    # The shared perm dim is q/k/v output_features = o_proj input_features.
    for block, names in by_attn_block.items():
        qkv = [n for n in names if n.endswith(("q_proj", "k_proj", "v_proj"))]
        o = [n for n in names if n.endswith("o_proj")]
        if qkv and o:
            # Use the first q/k/v as the (W_A, W_B) source for the joint perm
            w_q = _get_module(model, qkv[0]).weight
            w_o = _get_module(model, o[0]).weight
            shared = w_q.shape[0]  # q's out = o's in
            members = qkv + o
            # permute rows of q/k/v, columns of o
            perm_targets = [0] * len(qkv) + [1]
            chains.append(LinearChainSpec(
                kind="fan_out_chain",
                members=members,
                perm_targets=perm_targets,
                fisher_source=qkv[0],
                shared_dim_size=shared,
            ))
        elif o:
            # Solo o_proj (no q/k/v in this block)
            w_o = _get_module(model, o[0]).weight
            chains.append(LinearChainSpec(
                kind="solo",
                members=o,
                perm_targets=[1],
                fisher_source=o[0],
                shared_dim_size=w_o.shape[1],
            ))

    # MLP block: gate/up fan out on output dim, down_proj consumes.
    # The shared perm dim is gate/up output_features = down input_features.
    for block, names in by_mlp_block.items():
        gu = [n for n in names if n.endswith(("gate_proj", "up_proj"))]
        d = [n for n in names if n.endswith("down_proj")]
        if gu and d:
            w_g = _get_module(model, gu[0]).weight
            w_d = _get_module(model, d[0]).weight
            shared = w_g.shape[0]  # gate's out = down's in
            members = gu + d
            perm_targets = [0] * len(gu) + [1]
            chains.append(LinearChainSpec(
                kind="fan_out_chain",
                members=members,
                perm_targets=perm_targets,
                fisher_source=gu[0],
                shared_dim_size=shared,
            ))
        elif d:
            w_d = _get_module(model, d[0]).weight
            chains.append(LinearChainSpec(
                kind="solo",
                members=d,
                perm_targets=[1],
                fisher_source=d[0],
                shared_dim_size=w_d.shape[1],
            ))

    return chains


def _get_module(model: nn.Module, name: str) -> nn.Module:
    mod = model
    for part in name.split("."):
        mod = getattr(mod, part)
    return mod


def _set_module(model: nn.Module, name: str, new_mod: nn.Module) -> None:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_mod)


def apply_chain(
    model: nn.Module,
    chain: LinearChainSpec,
    perm: PermutationResult,
    F: torch.Tensor,
) -> None:
    """Apply a found permutation to all members of a chain."""
    P_np = perm.permutation.cpu().numpy()
    for n, target in zip(chain.members, chain.perm_targets):
        W = _get_module(model, n).weight
        if target == 0:
            # Permute rows (output channels)
            W_new = W[P_np, :].clone()
        elif target == 1:
            # Permute columns (input channels)
            W_new = W[:, P_np].clone()
        else:
            raise ValueError(f"perm_target must be 0 (rows) or 1 (cols), got {target}")
        mod = _get_module(model, n)
        mod.weight.data = W_new.to(mod.weight.data.device, mod.weight.data.dtype)


def quantize_model(
    model: nn.Module,
    tokenizer,
    config: ICSConfig,
    device: str | torch.device = "cuda",
    show_progress: bool = True,
) -> ICSResult:
    """Run the full ICS pipeline on a model.

    Steps:
        1. Discover chains (fan-out groups, solo layers).
        2. Compute activation Fisher for every relevant linear layer.
        3. For each chain, find a permutation (composite or sinkhorn-H).
        4. Apply permutations to the model's weights in-place.
        5. For each linear layer, assign bit-widths per block and
           quantize.
    """
    if show_progress:
        print("[1/5] Discovering linear chains...", flush=True)
    chains = discover_chains(model, config.skip_modules)
    if show_progress:
        print(f"      found {len(chains)} chains", flush=True)

    if show_progress:
        print("[2/5] Computing activation Fisher Information...", flush=True)
    fisher = compute_fisher(
        model,
        tokenizer,
        config.calibration_texts,
        layer_filter=lambda name, mod: any(name in c.members for c in chains),
        max_length=config.max_calibration_length,
        device=device,
        show_progress=show_progress,
    )
    if show_progress:
        print(f"      computed Fisher for {len(fisher)} layers", flush=True)

    # Convert to normalized per-channel tensors
    fisher_norm: dict[str, torch.Tensor] = {}
    for name, fs in fisher.items():
        f = fs.fisher / (fs.fisher.max() + 1e-12)
        fisher_norm[name] = f.cpu()

    if show_progress:
        print("[3/5] Finding permutations...", flush=True)
    perms: dict[str, PermutationResult] = {}
    for i, chain in enumerate(chains):
        F = fisher_norm[chain.fisher_source]
        # Use the first producer (rows-permuted) and the consumer
        # (cols-permuted) as the (W_A, W_B) for the joint perm. The
        # resulting P is then applied uniformly to all members'
        # appropriate dim.
        W_A = _get_module(model, chain.members[0]).weight.detach().float()
        if len(chain.members) > 1:
            W_B = _get_module(model, chain.members[-1]).weight.detach().float()
        else:
            W_B = W_A.clone()

        if W_A.shape[0] != W_B.shape[1]:
            # Mismatched dims (shouldn't happen with the new chain
            # structure, but guard). Fall back to single-W.
            W_B = W_A.clone()

        if config.method == "composite":
            perm = find_permutation_composite(
                W_A, W_B, F,
                alpha=config.alpha, beta=config.beta,
            )
        elif config.method == "sinkhorn_hungarian":
            perm = find_permutation_sinkhorn_hungarian(
                W_A, W_B, F, block_size=config.block_size,
            )
        else:
            raise ValueError(f"unknown method: {config.method}")

        key = "/".join(chain.members)
        perms[key] = perm
        if show_progress and (i + 1) % 4 == 0:
            print(f"      {i + 1}/{len(chains)} chains sorted", flush=True)

    if show_progress:
        print("[4/5] Applying permutations to weights...", flush=True)
    for chain in chains:
        key = "/".join(chain.members)
        F = fisher_norm[chain.fisher_source]
        apply_chain(model, chain, perms[key], F)

    if show_progress:
        print("[5/5] Quantizing layers...", flush=True)
    quant: dict[str, QuantizedTensor] = {}
    bit_widths: dict[str, torch.Tensor] = {}
    for chain in chains:
        for n in chain.members:
            if n in quant:
                continue
            W = _get_module(model, n).weight.detach().float().cpu()
            F = fisher_norm[n]
            bits = assign_bit_widths(
                F,
                block_size=config.block_size,
                int4_fraction=config.int4_fraction,
                int2_fraction=config.int2_fraction,
                int1_fraction=config.int1_fraction,
            )
            qt = quantize_blockwise(W, bits, block_size=config.block_size)
            quant[n] = qt
            bit_widths[n] = bits

    return ICSResult(
        perms=perms,
        quant=quant,
        bit_widths=bit_widths,
        fisher=fisher,
        config=config,
    )
