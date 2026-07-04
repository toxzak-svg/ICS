"""Safetensors export for ICS-quantized models.

The quantized model is saved as a directory with:
    - model.safetensors:    packed quantized weights (int8 storage)
    - scales.safetensors:   per-block scale factors (float32)
    - zeros.safetensors:    per-block zero points (int32)
    - bits.safetensors:     per-block bit-widths (int32)
    - ics_meta.json:        metadata (original shapes, permutations, etc.)
    - tokenizer files:      copied from the source model

To load: rebuild the dense weights by dequantizing, then either:
    (a) run inference in the dense (dequantized) form for verification
    (b) load as a custom NPU kernel that consumes the packed layout
For verification we provide the dense path.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from ics.pipeline import ICSResult, ICSConfig
from ics.quantize import QuantizedTensor, dequantize_blockwise


def _qt_to_tensors(qt: QuantizedTensor) -> dict[str, torch.Tensor]:
    return {
        "qdata": qt.qdata,
        "scales": qt.scales,
        "zeros": qt.zeros,
        "bits": qt.bits,
    }


def save_ics_model(
    result: ICSResult,
    output_dir: str | Path,
    tokenizer=None,
    source_model_dir: str | Path | None = None,
) -> Path:
    """Save the ICS-quantized model to a directory.

    Args:
        result: the output of `quantize_model`.
        output_dir: target directory (will be created).
        tokenizer: optional HF tokenizer to copy alongside.
        source_model_dir: optional source model dir to copy config from.

    Returns:
        Path to the output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qdata_dict: dict[str, torch.Tensor] = {}
    scales_dict: dict[str, torch.Tensor] = {}
    zeros_dict: dict[str, torch.Tensor] = {}
    bits_dict: dict[str, torch.Tensor] = {}
    meta: dict[str, Any] = {
        "block_size": result.config.block_size,
        "int4_fraction": result.config.int4_fraction,
        "int2_fraction": result.config.int2_fraction,
        "int1_fraction": result.config.int1_fraction,
        "method": result.config.method,
        "quant_method": result.config.quant_method,
        "gptq_group_size": result.config.gptq_group_size,
        "erc_enabled": result.config.erc_enabled,
        "erc_max_relative_error": result.config.erc_max_relative_error,
        "layers": {},
        "permutations": {},
        "layer_perms": {},  # GPTQ column permutations per layer
    }

    for layer_name, qt in result.quant.items():
        safe = layer_name.replace(".", "__")
        tensors = _qt_to_tensors(qt)
        qdata_dict[safe + ".qdata"] = tensors["qdata"]
        scales_dict[safe + ".scales"] = tensors["scales"]
        zeros_dict[safe + ".zeros"] = tensors["zeros"]
        bits_dict[safe + ".bits"] = tensors["bits"]
        meta["layers"][layer_name] = {
            "original_shape": list(qt.original_shape),
            "block_size": qt.block_size,
            "quant_dim": qt.quant_dim,
            "method": qt.method,
            "erc_promoted": (
                qt.erc_promoted.cpu().tolist()
                if qt.erc_promoted is not None
                else None
            ),
            "erc_error_scores": (
                qt.erc_error_scores.cpu().tolist()
                if qt.erc_error_scores is not None
                else None
            ),
        }
        if layer_name in result.layer_perms:
            meta["layer_perms"][layer_name] = result.layer_perms[layer_name].cpu().tolist()

    # Per-layer chain perm info so dequant can un-perm.
    meta["chain_members"] = result.chain_members

    for chain_key, perm in result.perms.items():
        safe = chain_key.replace(".", "__").replace("/", "_")
        meta["permutations"][chain_key] = {
            "method": perm.method,
            "score": perm.score,
            "permutation": perm.permutation.cpu().tolist(),
        }

    save_file(qdata_dict, str(output_dir / "model.safetensors"))
    save_file(scales_dict, str(output_dir / "scales.safetensors"))
    save_file(zeros_dict, str(output_dir / "zeros.safetensors"))
    save_file(bits_dict, str(output_dir / "bits.safetensors"))

    with open(output_dir / "ics_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if tokenizer is not None:
        tokenizer.save_pretrained(str(output_dir))
    if source_model_dir is not None:
        src = Path(source_model_dir)
        for fname in ("config.json", "generation_config.json", "model.safetensors.index.json"):
            sp = src / fname
            if sp.exists():
                shutil.copy2(sp, output_dir / fname)

    return output_dir


def load_ics_model(output_dir: str | Path) -> dict[str, Any]:
    """Load a saved ICS model. Returns dict with qdata, scales, zeros, bits, meta."""
    output_dir = Path(output_dir)
    qdata = load_file(str(output_dir / "model.safetensors"))
    scales = load_file(str(output_dir / "scales.safetensors"))
    zeros = load_file(str(output_dir / "zeros.safetensors"))
    bits = load_file(str(output_dir / "bits.safetensors"))
    with open(output_dir / "ics_meta.json") as f:
        meta = json.load(f)

    # Reconstruct QuantizedTensor per layer
    layers: dict[str, QuantizedTensor] = {}
    default_bs = meta["block_size"]
    for layer_name, info in meta["layers"].items():
        safe = layer_name.replace(".", "__")
        # Per-layer block_size wins; fall back to the top-level default for
        # backward compatibility with older artifacts.
        per_layer_bs = info.get("block_size", default_bs)
        qt = QuantizedTensor(
            qdata=qdata[safe + ".qdata"],
            scales=scales[safe + ".scales"],
            zeros=zeros[safe + ".zeros"],
            bits=bits[safe + ".bits"],
            block_size=per_layer_bs,
            original_shape=tuple(info["original_shape"]),
            quant_dim=info.get("quant_dim", -1),
            method=info["method"],
            erc_promoted=(
                torch.tensor(info["erc_promoted"], dtype=torch.bool)
                if info.get("erc_promoted") is not None
                else None
            ),
            erc_error_scores=(
                torch.tensor(info["erc_error_scores"], dtype=torch.float32)
                if info.get("erc_error_scores") is not None
                else None
            ),
        )
        layers[layer_name] = qt

    return {
        "qdata": qdata,
        "scales": scales,
        "zeros": zeros,
        "bits": bits,
        "meta": meta,
        "layers": layers,
        "layer_perms": meta.get("layer_perms", {}),
        "chain_members": meta.get("chain_members", {}),
    }


def dequantized_state_dict(loaded: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Build a dense (dequantized) state-dict for verification loading.

    Handles two layers of permutation:
    1. GPTQ column permutation (if layer was GPTQ-quantized). Reverses to
       recover the post-ICS-perm weight in its natural column order.
    2. ICS chain permutation. The saved weights are in chain-permuted order.
       Reversing recovers the original (unpermuted) weight.

    The output state dict is suitable for loading into the original
    (unpermuted) HF model.
    """
    state: dict[str, torch.Tensor] = {}
    layer_perms = loaded.get("layer_perms", {})
    chain_members = loaded.get("chain_members", {})

    # Build per-layer chain-perm lookup: layer_name -> (target, perm).
    # For GQA chains, members in gqa_sub_perm_members use the chain's
    # gqa_sub_perm (smaller length, derived from main perm) instead of
    # the main perm. We prefer the explicit `member_perms` mapping if
    # present (newer artifacts); fall back to the main perm for older
    # artifacts without GQA metadata.
    layer_chain_perm: dict[str, tuple[int, list[int]]] = {}
    for chain_key, info in chain_members.items():
        gqa_set = set(info.get("gqa_sub_perm_members", []) or [])
        member_perms = info.get("member_perms")
        for member, target in zip(info["members"], info["targets"]):
            if member_perms is not None and member in member_perms:
                perm_list = member_perms[member]
            elif member in gqa_set and info.get("gqa_sub_perm") is not None:
                perm_list = info["gqa_sub_perm"]
            else:
                perm_list = info["permutation"]
            layer_chain_perm[member] = (target, perm_list)

    for layer_name, qt in loaded["layers"].items():
        # All layers use dequantize_blockwise; the GPTQ path stores qdata
        # in the same block-row-major layout that dequantize_blockwise
        # expects (per-block: n_rows × block_size, row-major, then next block).
        W = dequantize_blockwise(qt)

        # 1. Un-perm the GPTQ column permutation (if any)
        if layer_name in layer_perms:
            gperm = torch.tensor(layer_perms[layer_name], dtype=torch.long)
            W_unperm = torch.zeros_like(W)
            W_unperm[:, gperm] = W
            W = W_unperm

        # 2. Un-perm the ICS chain permutation (if any)
        if layer_name in layer_chain_perm:
            target, cperm_list = layer_chain_perm[layer_name]
            cperm = torch.tensor(cperm_list, dtype=torch.long)
            P_len = len(cperm)
            W_unperm = torch.zeros_like(W)
            if target == 0:
                # Row perm (output channels). Mirror the fused-handling in
                # pipeline._apply_perm_to_weight: if the weight has more
                # rows than the perm and is an exact multiple, the forward
                # apply split it into groups of P_len and permuted each
                # group. We must undo it group-by-group.
                if W.shape[0] == P_len:
                    W_unperm[cperm] = W
                elif W.shape[0] > P_len and W.shape[0] % P_len == 0:
                    n_groups = W.shape[0] // P_len
                    for g in range(n_groups):
                        g_start = g * P_len
                        W_unperm[g_start : g_start + P_len][cperm] = W[g_start : g_start + P_len]
                else:
                    raise ValueError(
                        f"row unperm: weight shape {tuple(W.shape)} not compatible "
                        f"with perm length {P_len}"
                    )
            else:
                # Col perm (input channels). Same fused handling for 2-D and 3-D
                # weights (Conv1d from linear_attn).
                if W.dim() == 2:
                    if W.shape[1] == P_len:
                        W_unperm[:, cperm] = W
                    elif W.shape[1] > P_len and W.shape[1] % P_len == 0:
                        n_groups = W.shape[1] // P_len
                        for g in range(n_groups):
                            g_start = g * P_len
                            W_unperm[:, g_start : g_start + P_len][:, cperm] = W[:, g_start : g_start + P_len]
                    else:
                        raise ValueError(
                            f"col unperm: weight shape {tuple(W.shape)} not compatible "
                            f"with perm length {P_len}"
                        )
                elif W.dim() == 3:
                    if W.shape[1] == P_len:
                        W_unperm[:, cperm, :] = W
                    elif W.shape[1] > P_len and W.shape[1] % P_len == 0:
                        n_groups = W.shape[1] // P_len
                        for g in range(n_groups):
                            g_start = g * P_len
                            W_unperm[:, g_start : g_start + P_len, :][:, cperm, :] = W[:, g_start : g_start + P_len, :]
                    else:
                        raise ValueError(
                            f"col unperm (Conv1d): weight shape {tuple(W.shape)} not compatible "
                            f"with perm length {P_len}"
                        )
                else:
                    raise ValueError(f"unexpected weight dim {W.dim()} for col unperm")
            W = W_unperm

        state[layer_name + ".weight"] = W
    return state
