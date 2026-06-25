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
        "layers": {},
        "permutations": {},
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
        }

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
    bs = meta["block_size"]
    for layer_name, info in meta["layers"].items():
        safe = layer_name.replace(".", "__")
        qt = QuantizedTensor(
            qdata=qdata[safe + ".qdata"],
            scales=scales[safe + ".scales"],
            zeros=zeros[safe + ".zeros"],
            bits=bits[safe + ".bits"],
            block_size=bs,
            original_shape=tuple(info["original_shape"]),
            quant_dim=info.get("quant_dim", -1),
            method=info["method"],
        )
        layers[layer_name] = qt

    return {
        "qdata": qdata,
        "scales": scales,
        "zeros": zeros,
        "bits": bits,
        "meta": meta,
        "layers": layers,
    }


def dequantized_state_dict(loaded: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Build a dense (dequantized) state-dict for verification loading."""
    state: dict[str, torch.Tensor] = {}
    for layer_name, qt in loaded["layers"].items():
        state[layer_name + ".weight"] = dequantize_blockwise(qt)
    return state
