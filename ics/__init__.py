"""Isomorphic Channel Sorting (ICS).

Offline topological weight permutation + variable-bit block quantization
for NPU-friendly deployment. Replaces runtime weight patching (FABQ-RC
and friends) with a compile-time sort that yields a single contiguous
INT4/INT2/INT1 memory stream.

Pipeline:
    fisher.py       — activation-Fisher Information per channel
    permutation.py  — composite-score / Sinkhorn-Hungarian sort
    quantize.py     — variable-bit block quantization
    pipeline.py     — end-to-end orchestrator
    export.py       — safetensors serialization with perm metadata
"""

from ics.fisher import compute_fisher, FisherStats
from ics.permutation import (
    find_permutation_composite,
    find_permutation_sinkhorn_hungarian,
    apply_permutation_to_chain,
    PermutationResult,
)
from ics.quantize import quantize_blockwise, dequantize_blockwise, QuantizedTensor
from ics.pipeline import quantize_model, ICSConfig
from ics.export import save_ics_model, load_ics_model

__all__ = [
    "compute_fisher",
    "FisherStats",
    "find_permutation_composite",
    "find_permutation_sinkhorn_hungarian",
    "apply_permutation_to_chain",
    "PermutationResult",
    "quantize_blockwise",
    "dequantize_blockwise",
    "QuantizedTensor",
    "quantize_model",
    "ICSConfig",
    "save_ics_model",
    "load_ics_model",
]

__version__ = "0.1.0"
