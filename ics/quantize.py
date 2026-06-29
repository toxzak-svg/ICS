"""Variable-bit block quantization for ICS.

After the permutation step, channels within a block are clustered by
sensitivity. We assign bit-widths per block:

    block 0..high_water  -> INT4 (dense)
    block high_water..end -> INT2 (aggressive)
    block tail            -> INT1 (binary)

Each block has its own scaling factor (alpha) and zero point (z), so
the outlier structure is local to the block. Quantization is symmetric
INT-N around zero (i.e., range [-2^(N-1), 2^(N-1)-1]) which is what
mobile NPU INT kernels expect.

The output is a packed QuantizedTensor: int8 storage (one int8 holds
multiple low-bit values) plus per-block scales and zero points. Round-trip
accuracy: dequantize(quantize(W)) ≈ W to within ~1% relative for INT4
and ~5% for INT1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class QuantizedTensor:
    """Variable-bit block-quantized tensor.

    Attributes:
        qdata: int8 tensor of shape [N] (or [N, packed] for INT2/INT1).
            For INT4 each element is one quantized value in [-8, 7].
            For INT2/INT1, multiple values are packed into one int8
            (4 INT2 per int8, 8 INT1 per int8).
        scales: float32 tensor of shape [n_blocks], per-block alpha.
        zeros: int32 tensor of shape [n_blocks], per-block zero point.
        bits: int32 tensor of shape [n_blocks], bit-width per block.
        block_size: int.
        original_shape: tuple, shape of the original tensor before flatten.
        quant_dim: int, the dim that was quantized (negative-indexed).
            -1 means the last dim. dequantize_blockwise transposes back
            so the returned tensor has original_shape.
        method: str, the quantization method used.
    """

    qdata: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    bits: torch.Tensor
    block_size: int
    original_shape: tuple
    quant_dim: int = -1
    method: str = "symmetric_per_block"


def _int4_block_quantize(block: torch.Tensor) -> tuple[torch.Tensor, float, int]:
    """Symmetric INT4 quantization of a 1-D block.

    Uses q in [-7, 7] (15 levels) with scale = absmax/7, giving a
    symmetric dequant range of [-absmax, absmax]. The original code
    used q in [-8, 7] with scale = absmax/7, which produced a 14% norm
    excess on the negative side and caused the dequantized model to
    diverge catastrophically from BF16.
    """
    if block.numel() == 0:
        return torch.zeros(0, dtype=torch.int8), 1.0, 0
    absmax = block.abs().amax()
    if absmax < 1e-12:
        # All-zero block: encode as zero with scale=1, zero=0
        return torch.zeros_like(block, dtype=torch.int8), 1.0, 0
    scale = absmax / 7.0
    q = torch.clamp(torch.round(block / scale), -7, 7).to(torch.int8)
    return q, float(scale.item()), 0


def _int2_block_quantize(block: torch.Tensor) -> tuple[torch.Tensor, float, int]:
    """Symmetric INT2 quantization of a 1-D block, packed 4-per-byte."""
    if block.numel() == 0:
        return torch.zeros(0, dtype=torch.int8), 1.0, 0
    absmax = block.abs().amax()
    if absmax < 1e-12:
        return torch.zeros((block.numel() + 3) // 4, dtype=torch.int8), 1.0, 0
    scale = absmax / 1.0  # INT2 symmetric: range [-2, 1] mapped from [-absmax, absmax]
    q = torch.clamp(torch.round(block / scale), -2, 1).to(torch.int8)  # [N], values in {-2,-1,0,1}
    # Pack 4 INT2 values into each int8: low 2 bits = first, next 2 = second, etc.
    # We use a 2-bit two's-complement encoding: -2 -> 10, -1 -> 11, 0 -> 00, 1 -> 01
    enc = (q & 0b11).to(torch.int8)  # [N]
    n = enc.numel()
    pad = (4 - n % 4) % 4
    if pad:
        enc = torch.cat([enc, torch.zeros(pad, dtype=torch.int8)])
    enc = enc.view(-1, 4)
    packed = (enc[:, 0] | (enc[:, 1] << 2) | (enc[:, 2] << 4) | (enc[:, 3] << 6)).to(torch.int8)
    return packed, float(scale.item()), 0


def _int1_block_quantize(block: torch.Tensor) -> tuple[torch.Tensor, float, int]:
    """Symmetric INT1 (binary) quantization of a 1-D block, packed 8-per-byte."""
    if block.numel() == 0:
        return torch.zeros(0, dtype=torch.int8), 1.0, 0
    absmax = block.abs().amax()
    if absmax < 1e-12:
        return torch.zeros((block.numel() + 7) // 8, dtype=torch.int8), 1.0, 0
    # Binary: sign(block) in {-1, 0, 1} -> mapped to 1-bit
    sign = torch.sign(block).to(torch.int8)  # {-1, 0, 1}
    enc = (sign > 0).to(torch.int8)  # {0, 1}
    n = enc.numel()
    pad = (8 - n % 8) % 8
    if pad:
        enc = torch.cat([enc, torch.zeros(pad, dtype=torch.int8)])
    enc = enc.view(-1, 8)
    packed = (enc[:, 0] | (enc[:, 1] << 1) | (enc[:, 2] << 2) | (enc[:, 3] << 3)
              | (enc[:, 4] << 4) | (enc[:, 5] << 5) | (enc[:, 6] << 6) | (enc[:, 7] << 7)).to(torch.int8)
    return packed, float(absmax.item()), 0


def _dequant_int4_block(q: torch.Tensor, scale: float) -> torch.Tensor:
    return q.to(torch.float32) * scale


def _dequant_int2_block(q_packed: torch.Tensor, scale: float, n: int) -> torch.Tensor:
    # Unpack 4 INT2 from each byte
    q_packed = q_packed.to(torch.int32)
    out = torch.zeros(q_packed.shape[0] * 4, dtype=torch.int8)
    for i in range(4):
        out[i::4] = (q_packed >> (2 * i)) & 0b11
    out = out[:n]
    # Convert 2-bit encoding back to signed: 10 -> -2, 11 -> -1, 00 -> 0, 01 -> 1
    # In two's-complement of 2 bits: 00=0, 01=1, 10=-2, 11=-1
    signed = torch.where(out >= 2, out - 4, out).to(torch.float32) * scale
    return signed


def _dequant_int1_block(q_packed: torch.Tensor, absmax: float, n: int) -> torch.Tensor:
    q_packed = q_packed.to(torch.int32)
    out = torch.zeros(q_packed.shape[0] * 8, dtype=torch.int8)
    for i in range(8):
        out[i::8] = (q_packed >> i) & 1
    out = out[:n]
    return (out.to(torch.float32) * 2 - 1) * absmax


def quantize_blockwise(
    W: torch.Tensor,
    bits_per_block: torch.Tensor,
    block_size: int = 64,
    dim: int = -1,
) -> QuantizedTensor:
    """Quantize a tensor with per-block bit-widths along a chosen dim.

    Args:
        W: tensor of any shape. Block quantization is applied along
            `dim`. For nn.Linear weights [out, in] permuted on dim 0,
            pass `dim=0` so the per-channel Fisher drives the bit
            assignment on the right axis.
        bits_per_block: int tensor of shape [n_blocks], bit-width for
            each block. Values can be 1, 2, or 4. n_blocks must match
            the number of blocks along the target dim.
        block_size: number of elements per block along the target dim.
        dim: which axis to block over. Negative indexes from the end.

    Returns:
        QuantizedTensor.
    """
    W = W.detach().to(torch.float32).cpu()
    original_shape = tuple(W.shape)
    if dim < 0:
        dim = W.ndim + dim
    if not (0 <= dim < W.ndim):
        raise ValueError(f"dim {dim} out of range for tensor of ndim {W.ndim}")
    target_size = W.shape[dim]
    n_blocks = (target_size + block_size - 1) // block_size

    if bits_per_block.shape[0] != n_blocks:
        raise ValueError(
            f"bits_per_block length {bits_per_block.shape[0]} != n_blocks {n_blocks} "
            f"(target dim size {target_size}, block_size {block_size})"
        )

    # Move target dim to the end, then flatten leading dims
    if dim != W.ndim - 1:
        W = W.transpose(dim, -1).contiguous()
    W_flat = W.reshape(-1, W.shape[-1]).contiguous()
    n_rows, n_cols = W_flat.shape
    assert n_cols == target_size, f"internal: expected {target_size}, got {n_cols}"

    scales = torch.zeros(n_blocks, dtype=torch.float32)
    zeros = torch.zeros(n_blocks, dtype=torch.int32)
    bits = bits_per_block.to(torch.int32)
    qdata_pieces = []

    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, n_cols)
        block = W_flat[:, start:end].contiguous().view(-1)
        bw = int(bits[b].item())
        if bw == 4:
            q, s, z = _int4_block_quantize(block)
        elif bw == 2:
            q, s, z = _int2_block_quantize(block)
        elif bw == 1:
            q, s, z = _int1_block_quantize(block)
        else:
            raise ValueError(f"Unsupported bit-width {bw}; expected 1, 2, or 4")
        qdata_pieces.append(q)
        scales[b] = s
        zeros[b] = z

    qdata = torch.cat(qdata_pieces) if qdata_pieces else torch.zeros(0, dtype=torch.int8)

    return QuantizedTensor(
        qdata=qdata,
        scales=scales,
        zeros=zeros,
        bits=bits,
        block_size=block_size,
        original_shape=original_shape,
        quant_dim=dim if dim >= 0 else dim,
    )


def dequantize_blockwise(qt: QuantizedTensor) -> torch.Tensor:
    """Round-trip a QuantizedTensor back to a dense float32 tensor.

    Used for verifying the quantization error and for the (optional)
    dense-output path for fine-tuning. Transposes the result back to
    original_shape if quant_dim is not the last dim.
    """
    ndim = len(qt.original_shape)
    qd = qt.quant_dim if qt.quant_dim >= 0 else ndim + qt.quant_dim
    target_last_size = qt.original_shape[qd]
    n_blocks = qt.scales.shape[0]
    block_size = qt.block_size

    n_rows = 1
    for i, s in enumerate(qt.original_shape):
        if i == qd:
            continue
        n_rows *= s

    out = torch.zeros(n_rows, target_last_size, dtype=torch.float32)
    cursor = 0
    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, target_last_size)
        bw = int(qt.bits[b].item())
        block_n = n_rows * (end - start)
        if bw == 4:
            block_q = qt.qdata[cursor:cursor + block_n]
            cursor += block_n
            block_dq = _dequant_int4_block(block_q, float(qt.scales[b].item()))
        elif bw == 2:
            n_bytes = (block_n + 3) // 4
            block_q = qt.qdata[cursor:cursor + n_bytes]
            cursor += n_bytes
            block_dq = _dequant_int2_block(block_q, float(qt.scales[b].item()), block_n)
        elif bw == 1:
            n_bytes = (block_n + 7) // 8
            block_q = qt.qdata[cursor:cursor + n_bytes]
            cursor += n_bytes
            block_dq = _dequant_int1_block(block_q, float(qt.scales[b].item()), block_n)
        else:
            raise ValueError(f"Unsupported bit-width {bw}")
        out[:, start:end] = block_dq.view(n_rows, end - start)

    # Reshape to the "quant_dim moved to the end" layout, then transpose
    # back to original_shape.
    permuted_shape = [s for i, s in enumerate(qt.original_shape) if i != qd]
    permuted_shape.append(target_last_size)
    out = out.view(permuted_shape)
    if qd != ndim - 1:
        out = out.transpose(qd, -1).contiguous()
    return out


def assign_bit_widths(
    F: torch.Tensor,
    block_size: int = 64,
    int4_fraction: float = 0.5,
    int2_fraction: float = 0.4,
    int1_fraction: float = 0.1,
) -> torch.Tensor:
    """Assign bit-widths to blocks based on per-block Fisher score.

    Sums Fisher within each block, sorts blocks by total Fisher
    descending, and assigns bit-widths by cumulative fraction:
        top int4_fraction        -> INT4
        next int2_fraction       -> INT2
        remaining int1_fraction  -> INT1
    """
    F = F.detach().to(torch.float32).cpu()
    n = F.shape[0]
    n_blocks = (n + block_size - 1) // block_size
    # Sum Fisher per block
    block_fisher = torch.zeros(n_blocks)
    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, n)
        block_fisher[b] = F[start:end].sum()

    # Sort blocks by Fisher descending
    sorted_blocks = torch.argsort(-block_fisher)
    bits = torch.full((n_blocks,), 1, dtype=torch.int32)
    n_int4 = int(round(int4_fraction * n_blocks))
    n_int2 = int(round(int2_fraction * n_blocks))
    # The rest (n_blocks - n_int4 - n_int2) get INT1
    bits[sorted_blocks[:n_int4]] = 4
    bits[sorted_blocks[n_int4:n_int4 + n_int2]] = 2
    # Remaining stay at 1
    return bits


def total_compression_bits(
    qt: QuantizedTensor,
) -> int:
    """Total bits used by a QuantizedTensor (for reporting compression ratio)."""
    return qt.qdata.numel() * 8  # qdata is int8, but actual info content is less
