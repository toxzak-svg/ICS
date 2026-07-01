"""Tests for ERC balanced low-bit block promotion."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from ics.export import load_ics_model, save_ics_model
from ics.pipeline import ICSConfig, ICSResult
from ics.quantize import (
    QuantizedTensor,
    balance_bit_widths_by_error,
    quantize_blockwise,
)


def test_erc_promotes_unsafe_low_bit_block():
    torch.manual_seed(0)
    W = torch.randn(4, 8) * 0.05
    W[:, 0] = torch.tensor([4.0, -3.5, 3.0, -2.5])
    bits = torch.tensor([1, 1], dtype=torch.int32)
    fisher = torch.tensor([10.0, 8.0, 7.0, 6.0, 0.1, 0.1, 0.1, 0.1])

    balanced, promoted, scores = balance_bit_widths_by_error(
        W,
        bits,
        fisher,
        block_size=4,
        dim=1,
        max_relative_error=0.10,
    )

    assert balanced.tolist() == [4, 1]
    assert promoted.tolist() == [True, False]
    assert scores[0] > 0.10


def test_erc_keeps_low_bit_block_when_budget_allows_it():
    torch.manual_seed(1)
    W = torch.randn(4, 8) * 0.01
    bits = torch.tensor([2, 1], dtype=torch.int32)
    fisher = torch.ones(8)

    balanced, promoted, _ = balance_bit_widths_by_error(
        W,
        bits,
        fisher,
        block_size=4,
        dim=1,
        max_relative_error=3.0,
    )

    assert balanced.tolist() == [2, 1]
    assert promoted.tolist() == [False, False]


def test_quantize_blockwise_records_erc_promotions():
    W = torch.tensor(
        [
            [4.0, 0.05, -0.04, 0.02, 0.01, -0.01, 0.02, -0.02],
            [-3.5, 0.04, -0.03, 0.01, -0.01, 0.01, -0.02, 0.02],
        ],
        dtype=torch.float32,
    )
    bits = torch.tensor([1, 1], dtype=torch.int32)
    fisher = torch.tensor([10.0, 8.0, 6.0, 4.0, 0.1, 0.1, 0.1, 0.1])

    qt = quantize_blockwise(
        W,
        bits,
        block_size=4,
        dim=1,
        erc_fisher=fisher,
        erc_max_relative_error=0.10,
    )

    assert qt.bits.tolist() == [4, 1]
    assert qt.erc_promoted is not None
    assert qt.erc_promoted.tolist() == [True, False]


def test_export_round_trips_erc_metadata():
    qt = QuantizedTensor(
        qdata=torch.tensor([0, 1, 2, 3], dtype=torch.int8),
        scales=torch.ones(1, dtype=torch.float32),
        zeros=torch.zeros(1, dtype=torch.int32),
        bits=torch.tensor([4], dtype=torch.int32),
        block_size=4,
        original_shape=(1, 4),
        quant_dim=1,
        erc_promoted=torch.tensor([True]),
    )
    result = ICSResult(
        perms={},
        quant={"layer": qt},
        bit_widths={"layer": qt.bits},
        fisher={},
        config=ICSConfig(erc_enabled=True, erc_max_relative_error=0.10),
    )

    with tempfile.TemporaryDirectory() as tmp:
        out = save_ics_model(result, tmp)
        meta = json.loads((out / "ics_meta.json").read_text())
        loaded = load_ics_model(out)

    assert meta["erc_enabled"] is True
    assert meta["erc_max_relative_error"] == 0.10
    assert meta["layers"]["layer"]["erc_promoted"] == [True]
    assert loaded["layers"]["layer"].erc_promoted.tolist() == [True]


def main() -> int:
    print("=" * 60)
    print("ERC quantization tests")
    print("=" * 60)
    tests = [
        ("erc_promotes_unsafe_low_bit_block", test_erc_promotes_unsafe_low_bit_block),
        ("erc_keeps_low_bit_block_when_budget_allows_it", test_erc_keeps_low_bit_block_when_budget_allows_it),
        ("quantize_blockwise_records_erc_promotions", test_quantize_blockwise_records_erc_promotions),
        ("export_round_trips_erc_metadata", test_export_round_trips_erc_metadata),
    ]
    failures = []
    for name, fn in tests:
        print(f"\n[test] {name}")
        try:
            fn()
            print("  PASS")
        except Exception as exc:
            print(f"  FAIL: {exc}")
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
