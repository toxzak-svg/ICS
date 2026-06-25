"""Tests for the perplexity benchmark harness helpers."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_perplexity import (
    BenchmarkResult,
    format_results_table,
    parse_llama_perplexity,
    perplexity_from_nll,
)


def test_perplexity_from_nll():
    ppl = perplexity_from_nll(total_nll=math.log(10.0) * 4, token_count=4)
    assert abs(ppl - 10.0) < 1e-6


def test_parse_llama_perplexity_final_estimate():
    text = """
    [1]12.3000, window = 512, batch = 512
    Final estimate: PPL = 12.3456 +/- 0.1234
    """
    assert abs(parse_llama_perplexity(text) - 12.3456) < 1e-6


def test_parse_llama_perplexity_simple_label():
    text = "perplexity: 9.875\n"
    assert abs(parse_llama_perplexity(text) - 9.875) < 1e-6


def test_format_results_table_marks_unavailable():
    rows = [
        BenchmarkResult("fp16", 11.0, 128, 1.2, None),
        BenchmarkResult("q4_k_m", None, 0, 0.0, "missing q4 model"),
    ]
    table = format_results_table(rows)
    assert "fp16" in table
    assert "11.0000" in table
    assert "unavailable" in table
    assert "missing q4 model" in table


def main() -> int:
    tests = [
        test_perplexity_from_nll,
        test_parse_llama_perplexity_final_estimate,
        test_parse_llama_perplexity_simple_label,
        test_format_results_table_marks_unavailable,
    ]
    failures = []
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:
            print(f"FAIL {test.__name__}: {exc}")
            failures.append(test.__name__)
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print(f"ALL {len(tests)} TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
