"""Tests for the benchmark harness quality helpers."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_perplexity import (
    BenchmarkResult,
    format_results_table,
    parse_llama_perplexity,
    parse_args,
    parse_max_chains,
    perplexity_from_nll,
    summarize_quality,
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


def test_summarize_quality_scores_candidates_relative_to_baseline():
    rows = [
        BenchmarkResult("fp16", 20.0, 128, 1.2, None),
        BenchmarkResult("ics_dequantized", 22.0, 128, 1.4, None),
        BenchmarkResult("bad_candidate", 80.0, 128, 1.4, None),
    ]

    summary = summarize_quality(rows, baseline_name="fp16", min_quality_score=0.90)

    assert summary["baseline"] == "fp16"
    assert summary["threshold"] == 0.90
    assert abs(summary["candidates"]["ics_dequantized"]["quality_score"] - (20.0 / 22.0)) < 1e-6
    assert summary["candidates"]["ics_dequantized"]["status"] == "pass"
    assert summary["candidates"]["bad_candidate"]["status"] == "fail"


def test_summarize_quality_keeps_unavailable_candidates_out_of_gate():
    rows = [
        BenchmarkResult("fp16", 20.0, 128, 1.2, None),
        BenchmarkResult("ics_dequantized", None, 0, 0.0, "missing artifact"),
    ]

    summary = summarize_quality(rows, baseline_name="fp16", min_quality_score=0.90)

    candidate = summary["candidates"]["ics_dequantized"]
    assert candidate["quality_score"] is None
    assert candidate["status"] == "unavailable"
    assert candidate["note"] == "missing artifact"


def test_format_results_table_shows_quality_status():
    rows = [
        BenchmarkResult("fp16", 20.0, 128, 1.2, None),
        BenchmarkResult("ics_dequantized", 22.0, 128, 1.4, None),
    ]
    quality = summarize_quality(rows, baseline_name="fp16", min_quality_score=0.90)

    table = format_results_table(rows, quality)

    assert "| model | perplexity | quality | status | tokens | seconds | note |" in table
    assert "| fp16 | 20.0000 | 1.0000 | baseline | 128 | 1.2 |  |" in table
    assert "| ics_dequantized | 22.0000 | 0.9091 | pass | 128 | 1.4 |  |" in table


def test_parse_args_accepts_per_row_quant_method():
    with patch.object(sys, "argv", ["benchmark_perplexity.py", "--quant-method", "per_row_int4"]):
        args = parse_args()

    assert args.quant_method == "per_row_int4"


def test_parse_max_chains_accepts_all_for_full_pipeline():
    assert parse_max_chains("all") is None
    assert parse_max_chains("none") is None
    assert parse_max_chains("8") == 8


def test_parse_args_accepts_all_max_chains():
    with patch.object(sys, "argv", ["benchmark_perplexity.py", "--max-chains", "all"]):
        args = parse_args()

    assert args.max_chains is None


def test_parse_args_accepts_gptq_tuning_flags():
    with patch.object(
        sys,
        "argv",
        [
            "benchmark_perplexity.py",
            "--gptq-group-size",
            "64",
            "--gptq-percdamp",
            "0.05",
            "--gptq-blocksize",
            "64",
        ],
    ):
        args = parse_args()

    assert args.gptq_group_size == 64
    assert args.gptq_percdamp == 0.05
    assert args.gptq_blocksize == 64


def main() -> int:
    tests = [
        test_perplexity_from_nll,
        test_parse_llama_perplexity_final_estimate,
        test_parse_llama_perplexity_simple_label,
        test_format_results_table_marks_unavailable,
        test_summarize_quality_scores_candidates_relative_to_baseline,
        test_summarize_quality_keeps_unavailable_candidates_out_of_gate,
        test_format_results_table_shows_quality_status,
        test_parse_args_accepts_per_row_quant_method,
        test_parse_max_chains_accepts_all_for_full_pipeline,
        test_parse_args_accepts_all_max_chains,
        test_parse_args_accepts_gptq_tuning_flags,
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
