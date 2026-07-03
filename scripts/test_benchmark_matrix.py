"""Tests for the benchmark matrix harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_matrix import (
    artifact_stats,
    build_package_manifest,
    build_upload_plan,
    generate_results_markdown,
    parse_changed_files,
    parse_methods,
)


def test_parse_methods_accepts_known_methods():
    assert parse_methods("block,gptq,per_row_int4") == ["block", "gptq", "per_row_int4"]


def test_parse_methods_rejects_unknown_method():
    try:
        parse_methods("block,awq")
    except ValueError as exc:
        assert "unknown method" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_artifact_stats_reads_tiny_ics_artifact(tmp_path: Path):
    art = tmp_path / "artifact"
    art.mkdir()
    (art / "model.safetensors").write_bytes(b"12345678")
    (art / "scales.safetensors").write_bytes(b"1234")
    (art / "ics_meta.json").write_text(
        json.dumps(
            {
                "quant_method": "per_row_int4",
                "erc_enabled": True,
                "erc_max_relative_error": 0.1,
                "layers": {
                    "a.weight": {"original_shape": [2, 3]},
                    "b.weight": {"original_shape": [4, 5]},
                },
            }
        ),
        encoding="utf-8",
    )

    stats = artifact_stats(art, "3 dequantized tensors loaded; 308 base tensors unchanged")

    assert stats["artifact_bytes"] > 12
    assert stats["quantized_tensor_count"] == 2
    assert stats["dense_parameter_count"] == 26
    assert stats["estimated_bpw"] > 0
    assert stats["quant_method"] == "per_row_int4"
    assert stats["erc_enabled"] is True
    assert stats["base_tensors_unchanged"] == 308


def test_generate_results_markdown_labels_smoke_and_methods():
    manifest = {
        "model": "Qwen/Qwen3-0.6B",
        "eval_label": "smoke",
        "runs": [
            {
                "method": "per_row_int4",
                "baseline": {"perplexity": 10.0},
                "candidate": {"perplexity": 12.0},
                "quality": {"quality_score": 0.8333, "status": "fail"},
                "artifact": {"artifact_bytes": 100, "estimated_bpw": 4.2},
            }
        ],
        "caveats": ["Not a paper-grade dataset run."],
    }

    markdown = generate_results_markdown(manifest)

    assert "# ICS Benchmark Results" in markdown
    assert "Qwen/Qwen3-0.6B" in markdown
    assert "smoke" in markdown
    assert "per_row_int4" in markdown
    assert "Not a paper-grade dataset run." in markdown


def test_generate_results_markdown_surfaces_failed_process_note():
    manifest = {
        "model": "Qwen/Qwen3-0.6B",
        "eval_label": "smoke",
        "git": {"commit": "abc123", "dirty": True},
        "runs": [
            {
                "method": "per_row_int4",
                "returncode": 3221225477,
                "baseline": {},
                "candidate": {"error": "benchmark process exited with code 3221225477"},
                "quality": {},
                "artifact": {"artifact_bytes": 0, "estimated_bpw": None},
            }
        ],
        "caveats": [],
    }

    markdown = generate_results_markdown(manifest)

    assert "benchmark process exited with code 3221225477" in markdown


def test_parse_changed_files_handles_short_status_spacing():
    status = " M README.md\n?? scripts/benchmark_matrix.py\nM  pyproject.toml\n"

    assert parse_changed_files(status) == [
        "README.md",
        "scripts/benchmark_matrix.py",
        "pyproject.toml",
    ]


def test_package_manifest_includes_dirty_status_and_artifacts(tmp_path: Path):
    manifest = {
        "model": "Qwen/Qwen3-0.6B",
        "methods": ["per_row_int4"],
        "git": {"commit": "abc123", "dirty": True, "changed_files": ["README.md"]},
        "runs": [
            {"method": "per_row_int4", "artifact": {"artifact_bytes": 100, "estimated_bpw": 4.0}},
        ],
    }

    package = build_package_manifest(manifest, tmp_path)

    assert package["git"]["dirty"] is True
    assert package["git"]["changed_files"] == ["README.md"]
    assert package["publishable_files"] == ["RESULTS.md", "manifest.json", "package_manifest.json"]
    assert package["runs"][0]["artifact"]["artifact_bytes"] == 100


def test_upload_dry_run_creates_plan_without_network_calls(tmp_path: Path):
    package = {"output_dir": str(tmp_path), "publishable_files": ["RESULTS.md"]}

    plan = build_upload_plan(
        package,
        hf_repo="toxzak/test",
        git_dir="benchmarks/qwen3_06b_harness",
        dry_run=True,
    )

    assert plan["dry_run"] is True
    assert plan["hf"]["action"] == "skip_upload"
    assert plan["hf"]["repo"] == "toxzak/test"
    assert plan["git"]["action"] == "write_only"
    assert plan["git"]["path"] == "benchmarks/qwen3_06b_harness"


def main() -> int:
    tests = [
        test_parse_methods_accepts_known_methods,
        test_parse_methods_rejects_unknown_method,
        test_artifact_stats_reads_tiny_ics_artifact,
        test_generate_results_markdown_labels_smoke_and_methods,
        test_generate_results_markdown_surfaces_failed_process_note,
        test_parse_changed_files_handles_short_status_spacing,
        test_package_manifest_includes_dirty_status_and_artifacts,
        test_upload_dry_run_creates_plan_without_network_calls,
    ]
    failures = []
    import tempfile

    for test in tests:
        try:
            if "tmp_path" in test.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as d:
                    test(Path(d))
            else:
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
