"""Sequential benchmark matrix runner for local ICS methods.

This orchestrates `scripts/benchmark_perplexity.py` once per method and
packages the results into a small publishable folder. Heavy model artifacts stay
under the scratch directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

from scripts.chain_limits import format_max_chains, parse_max_chains


ALLOWED_METHODS = ("block", "gptq", "per_row_int4")
PUBLISHABLE_FILES = ["RESULTS.md", "manifest.json", "package_manifest.json"]


def parse_methods(value: str) -> list[str]:
    methods = [part.strip() for part in value.split(",") if part.strip()]
    if not methods:
        raise ValueError("at least one method is required")
    unknown = [method for method in methods if method not in ALLOWED_METHODS]
    if unknown:
        raise ValueError(f"unknown method(s): {', '.join(unknown)}")
    return methods


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def file_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _numel(shape: list[int] | tuple[int, ...]) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return n


def _parse_load_note(note: str | None) -> dict[str, int | None]:
    if not note:
        return {"dequantized_tensors_loaded": None, "base_tensors_unchanged": None}
    match = re.search(r"(\d+)\s+dequantized tensors loaded;\s+(\d+)\s+base tensors unchanged", note)
    if not match:
        return {"dequantized_tensors_loaded": None, "base_tensors_unchanged": None}
    return {
        "dequantized_tensors_loaded": int(match.group(1)),
        "base_tensors_unchanged": int(match.group(2)),
    }


def artifact_stats(artifact_dir: str | Path, load_note: str | None = None) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    meta_path = artifact_dir / "ics_meta.json"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    layers = meta.get("layers") if isinstance(meta.get("layers"), dict) else {}
    dense_params = 0
    for info in layers.values():
        shape = info.get("original_shape") if isinstance(info, dict) else None
        if shape:
            dense_params += _numel(shape)

    total_bytes = file_bytes(artifact_dir)
    stats: dict[str, Any] = {
        "artifact_dir": str(artifact_dir),
        "artifact_bytes": total_bytes,
        "artifact_mib": total_bytes / (1024 * 1024) if total_bytes else 0.0,
        "quantized_tensor_count": len(layers),
        "dense_parameter_count": dense_params,
        "estimated_bpw": (total_bytes * 8.0 / dense_params) if dense_params else None,
        "quant_method": meta.get("quant_method"),
        "erc_enabled": meta.get("erc_enabled"),
        "erc_max_relative_error": meta.get("erc_max_relative_error"),
    }
    stats.update(_parse_load_note(load_note))
    return stats


def git_info() -> dict[str, Any]:
    commit = _git_output(["git", "rev-parse", "HEAD"])
    status = _git_output(["git", "status", "--short"])
    lines = [line for line in status.splitlines() if line.strip()]
    return {
        "commit": commit.strip() or None,
        "dirty": bool(lines),
        "changed_files": parse_changed_files(status),
    }


def parse_changed_files(status_short: str) -> list[str]:
    files: list[str] = []
    for line in status_short.splitlines():
        if not line.strip():
            continue
        if line.startswith("?? "):
            files.append(line[3:].strip())
        elif len(line) >= 3:
            files.append(line[2:].strip())
        else:
            files.append(line.strip())
    return files


def _git_output(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def classify_eval(eval_text_file: str | None) -> str:
    if not eval_text_file:
        return "smoke"
    name = Path(eval_text_file).name.lower()
    if any(token in name for token in ("wikitext", "c4", "dataset")):
        return "dataset"
    return "fixed-text"


def _result_by_name(payload: dict[str, Any], name: str) -> dict[str, Any] | None:
    for result in payload.get("results", []):
        if result.get("name") == name:
            return result
    return None


def _quality_for(payload: dict[str, Any], name: str) -> dict[str, Any]:
    quality = payload.get("quality_summary", {})
    candidates = quality.get("candidates", {}) if isinstance(quality, dict) else {}
    entry = candidates.get(name) if isinstance(candidates, dict) else None
    return entry if isinstance(entry, dict) else {}


def run_method(args: argparse.Namespace, method: str, output_dir: Path, scratch_dir: Path) -> dict[str, Any]:
    runs_dir = output_dir / "runs"
    artifacts_dir = scratch_dir / "artifacts"
    runs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    run_json = runs_dir / f"qwen06_{method}.json"
    artifact_dir = artifacts_dir / f"qwen06_{method}"

    cmd = [
        sys.executable,
        "scripts/benchmark_perplexity.py",
        "--model",
        args.model,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--quant-method",
        method,
        "--ics-output",
        str(artifact_dir),
        "--output-json",
        str(run_json),
        "--max-eval-tokens",
        str(args.max_eval_tokens),
        "--max-calibration-samples",
        str(args.max_calibration_samples),
        "--max-calibration-length",
        str(args.max_calibration_length),
        "--max-chains",
        format_max_chains(args.max_chains),
        "--gptq-group-size",
        str(args.gptq_group_size),
        "--gptq-percdamp",
        str(args.gptq_percdamp),
        "--gptq-blocksize",
        str(args.gptq_blocksize),
        "--min-quality-score",
        str(args.min_quality_score),
        "--rebuild-ics",
    ]
    if args.local_files_only:
        cmd.append("--local-files-only")
    if args.offline:
        cmd.append("--offline")
    if args.eval_text_file:
        cmd.extend(["--eval-text-file", args.eval_text_file])
    if args.q4_gguf:
        cmd.extend(["--q4-gguf", args.q4_gguf])

    t0 = time.time()
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    elapsed = time.time() - t0

    process_error = None if proc.returncode == 0 else f"benchmark process exited with code {proc.returncode}"
    if run_json.exists():
        payload = json.loads(run_json.read_text(encoding="utf-8"))
    else:
        payload = {
            "model": args.model,
            "results": [],
            "quality_summary": {},
            "error": process_error or "benchmark_perplexity.py did not write output JSON",
        }

    baseline = _result_by_name(payload, args.dtype) or {}
    candidate = _result_by_name(payload, "ics_dequantized") or {}
    q4 = _result_by_name(payload, "q4_k_m") or {}
    quality = _quality_for(payload, "ics_dequantized")
    if process_error and not candidate:
        candidate = {"error": process_error}
    note = candidate.get("error")

    return {
        "method": method,
        "command": cmd,
        "returncode": proc.returncode,
        "seconds": elapsed,
        "run_json": str(run_json),
        "artifact_dir": str(artifact_dir),
        "baseline": baseline,
        "candidate": candidate,
        "q4_k_m": q4,
        "quality": quality,
        "artifact": artifact_stats(artifact_dir, note),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def generate_results_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# ICS Benchmark Results",
        "",
        f"- Model: `{manifest.get('model')}`",
        f"- Evaluation label: `{manifest.get('eval_label')}`",
        f"- Git commit: `{(manifest.get('git') or {}).get('commit')}`",
        f"- Dirty worktree: `{(manifest.get('git') or {}).get('dirty')}`",
        "",
        "| method | baseline PPL | ICS PPL | quality | status | artifact MiB | est. BPW | tensors | base unchanged | note |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for run in manifest.get("runs", []):
        baseline = run.get("baseline") or {}
        candidate = run.get("candidate") or {}
        quality = run.get("quality") or {}
        artifact = run.get("artifact") or {}
        lines.append(
            "| {method} | {baseline_ppl} | {candidate_ppl} | {quality_score} | {status} | "
            "{artifact_mib} | {bpw} | {tensors} | {base_unchanged} | {note} |".format(
                method=run.get("method", ""),
                baseline_ppl=_fmt_float(baseline.get("perplexity")),
                candidate_ppl=_fmt_float(candidate.get("perplexity")),
                quality_score=_fmt_float(quality.get("quality_score")),
                status=quality.get("status", ""),
                artifact_mib=_fmt_float(artifact.get("artifact_mib")),
                bpw=_fmt_float(artifact.get("estimated_bpw")),
                tensors=artifact.get("quantized_tensor_count", ""),
                base_unchanged=artifact.get("base_tensors_unchanged", ""),
                note=(candidate.get("error") or quality.get("note") or "").replace("|", "/"),
            )
        )

    lines.extend(["", "## Caveats", ""])
    for caveat in manifest.get("caveats", []):
        lines.append(f"- {caveat}")
    lines.extend(["", "## Commands", ""])
    for run in manifest.get("runs", []):
        lines.append(f"### {run.get('method')}")
        lines.append("")
        lines.append("```powershell")
        lines.append(" ".join(str(part) for part in run.get("command", [])))
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _fmt_float(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def build_package_manifest(manifest: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_dir": str(output_dir),
        "publishable_files": PUBLISHABLE_FILES.copy(),
        "model": manifest.get("model"),
        "methods": manifest.get("methods", []),
        "git": manifest.get("git", {}),
        "runs": manifest.get("runs", []),
    }


def build_upload_plan(
    package_manifest: dict[str, Any],
    hf_repo: str | None,
    git_dir: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "dry_run": dry_run,
        "hf": {
            "repo": hf_repo,
            "action": "skip_upload" if dry_run else ("upload_folder" if hf_repo else "not_configured"),
            "source_dir": package_manifest.get("output_dir"),
        },
        "git": {
            "path": git_dir,
            "action": "write_only" if dry_run else ("ready_to_stage" if git_dir else "not_configured"),
            "publishable_files": package_manifest.get("publishable_files", []),
        },
    }


def write_outputs(
    output_dir: Path,
    scratch_dir: Path,
    manifest: dict[str, Any],
    package_manifest: dict[str, Any],
    upload_plan: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    package_manifest = dict(package_manifest)
    package_manifest["upload_plan"] = upload_plan

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "RESULTS.md").write_text(generate_results_markdown(manifest), encoding="utf-8")
    (output_dir / "package_manifest.json").write_text(json.dumps(package_manifest, indent=2), encoding="utf-8")

    package_dir = scratch_dir / "packages"
    package_dir.mkdir(parents=True, exist_ok=True)
    archive = package_dir / f"{output_dir.name}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in PUBLISHABLE_FILES:
            zf.write(output_dir / fname, arcname=fname)
    return archive


def maybe_upload_package(package_manifest: dict[str, Any], upload_plan: dict[str, Any]) -> dict[str, Any]:
    if upload_plan.get("dry_run"):
        return {"ok": None, "dry_run": True, "message": "upload skipped"}
    hf = upload_plan.get("hf", {})
    repo = hf.get("repo")
    if not repo:
        return {"ok": None, "message": "no HF repo configured"}
    token = os.environ.get("HF_TOKEN")
    if not token:
        return {"ok": False, "error": "HF_TOKEN must be set for non-dry-run upload"}
    from huggingface_hub import HfApi  # type: ignore

    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(package_manifest["output_dir"]),
        repo_id=str(repo),
        repo_type="dataset",
        commit_message="Upload ICS benchmark harness package",
        token=token,
    )
    return {"ok": True, "repo": repo}


def build_manifest(args: argparse.Namespace, methods: list[str], runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": args.model,
        "methods": methods,
        "dtype": args.dtype,
        "device": args.device,
        "eval_text_file": args.eval_text_file,
        "eval_label": classify_eval(args.eval_text_file),
        "max_eval_tokens": args.max_eval_tokens,
        "max_calibration_samples": args.max_calibration_samples,
        "max_calibration_length": args.max_calibration_length,
        "max_chains": args.max_chains,
        "gptq_group_size": args.gptq_group_size,
        "gptq_percdamp": args.gptq_percdamp,
        "gptq_blocksize": args.gptq_blocksize,
        "min_quality_score": args.min_quality_score,
        "git": git_info(),
        "runs": runs,
        "caveats": [
            "PPL is the measured quantity; acceptance uses baseline-relative quality.",
            "Built-in eval text is a smoke check, not paper-grade evidence.",
            "gptq is the local ICS GPTQ backend, not a new GPTQ algorithm claim.",
            "External AWQ/RPTQ/QuaRot/SpinQuant baselines are intentionally out of v1.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sequential ICS benchmark matrix")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--methods", default="block,gptq,per_row_int4")
    parser.add_argument("--output-dir", default="benchmarks/qwen3_06b_harness")
    parser.add_argument("--scratch-dir", default=".pytest_cache/ics_harness")
    parser.add_argument("--eval-text-file")
    parser.add_argument("--max-eval-tokens", type=int, default=128)
    parser.add_argument("--max-calibration-samples", type=int, default=1)
    parser.add_argument("--max-calibration-length", type=int, default=8)
    parser.add_argument("--max-chains", type=parse_max_chains, default=1)
    parser.add_argument("--gptq-group-size", type=int, default=128)
    parser.add_argument("--gptq-percdamp", type=float, default=0.01)
    parser.add_argument("--gptq-blocksize", type=int, default=128)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--min-quality-score", type=float, default=0.90)
    parser.add_argument("--q4-gguf")
    parser.add_argument("--package-hf-repo")
    parser.add_argument("--package-git-dir")
    parser.add_argument("--dry-run-upload", type=parse_bool, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    methods = parse_methods(args.methods)
    output_dir = Path(args.output_dir)
    scratch_dir = Path(args.scratch_dir)

    runs: list[dict[str, Any]] = []
    for method in methods:
        print(f"\n=== benchmark method: {method} ===", flush=True)
        run = run_method(args, method, output_dir, scratch_dir)
        runs.append(run)
        if run["returncode"] != 0:
            print(f"[warn] {method} returned {run['returncode']}", flush=True)

    manifest = build_manifest(args, methods, runs)
    package_manifest = build_package_manifest(manifest, output_dir)
    upload_plan = build_upload_plan(
        package_manifest,
        hf_repo=args.package_hf_repo,
        git_dir=args.package_git_dir or str(output_dir),
        dry_run=args.dry_run_upload,
    )
    archive = write_outputs(output_dir, scratch_dir, manifest, package_manifest, upload_plan)
    upload_result = maybe_upload_package(package_manifest, upload_plan)

    print(f"\nWrote {output_dir / 'manifest.json'}")
    print(f"Wrote {output_dir / 'RESULTS.md'}")
    print(f"Wrote {output_dir / 'package_manifest.json'}")
    print(f"Wrote package archive {archive}")
    print(f"Upload result: {json.dumps(upload_result)}")
    return 0 if all(run["returncode"] == 0 for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
