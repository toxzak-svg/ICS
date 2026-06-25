"""Perplexity benchmark harness for Qwen3-0.6B.

Compares:
    - HF dense baseline (fp16/bf16/fp32 by argument)
    - optional ICS-dequantized weights loaded into the HF model
    - optional llama.cpp GGUF Q4_K_M via llama-perplexity

Defaults are intentionally small so a CPU-only workstation can run the harness.
Use larger --max-eval-tokens values and a dataset file for reportable numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ics.export import dequantized_state_dict, load_ics_model
from ics.pipeline import ICSConfig, quantize_model
from scripts.run_qwen3_06b_pipeline import DEFAULT_CALIBRATION, resolve_model_path


DEFAULT_EVAL_TEXTS = [
    "Transformers use attention to route information between token positions.",
    "Quantization reduces model size by representing weights with fewer bits.",
    "A small benchmark should be deterministic, repeatable, and explicit about its limits.",
    "The model predicts the next token from the previous context.",
]


@dataclass
class BenchmarkResult:
    name: str
    perplexity: float | None
    token_count: int
    seconds: float
    error: str | None = None


def perplexity_from_nll(total_nll: float, token_count: int) -> float:
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    return math.exp(total_nll / token_count)


def parse_llama_perplexity(output: str) -> float:
    patterns = [
        r"Final estimate:\s*PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        r"\bPPL\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        r"\bperplexity:\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, output, flags=re.IGNORECASE)
        if matches:
            return float(matches[-1])
    raise ValueError("could not parse llama.cpp perplexity from output")


def format_results_table(results: Sequence[BenchmarkResult]) -> str:
    lines = [
        "| model | perplexity | tokens | seconds | note |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        ppl = "unavailable" if result.perplexity is None else f"{result.perplexity:.4f}"
        note = result.error or ""
        lines.append(
            f"| {result.name} | {ppl} | {result.token_count} | "
            f"{result.seconds:.1f} | {note} |"
        )
    return "\n".join(lines)


def load_eval_text(args: argparse.Namespace) -> str:
    if args.eval_text_file:
        text = Path(args.eval_text_file).read_text(encoding="utf-8")
    else:
        text = "\n\n".join(DEFAULT_EVAL_TEXTS)
    return text


def tokenize_eval_text(tokenizer, text: str, max_eval_tokens: int, device: str) -> torch.Tensor:
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = enc["input_ids"][0]
    if max_eval_tokens > 0:
        input_ids = input_ids[:max_eval_tokens]
    if input_ids.numel() < 2:
        raise ValueError("evaluation text must tokenize to at least 2 tokens")
    return input_ids.unsqueeze(0).to(device)


@torch.no_grad()
def hf_perplexity(model, input_ids: torch.Tensor, block_size: int) -> tuple[float, int]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    seq_len = input_ids.shape[1]
    for start in range(0, seq_len - 1, block_size):
        end = min(start + block_size, seq_len)
        if end - start < 2:
            continue
        chunk = input_ids[:, start:end]
        out = model(input_ids=chunk, use_cache=False)
        logits = out.logits[:, :-1, :].float()
        labels = chunk[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="sum",
        )
        total_nll += float(loss.item())
        total_tokens += int(labels.numel())
    return perplexity_from_nll(total_nll, total_tokens), total_tokens


def load_hf_model(model_path: str, dtype: torch.dtype, device: str, local_files_only: bool):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=local_files_only,
    )
    return model.to(device)


def benchmark_hf_dense(args: argparse.Namespace, model_path: str, tokenizer, input_ids: torch.Tensor) -> BenchmarkResult:
    t0 = time.time()
    try:
        model = load_hf_model(model_path, _dtype_from_arg(args.dtype), args.device, args.local_files_only)
        ppl, tokens = hf_perplexity(model, input_ids, args.block_size)
        return BenchmarkResult(args.dtype, ppl, tokens, time.time() - t0)
    except Exception as exc:
        return BenchmarkResult(args.dtype, None, 0, time.time() - t0, str(exc))


def ensure_ics_output(args: argparse.Namespace, model_path: str, tokenizer) -> Path | None:
    if not args.ics_output:
        return None
    out_dir = Path(args.ics_output)
    if (out_dir / "ics_meta.json").exists() and not args.rebuild_ics:
        return out_dir

    model = load_hf_model(model_path, _dtype_from_arg(args.dtype), args.device, args.local_files_only)
    config = ICSConfig(
        block_size=args.ics_block_size,
        int4_fraction=args.int4_fraction,
        int2_fraction=args.int2_fraction,
        int1_fraction=args.int1_fraction,
        calibration_texts=DEFAULT_CALIBRATION[: args.max_calibration_samples],
        max_calibration_length=args.max_calibration_length,
        chain_name_filters=tuple(args.chain_filter),
        max_chains=args.max_chains,
        fisher_loss_mode=args.fisher_loss_mode,
    )
    result = quantize_model(model, tokenizer, config, device=args.device, show_progress=True)
    from ics.export import save_ics_model

    save_ics_model(result, out_dir, tokenizer=tokenizer)
    return out_dir


def benchmark_ics(args: argparse.Namespace, model_path: str, input_ids: torch.Tensor, ics_output: Path | None) -> BenchmarkResult:
    if ics_output is None:
        return BenchmarkResult("ics_dequantized", None, 0, 0.0, "no --ics-output provided")
    t0 = time.time()
    try:
        model = load_hf_model(model_path, _dtype_from_arg(args.dtype), args.device, args.local_files_only)
        loaded = load_ics_model(ics_output)
        state = dequantized_state_dict(loaded)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected ICS state keys: {unexpected[:3]}")
        ppl, tokens = hf_perplexity(model, input_ids, args.block_size)
        note = f"{len(state)} dequantized tensors loaded; {len(missing)} base tensors unchanged"
        return BenchmarkResult("ics_dequantized", ppl, tokens, time.time() - t0, note)
    except Exception as exc:
        return BenchmarkResult("ics_dequantized", None, 0, time.time() - t0, str(exc))


def benchmark_q4(args: argparse.Namespace, eval_text: str) -> BenchmarkResult:
    if not args.q4_gguf:
        return BenchmarkResult("q4_k_m", None, 0, 0.0, "no --q4-gguf provided")
    gguf = Path(args.q4_gguf)
    if not gguf.exists():
        return BenchmarkResult("q4_k_m", None, 0, 0.0, f"missing GGUF: {gguf}")

    t0 = time.time()
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as tmp:
        tmp.write(eval_text)
        tmp_path = Path(tmp.name)
    try:
        cmd = [
            args.llama_perplexity,
            "-m",
            str(gguf),
            "-f",
            str(tmp_path),
            "-c",
            str(args.llama_ctx),
            "-b",
            str(args.llama_batch),
            "--chunks",
            str(args.llama_chunks),
            "--no-warmup",
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            raise RuntimeError(output.strip()[-1000:])
        ppl = parse_llama_perplexity(output)
        token_count = max(args.llama_ctx * max(args.llama_chunks, 1), 0)
        return BenchmarkResult("q4_k_m", ppl, token_count, time.time() - t0)
    except Exception as exc:
        return BenchmarkResult("q4_k_m", None, 0, time.time() - t0, str(exc))
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _dtype_from_arg(dtype: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[dtype]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-0.6B perplexity")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--eval-text-file")
    parser.add_argument("--max-eval-tokens", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--output-json", default="perplexity_results.json")
    parser.add_argument("--ics-output", default="qwen3-0.6b-ics-bench")
    parser.add_argument("--rebuild-ics", action="store_true")
    parser.add_argument("--max-calibration-samples", type=int, default=1)
    parser.add_argument("--max-calibration-length", type=int, default=8)
    parser.add_argument("--max-chains", type=int, default=1)
    parser.add_argument("--chain-filter", action="append", default=[])
    parser.add_argument("--fisher-loss-mode", default="last_logit_mean", choices=["cross_entropy", "last_logit_mean"])
    parser.add_argument("--ics-block-size", type=int, default=64)
    parser.add_argument("--int4-fraction", type=float, default=0.5)
    parser.add_argument("--int2-fraction", type=float, default=0.4)
    parser.add_argument("--int1-fraction", type=float, default=0.1)
    parser.add_argument("--q4-gguf")
    parser.add_argument("--llama-perplexity", default="llama-perplexity")
    parser.add_argument("--llama-ctx", type=int, default=256)
    parser.add_argument("--llama-batch", type=int, default=128)
    parser.add_argument("--llama-chunks", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    model_path = resolve_model_path(args.model, args.local_files_only)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eval_text = load_eval_text(args)
    input_ids = tokenize_eval_text(tokenizer, eval_text, args.max_eval_tokens, args.device)

    results: list[BenchmarkResult] = []
    results.append(benchmark_hf_dense(args, model_path, tokenizer, input_ids))
    ics_output = ensure_ics_output(args, model_path, tokenizer)
    results.append(benchmark_ics(args, model_path, input_ids, ics_output))
    results.append(benchmark_q4(args, eval_text))

    table = format_results_table(results)
    print(table)

    payload = {
        "model": args.model,
        "model_path": model_path,
        "max_eval_tokens": args.max_eval_tokens,
        "block_size": args.block_size,
        "results": [asdict(r) for r in results],
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output_json}")
    return 0 if any(r.perplexity is not None for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
