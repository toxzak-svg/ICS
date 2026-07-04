"""Run a CPU-friendly ICS smoke pipeline on Qwen3-0.6B.

This is intended for local validation. It can run against a cached Hugging Face
snapshot without remote metadata calls, and defaults to one calibration sample
and one discovered chain so CPU-only machines can exercise the real pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ics.export import save_ics_model
from ics.pipeline import ICSConfig, discover_chains, quantize_model
from scripts.chain_limits import parse_max_chains


DEFAULT_CALIBRATION = [
    "Qwen3 is a causal language model. This short sample calibrates one ICS smoke run.",
    "Quantization groups important activation channels into higher precision blocks.",
]


class SyntheticTokenizer:
    """Deterministic calibration tokenizer for cache-only smoke runs."""

    pad_token = "<synthetic-pad>"
    eos_token = "<synthetic-eos>"
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self, vocab_size: int):
        self.vocab_size = max(vocab_size, 2)

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=512):
        limit = max(max_length, 2)
        raw = text.encode("utf-8") or b"ics"
        ids = [1 + (b % (self.vocab_size - 1)) for b in raw[:limit]]
        if len(ids) < 2:
            ids.extend([self.eos_token_id] * (2 - len(ids)))
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def save_pretrained(self, output_dir: str):
        path = Path(output_dir) / "synthetic_tokenizer.txt"
        path.write_text(
            "Synthetic tokenizer used for ICS smoke calibration only.\n",
            encoding="utf-8",
        )


def _default_hf_cache() -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub"


def resolve_model_path(model: str, local_files_only: bool) -> str:
    path = Path(model)
    if path.exists():
        return str(path)
    if not local_files_only:
        return model

    cache_name = "models--" + model.replace("/", "--")
    snapshots = _default_hf_cache() / cache_name / "snapshots"
    if not snapshots.exists():
        raise FileNotFoundError(
            f"No cached snapshots found for {model!r} under {snapshots}. "
            "Run without --local-files-only once to populate the cache."
        )
    candidates = [p for p in snapshots.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No snapshot directories found under {snapshots}")
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ICS on Qwen3-0.6B locally")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", default="./qwen3-0.6b-ics-smoke")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--offline", action="store_true", help="set HF offline env vars before loading")
    parser.add_argument("--max-calibration-samples", type=int, default=1)
    parser.add_argument("--max-calibration-length", type=int, default=32)
    parser.add_argument("--max-chains", type=parse_max_chains, default=1)
    parser.add_argument(
        "--fisher-loss-mode",
        default="last_logit_mean",
        choices=["cross_entropy", "last_logit_mean"],
    )
    parser.add_argument(
        "--chain-filter",
        action="append",
        default=[],
        help="substring filter for selected chain members; may be repeated",
    )
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument(
        "--synthetic-tokenizer",
        action="store_true",
        help="use deterministic synthetic token IDs for calibration",
    )
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]

    model_path = resolve_model_path(args.model, args.local_files_only)
    print(f"[load] model source: {model_path}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=args.local_files_only,
    )
    model.to(args.device)
    print(f"[load] model ready in {time.time() - t0:.1f}s", flush=True)

    tokenizer = None
    if not args.synthetic_tokenizer:
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=args.local_files_only)
        probe = tokenizer("tokenizer probe", return_tensors="pt")
        if probe["input_ids"].shape[-1] < 2:
            print("[load] tokenizer produced empty output; using synthetic calibration tokens", flush=True)
            tokenizer = None
        else:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"[load] tokenizer ready in {time.time() - t0:.1f}s", flush=True)
    if tokenizer is None:
        tokenizer = SyntheticTokenizer(model.config.vocab_size)

    chains = discover_chains(model, skip_modules=ICSConfig().skip_modules)
    print(f"[inspect] discovered chains: {len(chains)}", flush=True)
    for chain in chains[: min(4, len(chains))]:
        print(f"[inspect] {chain.kind}: {chain.members}", flush=True)

    calibration = DEFAULT_CALIBRATION[: args.max_calibration_samples]
    config = ICSConfig(
        block_size=args.block_size,
        calibration_texts=calibration,
        max_calibration_length=args.max_calibration_length,
        chain_name_filters=tuple(args.chain_filter),
        max_chains=args.max_chains,
        fisher_loss_mode=args.fisher_loss_mode,
    )

    t0 = time.time()
    result = quantize_model(model, tokenizer, config, device=args.device, show_progress=True)
    print(f"[pipeline] completed in {time.time() - t0:.1f}s", flush=True)
    print(f"[pipeline] permutations: {len(result.perms)}", flush=True)
    print(f"[pipeline] quantized layers: {len(result.quant)}", flush=True)

    if not args.no_save:
        print(f"[save] writing {args.output}", flush=True)
        save_ics_model(result, args.output, tokenizer=tokenizer)
        print("[save] done", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
