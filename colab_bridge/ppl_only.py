"""PPL eval only — uses the existing artifact and uploads just the bench JSON."""
from __future__ import annotations
import os
import sys
import time
import json
import math
import shutil
import traceback
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

ICS_DIR = "/content/ICS"
ART_DIR = "/content/qwen3.5-2b-ics-gptq-int4"
LOG_PATH = Path("/content/ppl_only.log")
QWEN_DIR = "/content/Qwen3.5-2B"

sys.path.insert(0, ICS_DIR)
for root, dirs, files in os.walk(ICS_DIR):
    for d in dirs:
        if d == "__pycache__":
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)
for mod in list(sys.modules):
    if mod.startswith("ics.") or mod.startswith("scripts."):
        del sys.modules[mod]


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def dequantized_state_dict_patched(loaded):
    """Like ics.export.dequantized_state_dict but catches shape mismatches."""
    import torch
    from ics.quantize import dequantize_blockwise
    state = {}
    layer_perms = loaded.get("layer_perms", {})
    chain_members = loaded.get("chain_members", {})
    layer_chain_perm = {}
    for chain_key, info in chain_members.items():
        perm_list = info["permutation"]
        for member, target in zip(info["members"], info["targets"]):
            layer_chain_perm[member] = (target, perm_list)
    skipped = []
    for layer_name, qt in loaded["layers"].items():
        W = dequantize_blockwise(qt)
        if layer_name in layer_perms:
            gperm = torch.tensor(layer_perms[layer_name], dtype=torch.long, device=W.device)
            if W.shape[1] == len(gperm):
                W = W[:, gperm]
            else:
                log(f"  WARN: GPTQ perm shape mismatch on {layer_name}: W={tuple(W.shape)} perm={len(gperm)} — skipping")
        if layer_name in layer_chain_perm:
            target, cperm_list = layer_chain_perm[layer_name]
            cperm = torch.tensor(cperm_list, dtype=torch.long, device=W.device)
            ok = False
            if target == 0 and W.shape[0] == len(cperm):
                W = W[cperm]
                ok = True
            elif target == 1 and W.shape[1] == len(cperm):
                W = W[:, cperm]
                ok = True
            if not ok:
                log(f"  WARN: chain perm shape mismatch on {layer_name}: W={tuple(W.shape)} target={target} perm_len={len(cperm)} — skipping")
                skipped.append(layer_name)
        state[layer_name + ".weight"] = W
    log(f"  dequant done; {len(skipped)} layers had chain perm skipped")
    return state


def compute_perplexity(model, tokenizer, texts, block_size=256, device="cuda"):
    import torch
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=False)
            input_ids = enc["input_ids"].to(device)
            if input_ids.numel() < 2:
                continue
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
    if total_tokens == 0:
        return float("nan"), 0
    return math.exp(total_nll / total_tokens), total_tokens


def main():
    LOG_PATH.write_text("")
    log("===== Qwen3.5-2B ICS+GPTQ+INT4 PPL eval =====")
    import torch
    log(f"VRAM free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

    log("STEP 1: load ICS artifact + dequant")
    from ics.export import load_ics_model, dequantized_state_dict
    t0 = time.time()
    loaded = load_ics_model(ART_DIR)
    log(f"  loaded in {time.time()-t0:.1f}s")
    t0 = time.time()
    sd = dequantized_state_dict(loaded)
    log(f"  dequantized {len(sd)} tensors in {time.time()-t0:.1f}s")
    del loaded
    torch.cuda.empty_cache()

    log("STEP 2: load BF16 model")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    bf16_model = AutoModelForCausalLM.from_pretrained(QWEN_DIR, dtype=torch.bfloat16, device_map="cuda")
    tok = AutoTokenizer.from_pretrained(QWEN_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    log(f"  loaded; layers={len(bf16_model.model.layers)}")

    log("STEP 3: assign dequantized weights")
    missing = bf16_model.load_state_dict(sd, strict=False)
    log(f"  load_state_dict: missing={len(missing.missing_keys)} unexpected={len(missing.unexpected_keys)}")
    if missing.missing_keys:
        log(f"  first 3 missing: {missing.missing_keys[:3]}")
    if missing.unexpected_keys:
        log(f"  first 3 unexpected: {missing.unexpected_keys[:3]}")
    del sd
    torch.cuda.empty_cache()

    log("STEP 4: PPL eval on wikitext-style text")
    EVAL_TEXTS = [
        "Transformers use attention to route information between token positions in a sequence. The model learns to predict the next token from the previous context, and this next-token prediction objective is the basis for most modern language modeling.",
        "Quantization reduces model size by representing weights with fewer bits. The challenge is to preserve accuracy while using lower precision arithmetic. Mixed-precision quantization, where sensitive weights are kept at higher precision, is one approach.",
        "The scientific method involves forming hypotheses, designing experiments to test them, and revising theories based on evidence. In machine learning research, the cycle of benchmark design, model training, and ablation study mirrors this process.",
        "The history of computing traces a path from mechanical calculators to electronic digital computers. Each generation of hardware has enabled new classes of algorithms, from numerical methods to deep learning.",
        "A small benchmark should be deterministic, repeatable, and explicit about its limits. Reporting token count and wall-clock time alongside perplexity is essential for fair comparison.",
    ]
    t0 = time.time()
    ppl, n_tok = compute_perplexity(bf16_model, tok, EVAL_TEXTS, block_size=256)
    log(f"  PPL: {ppl:.4f} on {n_tok} tokens in {time.time()-t0:.1f}s")

    bench = {
        "model": "Qwen/Qwen3.5-2B",
        "method": "ICS+GPTQ+INT4 (dequantized for PPL)",
        "ppl": ppl,
        "eval_tokens": n_tok,
        "eval_texts": len(EVAL_TEXTS),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    Path("/content/ppl_results.json").write_text(json.dumps(bench, indent=2))

    log("STEP 5: upload bench + log to dataset repo (file-by-file)")
    bench_dir = "/content/_bench_stage"
    Path(bench_dir).mkdir(exist_ok=True)
    Path(bench_dir, "ppl_results.json").write_text(json.dumps(bench, indent=2))
    shutil.copy2("/content/pipeline.log", Path(bench_dir, "pipeline.log")) if Path("/content/pipeline.log").exists() else None
    if Path("/content/recovery.log").exists():
        shutil.copy2("/content/recovery.log", Path(bench_dir, "recovery.log"))
    shutil.copy2(str(LOG_PATH), Path(bench_dir, "ppl_only.log"))
    if Path("/content/recovery2.log").exists():
        shutil.copy2("/content/recovery2.log", Path(bench_dir, "recovery2.log"))

    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    bench_files = sorted(Path(bench_dir).iterdir())
    for f in bench_files:
        if not f.is_file():
            continue
        sz = f.stat().st_size
        log(f"  uploading {f.name} ({sz/1e3:.1f} KB)")
        t0 = time.time()
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id="toxzak/ics-quantization-artifacts",
            repo_type="dataset",
            commit_message=f"qwen3.5-2b bench: {f.name}",
            token=os.environ["HF_TOKEN"],
        )
        log(f"    OK in {time.time()-t0:.1f}s")

    log("===== PPL EVAL DONE =====")
    log(f"  PPL: {ppl:.4f}")
    log(f"  bench: https://huggingface.co/datasets/toxzak/ics-quantization-artifacts")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        log(traceback.format_exc())
        sys.exit(1)