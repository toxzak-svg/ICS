"""Run the perplexity benchmark: BF16 dense vs ICS int4 vs Q4_K_M GGUF."""
import sys, time
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=2400)

# The benchmark script handles BF16 dense + ICS-dequant; we'll run the GGUF separately.
script = r'''
import sys, os, time, json
sys.path.insert(0, "/content/ICS")
os.environ["PYTHONUNBUFFERED"] = "1"

# 1) BF16 dense baseline + ICS-dequant PPL
from ics.export import load_ics_model, dequantized_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("/content/Qwen3-0.6B")
texts = [
    "The transformer architecture uses self-attention to model long-range dependencies in sequences.",
    "Quantization maps continuous values to a discrete grid; per-block scaling factors preserve precision.",
    "In 1969, the Apollo 11 mission landed the first humans on the Moon, marking a milestone in human history.",
    "The Fisher information matrix measures how sensitive the model loss is to perturbations in parameters.",
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
    "Climate change is driving more frequent extreme weather events across the globe.",
    "Neural networks trained with backpropagation learn hierarchical features automatically.",
    "Renewable energy sources include solar, wind, hydro, and geothermal power generation.",
    "The human brain contains roughly 86 billion neurons connected by trillions of synapses.",
    "DNA encodes genetic information using four nucleotide bases: adenine, thymine, guanine, and cytosine.",
]
joined = "\n\n".join(texts)
enc = tok(joined, return_tensors="pt")
input_ids = enc["input_ids"]
print(f"eval tokens: {input_ids.shape[-1]}")

def ppl_blockwise(model, ids, block=128):
    model.eval()
    nll = 0.0
    n_tok = 0
    with torch.no_grad():
        for i in range(0, ids.shape[-1] - 1, block):
            chunk = ids[:, i:i+block+1].to(model.device)
            out = model(input_ids=chunk, use_cache=False)
            logits = out.logits[:, :-1, :].float()
            targets = chunk[:, 1:]
            ll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="sum",
            )
            nll += ll.item()
            n_tok += targets.numel()
    return float(torch.tensor(nll / n_tok).exp().item()), n_tok

# BF16 dense baseline
print("\n=== BF16 dense baseline ===")
t0 = time.time()
m = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", dtype=torch.bfloat16, device_map="cuda")
p, n = ppl_blockwise(m, input_ids)
print(f"BF16 PPL: {p:.3f}  ({n} tokens, {time.time()-t0:.1f}s)")
del m
torch.cuda.empty_cache()
bf16_ppl, bf16_n = p, n

# ICS int4 dequant
print("\n=== ICS int4 (fresh) ===")
t0 = time.time()
loaded = load_ics_model("/content/qwen3-0.6b-ics-fresh-int4")
sd = dequantized_state_dict(loaded)
m = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", dtype=torch.bfloat16, device_map="cuda")
missing, unexpected = m.load_state_dict(sd, strict=False)
print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
# Only count the quantized layers
n_quant = len(loaded["layers"])
print(f"loaded {n_quant} dequantized layers into BF16 model")
p, n = ppl_blockwise(m, input_ids)
print(f"ICS int4 PPL: {p:.3f}  ({n} tokens, {time.time()-t0:.1f}s)")
del m
torch.cuda.empty_cache()
ics_ppl, ics_n = p, n

# 2) llama.cpp Q4_K_M via direct subprocess (since we installed llama-cpp-python)
print("\n=== Q4_K_M GGUF via llama.cpp ===")
gguf_path = "/content/gguf/Qwen3-0.6B-Q4_K_M.gguf"
from llama_cpp import Llama
t0 = time.time()
llm = Llama(model_path=gguf_path, n_ctx=512, n_threads=8, verbose=False)
# Compute PPL using logprobs
total_nll = 0.0
total_n = 0
for text in texts:
    toks = llm.tokenize(text.encode("utf-8"))
    if len(toks) < 2: continue
    res = llm.eval(toks)
    # llama-cpp returns the per-token logprobs via 'logits' or 'output'
    # Simpler: use create_completion with echo=True and parse logprobs
    out = llm.create_completion(
        text, max_tokens=0, logprobs=True, echo=True,
    )
    lps = out["choices"][0].get("logprobs", {}).get("token_logprobs", [])
    # lps has -inf at position 0 (no history), then real logprobs for tokens[1:]
    for i, lp in enumerate(lps):
        if lp is not None and i > 0 and i < len(toks):
            total_nll += -lp
            total_n += 1
ppl_gguf = float(torch.tensor(total_nll / max(total_n, 1)).exp().item())
print(f"Q4_K_M PPL: {ppl_gguf:.3f}  ({total_n} tokens, {time.time()-t0:.1f}s)")

# Final summary
print()
print("=" * 60)
print("PERPLEXITY COMPARISON  (Qwen3-0.6B, 512 ctx, ~150 tokens)")
print("=" * 60)
print(f"  BF16 dense:      {bf16_ppl:8.3f}  ({bf16_n} tokens)")
print(f"  ICS int4:        {ics_ppl:8.3f}  ({ics_n} tokens)  hit = {ics_ppl/bf16_ppl:.2f}x")
print(f"  llama.cpp Q4_K_M:{ppl_gguf:8.3f}  ({total_n} tokens) hit = {ppl_gguf/bf16_ppl:.2f}x")
print()
print(f"ICS vs Q4_K_M: {ics_ppl/ppl_gguf:.2f}x {'better' if ics_ppl < ppl_gguf else 'worse'}")
print(f"ICS int4 model size: 381.7 MB (vs ~1.2 GB BF16, ~400 MB Q4_K_M)")
'''

t0 = time.time()
r = c.exec(script, timeout=2400)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[-2500:])
print(f"\nwall time: {time.time()-t0:.1f}s")
