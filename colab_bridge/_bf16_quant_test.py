"""Test: load BF16 (no bnb), do ICS int4 quant (no perm), check PPL.

This isolates: bnb 4-bit noise vs ICS int4 quant noise.
"""
import sys, time
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=600)

script = r'''
import sys, os, time, importlib, copy
for mod_name in list(sys.modules):
    if mod_name.startswith("ics."):
        del sys.modules[mod_name]
if "ics" in sys.modules:
    del sys.modules["ics"]
sys.path.insert(0, "/content/ICS")
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ics.pipeline import discover_chains, _get_module, assign_bit_widths
from ics.quantize import quantize_blockwise
from ics.export import dequantize_blockwise

# Load BF16 directly, no bnb
print("loading BF16 ...")
model = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", dtype=torch.bfloat16, device_map="cuda")
tok = AutoTokenizer.from_pretrained("/content/Qwen3-0.6B")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

chains = discover_chains(model, skip_modules=())
print(f"chains: {len(chains)}")

texts = [
    "The transformer architecture uses self-attention to model long-range dependencies in sequences.",
    "Quantization maps continuous values to a discrete grid; per-block scaling factors preserve precision.",
    "The Fisher information matrix measures how sensitive the model loss is to perturbations in parameters.",
]
joined = "\n\n".join(texts)
enc = tok(joined, return_tensors="pt")
input_ids = enc["input_ids"]

def ppl(m, ids, block=128):
    m.eval()
    nll, nt = 0.0, 0
    with torch.no_grad():
        for i in range(0, ids.shape[-1] - 1, block):
            chunk = ids[:, i:i+block+1].to(m.device)
            o = m(input_ids=chunk, use_cache=False).logits[:, :-1, :].float()
            t = chunk[:, 1:]
            nll += torch.nn.functional.cross_entropy(o.reshape(-1, o.size(-1)), t.reshape(-1), reduction="sum").item()
            nt += t.numel()
    return float(torch.tensor(nll / nt).exp().item()), nt

p_bf16, _ = ppl(model, input_ids)
print(f"BF16 PPL: {p_bf16:.3f}")

# Quantize ALL Linear layers (since no bnb noise) with int4
print("quantizing (no bnb noise, no perm) ...")
sd = {}
for name, mod in model.named_modules():
    if not isinstance(mod, torch.nn.Linear):
        continue
    W = mod.weight.detach().float().cpu()
    # Skip embed_tokens (would be a different quant scheme), but q/k/v/o/gate/up/down all quantized
    bs = 64
    n_blocks_w = (W.shape[0] + bs - 1) // bs
    bits = torch.full((n_blocks_w,), 4, dtype=torch.int32)
    qt = quantize_blockwise(W, bits, block_size=bs, dim=0)
    sd[name + ".weight"] = dequantize_blockwise(qt)

# Build ics model (load_state_dict non-strict, leaves the rest as BF16)
ics = copy.deepcopy(model)
ics.load_state_dict(sd, strict=False)
p, _ = ppl(ics, input_ids)
print(f"ICS int4 (no bnb, no perm) PPL: {p:.3f}  ({p/p_bf16:.1f}x BF16)")

# Forward diff
text = "The quick brown fox jumps over the lazy dog."
enc = tok(text, return_tensors="pt").to("cuda")
with torch.no_grad():
    o_ref = model(input_ids=enc["input_ids"]).logits
    o_ics = ics(input_ids=enc["input_ids"]).logits
diff = (o_ics - o_ref).abs()
print(f"forward diff: max={diff.max().item():.4f} mean={diff.mean().item():.4f}")
print(f"  next-token: ref={tok.decode([o_ref[0,-1].argmax().item()])!r}  ics={tok.decode([o_ics[0,-1].argmax().item()])!r}")
'''

t0 = time.time()
r = c.exec(script, timeout=600)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[-2000:])
print(f"\nwall time: {time.time()-t0:.1f}s")
