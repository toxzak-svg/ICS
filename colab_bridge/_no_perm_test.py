"""Quantize WITHOUT applying permutations. Forward pass should match BF16 closely."""
import sys, time
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=600)

# Patch pipeline.py with a "no perm" mode (skip apply_chain step)
# Then run the quant + benchmark
script = r'''
import sys, os, time, importlib
sys.path.insert(0, "/content/ICS")
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
# Force-reload the ics modules so we always see the latest /content/ICS/ics/*.py
import ics
for mod_name in list(sys.modules):
    if mod_name.startswith("ics."):
        del sys.modules[mod_name]
del sys.modules["ics"]
import ics.pipeline
import ics.quantize
import ics.export

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ics.pipeline import (
    discover_chains, ICSConfig, compute_fisher, _dequantize_bnb_inplace,
    find_permutation_composite, find_permutation_sinkhorn_hungarian,
    _get_module, assign_bit_widths
)
from ics.quantize import quantize_blockwise
from ics.quantize import QuantizedTensor
from ics.export import dequantized_state_dict, save_ics_model
import bitsandbytes as bnb

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", quantization_config=bnb_cfg, device_map="cuda")
tok = AutoTokenizer.from_pretrained("/content/Qwen3-0.6B")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

chains = discover_chains(model, skip_modules=ICSConfig().skip_modules)
print(f"chains: {len(chains)}")

# Tiny calibration for speed
print("Fisher...")
fisher = compute_fisher(
    model, tok,
    ["The quick brown fox.", "Quantization maps continuous values to discrete grids."],
    layer_filter=lambda n, m: any(n in c.members for c in chains),
    max_length=64, device="cuda", show_progress=False,
)
print(f"fisher: {len(fisher)}")

n = _dequantize_bnb_inplace(model)
print(f"dequantized {n} layers")
chains = discover_chains(model, skip_modules=ICSConfig().skip_modules)

fisher_norm = {name: (fs.fisher / (fs.fisher.max() + 1e-12)).cpu() for name, fs in fisher.items()}

# Find perms but DON'T apply them
perms = {}
for i, ch in enumerate(chains):
    F = fisher_norm[ch.fisher_source][:ch.shared_dim_size]
    W_A_raw = _get_module(model, ch.members[0]).weight.detach().float()
    W_A = W_A_raw[:ch.shared_dim_size, :]
    if len(ch.members) > 1:
        W_B_raw = _get_module(model, ch.members[-1]).weight.detach().float()
        W_B = W_B_raw[:, :ch.shared_dim_size]
    else:
        W_B = W_A.clone()
    perm = find_permutation_composite(W_A, W_B, F)
    perms["/".join(ch.members)] = perm
print(f"perms: {len(perms)}")

# Quantize WITHOUT applying perms
quant = {}
for ch in chains:
    chain_F = fisher_norm[ch.fisher_source][:ch.shared_dim_size]
    for n, target in zip(ch.members, ch.perm_targets):
        if n in quant: continue
        mod = _get_module(model, n)
        W = mod.weight.detach().float().cpu()
        if target == 0 and n in fisher_norm and fisher_norm[n].shape[0] == W.shape[0]:
            F = fisher_norm[n]
        else:
            F = chain_F
        bits = assign_bit_widths(F, block_size=64, int4_fraction=1.0, int2_fraction=0.0, int1_fraction=0.0)
        qt = quantize_blockwise(W, bits, block_size=64, dim=target)
        quant[n] = qt
print(f"quantized {len(quant)} layers (NO perm applied)")

# Now build a state dict (no perm baked in, just quant+dequant)
from ics.export import dequantize_blockwise
sd = {n + ".weight": dequantize_blockwise(qt) for n, qt in quant.items()}

# Load into BF16 model and check forward pass
ref = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", dtype=torch.bfloat16, device_map="cuda")
text = "The quick brown fox jumps over the lazy dog."
enc = tok(text, return_tensors="pt").to("cuda")
with torch.no_grad():
    out_ref = ref(input_ids=enc["input_ids"]).logits

import copy
ics_model = copy.deepcopy(ref)
ics_model.load_state_dict(sd, strict=False)
ics_model.eval()
with torch.no_grad():
    out_ics = ics_model(input_ids=enc["input_ids"]).logits

diff = (out_ics - out_ref).abs()
print(f"\n=== FORWARD DIFF (no perm, just int4 quant) ===")
print(f"max_abs={diff.max().item():.4f}  mean_abs={diff.mean().item():.4f}")
pred_ref = out_ref[0, -1].argmax().item()
pred_ics = out_ics[0, -1].argmax().item()
print(f"next-token: ref={tok.decode([pred_ref])!r}  ics={tok.decode([pred_ics])!r}")

# PPL on a longer eval
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
    nll = 0.0
    nt = 0
    with torch.no_grad():
        for i in range(0, ids.shape[-1] - 1, block):
            chunk = ids[:, i:i+block+1].to(m.device)
            o = m(input_ids=chunk, use_cache=False).logits[:, :-1, :].float()
            t = chunk[:, 1:]
            nll += torch.nn.functional.cross_entropy(o.reshape(-1, o.size(-1)), t.reshape(-1), reduction="sum").item()
            nt += t.numel()
    return float(torch.tensor(nll / nt).exp().item()), nt

p, n_t = ppl(ref, input_ids)
print(f"\nBF16 PPL: {p:.3f}  ({n_t} tokens)")
p, n_t = ppl(ics_model, input_ids)
print(f"ICS int4 (NO perm) PPL: {p:.3f}  ({n_t} tokens)")
'''

t0 = time.time()
r = c.exec(script, timeout=600)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[-2000:])
print(f"\nwall time: {time.time()-t0:.1f}s")
