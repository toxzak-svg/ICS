"""Test: does increasing block_size fix the PPL? Tests the outlier-crushing hypothesis."""
import sys, time, importlib
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=600)

script = r'''
import sys, os, time, importlib
sys.path.insert(0, "/content/ICS")
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
# Force-reload
for mod_name in list(sys.modules):
    if mod_name.startswith("ics."):
        del sys.modules[mod_name]
if "ics" in sys.modules:
    del sys.modules["ics"]

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ics.pipeline import (
    discover_chains, ICSConfig, compute_fisher, _dequantize_bnb_inplace,
    _get_module, assign_bit_widths
)
from ics.quantize import quantize_blockwise
from ics.export import dequantize_blockwise

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", quantization_config=bnb_cfg, device_map="cuda")
tok = AutoTokenizer.from_pretrained("/content/Qwen3-0.6B")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

chains = discover_chains(model, skip_modules=ICSConfig().skip_modules)
print(f"chains: {len(chains)}")

print("Fisher (small)...")
fisher = compute_fisher(
    model, tok,
    ["The quick brown fox.", "Quantization maps continuous values to discrete grids."],
    layer_filter=lambda n, m: any(n in c.members for c in chains),
    max_length=64, device="cuda", show_progress=False,
)
n = _dequantize_bnb_inplace(model)
chains = discover_chains(model, skip_modules=ICSConfig().skip_modules)
fisher_norm = {name: (fs.fisher / (fs.fisher.max() + 1e-12)).cpu() for name, fs in fisher.items()}

ref = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", dtype=torch.bfloat16, device_map="cuda")

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

p_bf16, _ = ppl(ref, input_ids)
print(f"BF16 PPL: {p_bf16:.3f}")

# Now quantize at different block sizes
import copy
for bs in [64, 128, 256, 512, 1024]:
    sd = {}
    for ch in chains:
        chain_F = fisher_norm[ch.fisher_source][:ch.shared_dim_size]
        for n, target in zip(ch.members, ch.perm_targets):
            if n + ".weight" in sd: continue
            mod = _get_module(model, n)
            W = mod.weight.detach().float().cpu()
            if target == 0 and n in fisher_norm and fisher_norm[n].shape[0] == W.shape[0]:
                F = fisher_norm[n]
            else:
                F = chain_F
            try:
                bits = assign_bit_widths(F, block_size=bs, int4_fraction=1.0, int2_fraction=0.0, int1_fraction=0.0)
                qt = quantize_blockwise(W, bits, block_size=bs, dim=target)
                sd[n + ".weight"] = dequantize_blockwise(qt)
            except Exception as e:
                print(f"  block_size={bs} layer={n} FAILED: {e}")
                break
    m = copy.deepcopy(ref)
    m.load_state_dict(sd, strict=False)
    p, _ = ppl(m, input_ids)
    print(f"ICS int4 (no perm) block_size={bs:4d}: PPL={p:.3f}  ({p/p_bf16:.1f}x BF16)")
    del m
    torch.cuda.empty_cache()
'''

t0 = time.time()
r = c.exec(script, timeout=600)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[-2000:])
print(f"\nwall time: {time.time()-t0:.1f}s")
