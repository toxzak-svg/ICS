"""Test the full pipeline on a single layer."""
import sys, time
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=600)

script = r'''
import sys, os, importlib, time
for mod_name in list(sys.modules):
    if mod_name.startswith("ics."):
        del sys.modules[mod_name]
if "ics" in sys.modules:
    del sys.modules["ics"]
sys.path.insert(0, "/content/ICS")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ics.pipeline import (
    discover_chains, compute_fisher, _dequantize_bnb_inplace, _get_module,
    find_permutation_composite, apply_chain,
)
from ics.gptq import compute_layer_hessian, gptq_quantize
from ics.quantize import QuantizedTensor, dequantize_blockwise
from ics.export import dequantized_state_dict, load_ics_model

# Load BF16, no bnb
print("loading BF16...")
model = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", dtype=torch.bfloat16, device_map="cuda")
tok = AutoTokenizer.from_pretrained("/content/Qwen3-0.6B")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Take a single attention chain: q_proj + o_proj from layer 0
chain = list(discover_chains(model, skip_modules=("lm_head", "embed_tokens", "norm", "mtp.")))[0]
print(f"chain: {chain.members} shared={chain.shared_dim_size}")

# Compute Fisher for just this chain
print("Fisher (small)...")
fisher = compute_fisher(
    model, tok,
    ["The quick brown fox.", "Quantization maps continuous to discrete."],
    layer_filter=lambda n, m: n in chain.members,
    max_length=64, device="cuda", show_progress=False,
)

# Find perm
q_name = chain.members[0]
W_A = _get_module(model, q_name).weight.detach().float()
W_B = _get_module(model, chain.members[-1]).weight.detach().float()
F = fisher[q_name].fisher / fisher[q_name].fisher.max()
F = F[:chain.shared_dim_size]
W_A = W_A[:chain.shared_dim_size, :]
W_B = W_B[:, :chain.shared_dim_size] if len(chain.members) > 1 else W_A.clone()
print(f"W_A shape: {W_A.shape}  W_B shape: {W_B.shape}")
perm = find_permutation_composite(W_A, W_B, F)
perm_t = perm.permutation
print(f"perm shape: {perm_t.shape} first 10: {perm_t[:10].tolist()}")

# Apply the perm
apply_chain(model, chain, perm, F)
print("applied chain perm")

# Get the chain-permuted weight
W_after = _get_module(model, q_name).weight.detach().float()
print(f"W_after shape: {W_after.shape} (should still be (out, in) but rows permuted)")

# Compute Hessian
print("Hessian...")
calib = []
for text in ["The quick brown fox.", "Quantization maps continuous to discrete."]:
    enc = tok(text, return_tensors="pt", truncation=True, max_length=64)
    if enc["input_ids"].shape[-1] >= 2:
        calib.append({k: v.to("cuda") for k, v in enc.items()})
hessians = compute_layer_hessian(
    model, calib,
    layer_filter=lambda n, m: n == q_name,
    device="cuda",
)
H = hessians[q_name]
print(f"H shape: {H.shape}")

# GPTQ
print("GPTQ...")
Q, scales, zeros, gptq_perm = gptq_quantize(W_after, H, bits=4, group_size=128)
print(f"  Q shape: {Q.shape}  gptq_perm[:5]: {gptq_perm[:5].tolist()}")
print(f"  scales: {scales.tolist()}")

# Save as QuantizedTensor
qt = QuantizedTensor(
    qdata=Q.reshape(-1).contiguous().cpu(),
    scales=scales.cpu(),
    zeros=zeros.cpu(),
    bits=torch.full((scales.shape[0],), 4, dtype=torch.int32),
    block_size=128,
    original_shape=tuple(W_after.shape),
    quant_dim=1,
    method="gptq_per_group",
)
print(f"qt.qdata shape: {qt.qdata.shape}, scales.shape: {qt.scales.shape}, block_size: {qt.block_size}")

# Dequant and compare
W_dequant = dequantize_blockwise(qt)
W_after_cpu = W_after.cpu()
W_dequant_cpu = W_dequant.cpu()
print(f"\nW_dequant shape: {W_dequant.shape}")
gptq_perm_cpu = gptq_perm.cpu()
diff = (W_dequant_cpu - W_after_cpu[:, gptq_perm_cpu]).abs()
print(f"vs W_after[:, gptq_perm]: max={diff.max().item():.4f} mean={diff.mean().item():.4f}")
'''

t0 = time.time()
r = c.exec(script, timeout=600)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[-2000:])
print(f"wall time: {time.time()-t0:.1f}s")
