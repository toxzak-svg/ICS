"""Debug the GPTQ dequant: compare to a fresh GPTQ on the same input."""
import sys, time
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=300)

script = r'''
import sys, importlib
for mod_name in list(sys.modules):
    if mod_name.startswith("ics."):
        del sys.modules[mod_name]
if "ics" in sys.modules:
    del sys.modules["ics"]
sys.path.insert(0, "/content/ICS")

import torch
from ics.export import load_ics_model, dequantize_gptq_per_group
from transformers import AutoModelForCausalLM
from ics.gptq import gptq_quantize, dequantize_gptq
import bitsandbytes as bnb
from ics.pipeline import _dequantize_bnb_inplace, find_permutation_composite, apply_chain, discover_chains, compute_fisher
import torch.nn as nn

# Load BF16
model = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", dtype=torch.bfloat16, device_map="cuda")
tok = __import__("transformers").AutoTokenizer.from_pretrained("/content/Qwen3-0.6B")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Pick a single layer
chains = discover_chains(model, skip_modules=())
chain = chains[0]
print(f"chain: {chain.members}")

# Compute Fisher for this chain only
fisher = compute_fisher(
    model, tok,
    ["The quick brown fox.", "Quantization maps continuous to discrete."],
    layer_filter=lambda n, m: n in chain.members,
    max_length=64, device="cuda", show_progress=False,
)
q_name = chain.members[0]
F = fisher[q_name].fisher / fisher[q_name].fisher.max()
F = F[:chain.shared_dim_size]
W_A = _get_module_local = model
for p in q_name.split("."):
    W_A = getattr(W_A, p)
W_A = W_A.weight.detach().float()
W_A = W_A[:chain.shared_dim_size, :]
W_B = model
for p in chain.members[-1].split("."):
    W_B = getattr(W_B, p)
W_B = W_B.weight.detach().float()
W_B = W_B[:, :chain.shared_dim_size]
perm_result = find_permutation_composite(W_A, W_B, F)
apply_chain(model, chain, perm_result, F)

# Get the chain-permuted W
W_after = model
for p in q_name.split("."):
    W_after = getattr(W_after, p)
W_after = W_after.weight.detach().float()
print(f"W_after shape: {W_after.shape}, absmax: {W_after.abs().amax().item():.4f}")

# Load the saved qdata for this layer
loaded = load_ics_model("/content/qwen3-0.6b-ics-gptq-int4")
qt = loaded["layers"][q_name]
gptq_perm_saved = torch.tensor(loaded["layer_perms"][q_name], dtype=torch.long)
chain_perm_saved = torch.tensor(loaded["chain_members"][f"{q_name}/{q_name.replace('q_proj', 'o_proj')}"]["permutation"], dtype=torch.long)
chain_target = loaded["chain_members"][f"{q_name}/{q_name.replace('q_proj', 'o_proj')}"]["targets"][0]

print(f"saved qdata[0:5]: {qt.qdata[0:5].tolist()}")
print(f"saved qdata[1024:1029]: {qt.qdata[1024:1029].tolist()}")

# Now run GPTQ fresh on W_after
print("computing Hessian for q_proj...")
calib = []
for text in ["The quick brown fox.", "Quantization maps continuous to discrete."]:
    enc = tok(text, return_tensors="pt", truncation=True, max_length=64)
    if enc["input_ids"].shape[-1] >= 2:
        calib.append({k: v.to("cuda") for k, v in enc.items()})
from ics.gptq import compute_layer_hessian
hessians = compute_layer_hessian(
    model, calib,
    layer_filter=lambda n, m: n == q_name,
    device="cuda",
)
H = hessians[q_name]
print(f"H diag range: {H.diag().min().item():.4f} to {H.diag().max().item():.4f}")

# Fresh GPTQ
Q_fresh, scales_fresh, zeros_fresh, perm_fresh = gptq_quantize(W_after, H, bits=4, group_size=128)
print(f"fresh Q[0, :10]: {Q_fresh[0, :10].tolist()}")
print(f"fresh scales: {scales_fresh.tolist()}")
print(f"fresh perm: {perm_fresh[:10].tolist()}")
print(f"saved scales: {qt.scales.tolist()}")
print(f"saved perm: {gptq_perm_saved[:10].tolist()}")

# Compare qdata
print(f"\\nfresh qdata (flat, first 20): {Q_fresh.reshape(-1)[:20].tolist()}")
print(f"saved qdata (flat, first 20): {qt.qdata[:20].tolist()}")

# Check if saved qdata == fresh Q.reshape(-1) (row-major natural)
diff = (Q_fresh.reshape(-1).float() - qt.qdata.float()).abs()
print(f"\\nsaved qdata vs fresh Q.flat: max={diff.max().item():.4f} mean={diff.mean().item():.4f}")
'''

r = c.exec(script, timeout=300)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[-2000:])
