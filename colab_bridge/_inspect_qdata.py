"""Inspect the raw GPTQ output (qdata) and verify the gptq_perm makes sense."""
import sys
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
from ics.export import load_ics_model
from ics.quantize import dequantize_blockwise
from transformers import AutoModelForCausalLM

loaded = load_ics_model("/content/qwen3-0.6b-ics-gptq-int4")
ref = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", dtype=torch.bfloat16, device_map="cpu")
sd_ref = ref.state_dict()

# Take q_proj from layer 0
q_name = "model.layers.0.self_attn.q_proj"
qt = loaded["layers"][q_name]
print(f"qt method: {qt.method}")
print(f"qt original_shape: {qt.original_shape}")
print(f"qt quant_dim: {qt.quant_dim}")
print(f"qt block_size: {qt.block_size}")
print(f"qt qdata shape: {tuple(qt.qdata.shape)}")
print(f"qt scales shape: {tuple(qt.scales.shape)}")
print(f"qt zeros shape: {tuple(qt.zeros.shape)}")
print(f"qt bits shape: {tuple(qt.bits.shape)}")
print(f"qdata first 20: {qt.qdata[:20].tolist()}")
print(f"scales: {qt.scales.tolist()}")
print(f"zeros: {qt.zeros.tolist()}")
print(f"bits: {qt.bits.tolist()}")

# The qdata is stored as (n_rows * n_cols,) for dim=1 quant (after transpose to make dim 1 last)
# For shape (2048, 1024) and block_size=128, dim=1: 8 groups along col dim, 2048 rows per group
# So total qdata = 8 * 2048 * 128 = 2097152 elements

# Dequant: this gives the (chain row perm, gptq col perm) ordered weight
W_saved = dequantize_blockwise(qt)
print(f"\nW_saved shape: {tuple(W_saved.shape)}")
print(f"W_saved[0, :10]: {W_saved[0, :10].tolist()}")

# Apply gptq col perm to recover chain-permuted weight
gptq_perm = torch.tensor(loaded["layer_perms"][q_name], dtype=torch.long)
W_after_gptq_unperm = torch.zeros_like(W_saved)
W_after_gptq_unperm[:, gptq_perm] = W_saved
print(f"\nW_after_gptq_unperm[0, :10]: {W_after_gptq_unperm[0, :10].tolist()}")
# This should be the chain-permuted weight

# Compare to the chain-permuted reference
chain_info = loaded["chain_members"][f"{q_name}/{q_name.replace('q_proj', 'o_proj')}"]
chain_perm = torch.tensor(chain_info["permutation"], dtype=torch.long)
W_ref = sd_ref[q_name + ".weight"].float()
W_ref_chain = W_ref[chain_perm]  # row perm (target=0)
print(f"\nW_ref_chain[0, :10]: {W_ref_chain[0, :10].tolist()}")
diff = (W_after_gptq_unperm - W_ref_chain).abs()
print(f"diff: max={diff.max().item():.4f} mean={diff.mean().item():.4f}")
'''

r = c.exec(script, timeout=300)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[-2000:])
