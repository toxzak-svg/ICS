"""Trace one chain end-to-end: apply perm, quant, dequant, un-perm, compare."""
import sys, time
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=600)

script = r'''
import sys, importlib
for mod_name in list(sys.modules):
    if mod_name.startswith("ics."):
        del sys.modules[mod_name]
if "ics" in sys.modules:
    del sys.modules["ics"]
sys.path.insert(0, "/content/ICS")

import torch
from ics.export import load_ics_model, dequantized_state_dict
from ics.quantize import dequantize_blockwise
from transformers import AutoModelForCausalLM

loaded = load_ics_model("/content/qwen3-0.6b-ics-gptq-int4")
ref = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", dtype=torch.bfloat16, device_map="cpu")
sd_ref = ref.state_dict()

# Take q_proj from layer 0
q_name = "model.layers.0.self_attn.q_proj"
qt = loaded["layers"][q_name]
W_saved = dequantize_blockwise(qt)  # this is in (chain_perm, gptq_perm) order
print(f"W_saved shape: {tuple(W_saved.shape)}  expected (2048, 1024)")

# Reference
W_ref = sd_ref[q_name + ".weight"].float()
print(f"W_ref shape: {tuple(W_ref.shape)}")

# Get the perms
gptq_perm = torch.tensor(loaded["layer_perms"][q_name], dtype=torch.long)
chain_info = loaded["chain_members"][f"{q_name}/{q_name.replace('q_proj', 'o_proj')}"]
chain_perm = torch.tensor(chain_info["permutation"], dtype=torch.long)
chain_target = chain_info["targets"][0]  # 0 for q_proj
print(f"chain_perm shape: {tuple(chain_perm.shape)}  target: {chain_target}")
print(f"gptq_perm shape: {tuple(gptq_perm.shape)}")
print(f"gptq_perm[:10]: {gptq_perm[:10].tolist()}")
print(f"chain_perm[:10]: {chain_perm[:10].tolist()}")

# Step 1: apply chain perm to reference
if chain_target == 0:
    W_ref_chain = W_ref[chain_perm]  # row perm
else:
    W_ref_chain = W_ref[:, chain_perm]  # col perm
print(f"\nW_ref_chain (after applying chain perm):")
diff = (W_saved - W_ref_chain).abs()
print(f"  vs W_saved: max={diff.max().item():.4f} mean={diff.mean().item():.4f}")

# Step 2: apply GPTQ perm to chain-permuted reference
W_ref_gptq = W_ref_chain[:, gptq_perm]
print(f"\nW_ref_gptq (after applying gptq perm):")
diff = (W_saved - W_ref_gptq).abs()
print(f"  vs W_saved: max={diff.max().item():.4f} mean={diff.mean().item():.4f}")

# So the saved W should equal W_ref_gptq approximately (modulo GPTQ error)
# Then un-perm GPTQ: W_saved[:, gptq_perm] = W_chain
# And un-perm chain: W_chain[chain_perm] = W_orig (if target=0)

# Let me verify by un-perm step by step
W_after_gptq_unperm = torch.zeros_like(W_saved)
W_after_gptq_unperm[:, gptq_perm] = W_saved
print(f"\nAfter un-perm GPTQ:")
diff = (W_after_gptq_unperm - W_ref_chain).abs()
print(f"  vs W_ref_chain: max={diff.max().item():.4f} mean={diff.mean().item():.4f}")

W_after_chain_unperm = torch.zeros_like(W_after_gptq_unperm)
if chain_target == 0:
    W_after_chain_unperm[chain_perm] = W_after_gptq_unperm
else:
    W_after_chain_unperm[:, chain_perm] = W_after_gptq_unperm
print(f"\nAfter un-perm chain (final):")
diff = (W_after_chain_unperm - W_ref).abs()
print(f"  vs W_ref: max={diff.max().item():.4f} mean={diff.mean().item():.4f}")

# Also check: does the dequantized_state_dict do the same thing?
sd = dequantized_state_dict(loaded)
W_dsd = sd[q_name + ".weight"].float()
diff = (W_dsd - W_ref).abs()
print(f"\ndequantized_state_dict vs W_ref: max={diff.max().item():.4f} mean={diff.mean().item():.4f}")
print(f"  diff vs my manual unperm: max={(W_dsd - W_after_chain_unperm).abs().max().item():.4f}")
'''

r = c.exec(script, timeout=600)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[-2000:])
