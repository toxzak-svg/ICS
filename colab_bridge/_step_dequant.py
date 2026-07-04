"""Step-by-step dequant debug."""
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
from ics.export import load_ics_model
from ics.quantize import dequantize_blockwise
from transformers import AutoModelForCausalLM

# Load everything
loaded = load_ics_model("/content/qwen3-0.6b-ics-gptq-int4")
ref = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", dtype=torch.bfloat16, device_map="cpu")
sd_ref = ref.state_dict()

q_name = "model.layers.0.self_attn.q_proj"
qt = loaded["layers"][q_name]
print(f"qt.method: {qt.method}")
print(f"qt block_size: {qt.block_size}")

# 1. Dequantize the saved GPTQ block-row-major layout.
W_dq = dequantize_blockwise(qt)
print(f"W_dq shape: {W_dq.shape}, sample [0, :5]: {W_dq[0, :5].tolist()}")

# 2. Apply GPTQ col un-perm.
gptq_perm = torch.tensor(loaded["layer_perms"][q_name], dtype=torch.long)
W_after_gptq_unperm = torch.zeros_like(W_dq)
W_after_gptq_unperm[:, gptq_perm] = W_dq
print(f"W_after_gptq_unperm sample [0, :5]: {W_after_gptq_unperm[0, :5].tolist()}")

# 3. Apply chain un-perm (target=0 row perm).
chain_info = loaded["chain_members"][f"{q_name}/{q_name.replace('q_proj', 'o_proj')}"]
chain_perm = torch.tensor(chain_info["permutation"], dtype=torch.long)

W_orig = torch.zeros_like(W_after_gptq_unperm)
W_orig[chain_perm] = W_after_gptq_unperm
diff = (W_orig - sd_ref[q_name + ".weight"].float()).abs()
print(f"\nAfter dequantize_blockwise + un-perms vs ref: max={diff.max().item():.4f} mean={diff.mean().item():.4f}")
'''

r = c.exec(script, timeout=300)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[-2000:])
