"""Verify qdata layout in the saved file."""
import sys
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=60)

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
loaded = load_ics_model("/content/qwen3-0.6b-ics-gptq-int4")
q_name = "model.layers.0.self_attn.q_proj"
qt = loaded["layers"][q_name]
n_rows, n_cols = qt.original_shape
print(f"n_rows={n_rows}, n_cols={n_cols}, block_size={qt.block_size}")

# Probe different positions
print("\n--- qdata at different positions ---")
for pos in [0, 1024, 2048, 10000, 20000, 100000, 1000000]:
    if pos + 5 <= qt.qdata.numel():
        print(f"  qdata[{pos}:{pos+5}]: {qt.qdata[pos:pos+5].tolist()}")

# If row-major (natural): qdata[1024:1029] = row 1, cols 0-4
# If block-row-major: qdata[1024:1029] = row 8, cols 0-4 (since rows 0-7 are in cols 0-127 of block 0)

# Cross-check: pick a known location
# We can compute what row 1 col 0 should be from the saved Q
# But we don't have the original W. Let me just look at the pattern.
# If row-major, qdata[0:1024] = all of row 0. qdata[1024:2048] = all of row 1.
# In a typical weight matrix, row 0 and row 1 should have similar statistics.
# In block-row-major, qdata[0:128] = row 0 cols 0-127. qdata[128:256] = row 1 cols 0-127. They could differ.

# Simpler check: the dequantize_blockwise should now work
from ics.quantize import dequantize_blockwise
from transformers import AutoModelForCausalLM
ref = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", dtype=torch.bfloat16, device_map="cpu")
sd_ref = ref.state_dict()
W_ref = sd_ref[q_name + ".weight"].float()
gptq_perm = torch.tensor(loaded["layer_perms"][q_name], dtype=torch.long)
chain_info = loaded["chain_members"][f"{q_name}/{q_name.replace('q_proj', 'o_proj')}"]
chain_perm = torch.tensor(chain_info["permutation"], dtype=torch.long)
W_after = W_ref[chain_perm]
W_after_in_gptq = W_after[:, gptq_perm]

# Manual dequant
qdata_2d = qt.qdata.reshape(n_rows, n_cols)
n_groups = qt.scales.shape[0]
group_size = n_cols // n_groups
W_manual = torch.zeros((n_rows, n_cols))
for g in range(n_groups):
    s = g * group_size
    e = (g + 1) * group_size
    W_manual[:, s:e] = qdata_2d[:, s:e].float() * qt.scales[g].item()

# dequantize_blockwise
W_blockwise = dequantize_blockwise(qt)

# Compare
diff = (W_manual - W_after_in_gptq).abs()
print(f"\nmanual vs W_after_in_gptq: max={diff.max().item():.4f} mean={diff.mean().item():.4f}")
diff = (W_blockwise - W_after_in_gptq).abs()
print(f"blockwise vs W_after_in_gptq: max={diff.max().item():.4f} mean={diff.mean().item():.4f}")

# Are W_manual and W_blockwise the same?
diff = (W_manual - W_blockwise).abs()
print(f"manual vs blockwise: max={diff.max().item():.4f} mean={diff.mean().item():.4f}")
'''

r = c.exec(script)
print(r.stdout)
