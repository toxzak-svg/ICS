"""Check qdata distribution and identify where the diff comes from."""
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

q_name = "model.layers.0.self_attn.q_proj"
qt = loaded["layers"][q_name]
W_ref = sd_ref[q_name + ".weight"].float()
gptq_perm = torch.tensor(loaded["layer_perms"][q_name], dtype=torch.long)
chain_info = loaded["chain_members"][f"{q_name}/{q_name.replace('q_proj', 'o_proj')}"]
chain_perm = torch.tensor(chain_info["permutation"], dtype=torch.long)

# qdata distribution
print("qdata: min={}, max={}, abs_mean={}, abs_max={}".format(
    qt.qdata.min().item(), qt.qdata.max().item(),
    qt.qdata.float().abs().mean().item(),
    qt.qdata.float().abs().max().item()
))
print("value distribution: ", torch.bincount((qt.qdata + 8).clamp(0, 16).long()))

# Reconstruct manually
W_saved = dequantize_blockwise(qt)  # in (chain_row_perm, gptq_col_perm) order
# Try dequant with a different block iteration
n_rows = qt.original_shape[0]  # 2048
n_cols = qt.original_shape[1]  # 1024
n_groups = qt.scales.shape[0]  # 8
group_size = n_cols // n_groups  # 128
print(f"n_rows={n_rows} n_cols={n_cols} n_groups={n_groups} group_size={group_size}")

# Manual dequant: each group is one scale
W_manual = torch.zeros((n_rows, n_cols), dtype=torch.float32)
for g in range(n_groups):
    start = g * group_size
    end = (g + 1) * group_size
    # qdata layout: row-major (n_rows, n_cols). For group g, the data is
    # for cols [start:end] across all rows. So:
    #   qdata_for_group = qdata.reshape(n_rows, n_cols)[:, start:end].flatten()
    qdata_2d = qt.qdata.reshape(n_rows, n_cols)
    block_q = qdata_2d[:, start:end]  # (n_rows, group_size)
    W_manual[:, start:end] = block_q.float() * qt.scales[g].item()
print(f"W_manual shape: {W_manual.shape}, sample: {W_manual[0, :5].tolist()}")

# Compare to dequantize_blockwise result
diff = (W_saved - W_manual).abs()
print(f"dequantize_blockwise vs manual: max={diff.max().item():.6f} mean={diff.mean().item():.6f}")

# Now check: is W_saved == W_after col-permuted? 
# We need to reconstruct W_after = W_ref[chain_perm]
W_after = W_ref[chain_perm]
# Apply gptq col perm: gptq_perm[i] is the i-th column in GPTQ order
# So W_after_in_gptq_order[:, i] = W_after[:, gptq_perm[i]]
W_after_in_gptq = W_after[:, gptq_perm]

# Compare to W_manual (which is in the gptq col order)
diff = (W_manual - W_after_in_gptq).abs()
print(f"W_manual vs W_after_in_gptq: max={diff.max().item():.4f} mean={diff.mean().item():.4f}")
# If they match, the storage/dequant is correct
# If not, there's a bug somewhere

# Look at the largest diffs
flat_diff = (W_manual - W_after_in_gptq).flatten()
print(f"top 5 abs diffs: {flat_diff.abs().topk(5).values.tolist()}")
print(f"  at positions: {flat_diff.abs().topk(5).indices.tolist()}")
top_idx = flat_diff.abs().argmax().item()
i, j = top_idx // n_cols, top_idx % n_cols
print(f"  W_manual[{i},{j}] = {W_manual[i,j].item():.4f}")
print(f"  W_after_in_gptq[{i},{j}] = {W_after_in_gptq[i,j].item():.4f}")
print(f"  original W_ref[chain_perm[i], gptq_perm[j]] = {W_after[i, gptq_perm[j]].item():.4f}")
print(f"  gptq_perm[{j}] = {gptq_perm[j].item()}")
'''

r = c.exec(script, timeout=300)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[-2000:])
