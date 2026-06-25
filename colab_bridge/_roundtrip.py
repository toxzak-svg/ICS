"""Isolated round-trip test on the real q_proj weight."""
import sys
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=300)

script = r'''
import sys
for mod_name in list(sys.modules):
    if mod_name.startswith("ics."):
        del sys.modules[mod_name]
if "ics" in sys.modules:
    del sys.modules["ics"]
sys.path.insert(0, "/content/ICS")

import torch
from ics.quantize import quantize_blockwise, dequantize_blockwise
from ics.pipeline import _dequantize_bnb_inplace
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", quantization_config=bnb_cfg, device_map="cuda")
n = _dequantize_bnb_inplace(model)
print(f"dequantized {n} layers")

q = model.model.layers[0].self_attn.q_proj
W = q.weight.detach().float().cpu()
print(f"q_proj weight: shape={tuple(W.shape)} std={W.std().item():.4f} absmax={W.abs().amax().item():.4f}")
print(f"  per-block absmax (block_size=64, dim=0):")
n_blocks = W.shape[0] // 64
W_for_block = W.transpose(0, 1).contiguous()  # (1024, 2048) so dim 0 is the original dim 0
for b in range(min(5, n_blocks)):
    block = W_for_block[:, b*64:(b+1)*64].reshape(-1)
    print(f"    block {b}: absmax={block.abs().amax().item():.4f}  mean_abs={block.abs().mean().item():.4f}  median_abs={block.abs().median().item():.4f}")

# Roundtrip
bits = torch.full(((W.shape[0] + 63) // 64,), 4, dtype=torch.int32)
qt = quantize_blockwise(W, bits, block_size=64, dim=0)
W_dq = dequantize_blockwise(qt)
diff = (W_dq - W).abs()
print(f"\nroundtrip dim=0: max_abs_diff={diff.max().item():.4f}  mean_abs_diff={diff.mean().item():.4f}")

# Also dim=1
qt = quantize_blockwise(W, bits, block_size=64, dim=1)
W_dq = dequantize_blockwise(qt)
diff = (W_dq - W).abs()
print(f"roundtrip dim=1: max_abs_diff={diff.max().item():.4f}  mean_abs_diff={diff.mean().item():.4f}")

# The pipeline uses target=0 for q_proj (rows) and target=1 for o_proj (cols)
# So q_proj is quantized with dim=0, o_proj with dim=1
# Both should work. Let me also test what happens if I quantize the FIRST dim with a small
# block and see how the absmax of the block is dominated by outliers

# Test: take the original weight, apply a small "structural" change
# If we permute the rows of W to sort by row-magnitude, then quantize, the error should be similar
W_sorted = W[torch.argsort(W.abs().amax(dim=1), descending=True), :]
qt = quantize_blockwise(W_sorted, bits, block_size=64, dim=0)
W_dq = dequantize_blockwise(qt)
diff = (W_dq - W_sorted).abs()
print(f"\nafter sorting rows by absmax (mimicking what ICS perm does):")
print(f"  max_abs_diff={diff.max().item():.4f}  mean_abs_diff={diff.mean().item():.4f}")
print(f"  (this should be no better than unsorted, but may cluster the error)")
'''

r = c.exec(script, timeout=300)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[-2000:])
