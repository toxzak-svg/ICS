"""Smoke test for the GPTQ module on a single Linear layer with synthetic data."""
import sys, time
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=120)

script = r'''
import sys, importlib, time
for mod_name in list(sys.modules):
    if mod_name.startswith("ics."):
        del sys.modules[mod_name]
if "ics" in sys.modules:
    del sys.modules["ics"]
sys.path.insert(0, "/content/ICS")

import torch
from ics.gptq import gptq_quantize, dequantize_gptq, compute_layer_hessian

# 1) Synthetic test: random weight + Hessian
torch.manual_seed(0)
out_f, in_f = 512, 1024
W = torch.randn(out_f, in_f) * 0.05
X = torch.randn(64 * 128, in_f)  # 64 samples, 128 tokens, input_features
H = 2.0 * (X.T @ X) / X.shape[0]
H += 0.01 * torch.mean(torch.diag(H)) * torch.eye(in_f)
print(f"synthetic: W={tuple(W.shape)} H={tuple(H.shape)}")

t0 = time.time()
Q, scales, zeros, perm = gptq_quantize(W.clone(), H.clone(), bits=4, group_size=128)
print(f"GPTQ quant: {time.time()-t0:.2f}s")
print(f"  Q range: [{Q.min().item()}, {Q.max().item()}] dtype={Q.dtype}")
print(f"  scales: shape={tuple(scales.shape)} sample={scales[:3].tolist()}")
print(f"  perm: shape={tuple(perm.shape)} first10={perm[:10].tolist()}")

# Dequantize
W_dq = dequantize_gptq(Q, scales, zeros, perm, group_size=128, in_features=in_f)
diff = (W_dq - W).abs()
print(f"  dequant max_abs={diff.max().item():.4f}  mean_abs={diff.mean().item():.4f}")
print(f"  rel_max={(diff / (W.abs() + 1e-9)).max().item():.4f}  rel_mean={(diff / (W.abs() + 1e-9)).mean().item():.4f}")

# 2) Compare to per-block INT4 (the current broken approach) on the same W
from ics.quantize import quantize_blockwise, dequantize_blockwise
bs = 128
bits = torch.full((in_f // bs,), 4, dtype=torch.int32)
qt = quantize_blockwise(W, bits, block_size=bs, dim=1)
W_pb = dequantize_blockwise(qt)
diff_pb = (W_pb - W).abs()
print(f"per-block int4 (bs={bs}): max_abs={diff_pb.max().item():.4f}  mean_abs={diff_pb.mean().item():.4f}")
print(f"  rel_max={(diff_pb / (W.abs() + 1e-9)).max().item():.4f}  rel_mean={(diff_pb / (W.abs() + 1e-9)).mean().item():.4f}")
'''

r = c.exec(script, timeout=120)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[-2000:])
