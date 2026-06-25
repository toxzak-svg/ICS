"""Test GPTQ on a single real layer in isolation, verify round-trip and forward diff."""
import sys, time
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=600)

script = r'''
import sys, os, importlib, copy
for mod_name in list(sys.modules):
    if mod_name.startswith("ics."):
        del sys.modules[mod_name]
if "ics" in sys.modules:
    del sys.modules["ics"]
sys.path.insert(0, "/content/ICS")
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ics.pipeline import discover_chains, _dequantize_bnb_inplace, _get_module
from ics.gptq import compute_layer_hessian, gptq_quantize, dequantize_gptq

# Load BF16, no bnb
print("loading BF16...")
model = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", dtype=torch.bfloat16, device_map="cuda")
tok = AutoTokenizer.from_pretrained("/content/Qwen3-0.6B")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Get one layer: q_proj from layer 0
q = model.model.layers[0].self_attn.q_proj
W = q.weight.detach().float().cpu()
print(f"q_proj weight: shape={tuple(W.shape)} std={W.std().item():.4f} absmax={W.abs().amax().item():.4f}")

# Build calibration batches (use a few short texts)
calib = []
for text in [
    "The quick brown fox jumps over the lazy dog.",
    "In transformer architectures, attention computes a weighted sum.",
    "Quantization maps continuous values to a discrete grid.",
    "The Fisher information matrix measures sensitivity to perturbations.",
]:
    enc = tok(text, return_tensors="pt", truncation=True, max_length=64)
    if enc["input_ids"].shape[-1] >= 2:
        calib.append({k: v.to("cuda") for k, v in enc.items()})

# Compute Hessian for just this layer
print("computing Hessian for q_proj...")
hessians = compute_layer_hessian(
    model, calib,
    layer_filter=lambda n, m: n == "model.layers.0.self_attn.q_proj",
    device="cuda",
)
H = hessians["model.layers.0.self_attn.q_proj"]
print(f"Hessian: shape={tuple(H.shape)} diag_mean={torch.diag(H).mean().item():.4f} diag_min={torch.diag(H).min().item():.4f} diag_max={torch.diag(H).max().item():.4f}")
print(f"  cond approx: {torch.diag(H).max().item() / max(torch.diag(H).min().item(), 1e-9):.2e}")

# GPTQ quant
print("running GPTQ...")
t0 = time.time()
Q, scales, zeros, perm = gptq_quantize(W, H, bits=4, group_size=128, percdamp=0.01, blocksize=128)
print(f"  done in {time.time()-t0:.2f}s")
print(f"  Q shape: {tuple(Q.shape)} dtype={Q.dtype} range=[{Q.min().item()}, {Q.max().item()}]")
print(f"  scales: shape={tuple(scales.shape)} first 3: {scales[:3].tolist()}")
print(f"  perm: shape={tuple(perm.shape)} first 5: {perm[:5].tolist()}")

# Dequant and compare
W_dq = dequantize_gptq(Q, scales, zeros, perm, group_size=128, in_features=W.shape[1])
diff = (W_dq - W).abs()
print(f"\ndequant vs original:")
print(f"  max_abs_diff={diff.max().item():.4f}  mean_abs_diff={diff.mean().item():.4f}")
print(f"  rel_max={(diff / (W.abs() + 1e-9)).max().item():.4f}  rel_mean={(diff / (W.abs() + 1e-9)).mean().item():.4f}")

# Forward test: load the GPTQ-dequantized weight into the model and run forward
print("\n--- forward test: replace q_proj with GPTQ-dequant, run forward ---")
import copy
m2 = copy.deepcopy(model)
m2.model.layers[0].self_attn.q_proj.weight.data = W_dq.to(model.dtype).to("cuda")
text = "The quick brown fox jumps over the lazy dog."
enc = tok(text, return_tensors="pt").to("cuda")
with torch.no_grad():
    o_ref = model(input_ids=enc["input_ids"]).logits
    o_ics = m2(input_ids=enc["input_ids"]).logits
diff = (o_ics - o_ref).abs()
print(f"  forward diff: max={diff.max().item():.4f}  mean={diff.mean().item():.4f}")
print(f"  next-token: ref={tok.decode([o_ref[0,-1].argmax().item()])!r}  ics={tok.decode([o_ics[0,-1].argmax().item()])!r}")

# Compare to per-block INT4 on the same layer
print("\n--- for comparison: per-block INT4 (bs=128) on the same layer ---")
from ics.quantize import quantize_blockwise, dequantize_blockwise
bs = 128
bits = torch.full((W.shape[1] // bs,), 4, dtype=torch.int32)
qt = quantize_blockwise(W, bits, block_size=bs, dim=1)
W_pb = dequantize_blockwise(qt)
diff = (W_pb - W).abs()
print(f"  per-block max_abs_diff={diff.max().item():.4f}  mean_abs_diff={diff.mean().item():.4f}")
m3 = copy.deepcopy(model)
m3.model.layers[0].self_attn.q_proj.weight.data = W_pb.to(model.dtype).to("cuda")
with torch.no_grad():
    o_pb = m3(input_ids=enc["input_ids"]).logits
diff = (o_pb - o_ref).abs()
print(f"  per-block forward diff: max={diff.max().item():.4f}  mean={diff.mean().item():.4f}")
print(f"  per-block next-token: ref={tok.decode([o_ref[0,-1].argmax().item()])!r}  pb={tok.decode([o_pb[0,-1].argmax().item()])!r}")
'''

t0 = time.time()
r = c.exec(script, timeout=600)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[-3000:])
print(f"\nwall time: {time.time()-t0:.1f}s")
