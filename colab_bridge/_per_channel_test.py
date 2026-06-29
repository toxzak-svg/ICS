"""Confirm the rounding-to-zero issue, then test per-channel INT4 as a quick fix."""
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from ics.pipeline import discover_chains, _get_module
from ics.quantize import quantize_blockwise, dequantize_blockwise

# Load BF16, no bnb
print("loading BF16 ...")
model = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", dtype=torch.bfloat16, device_map="cuda")
tok = AutoTokenizer.from_pretrained("/content/Qwen3-0.6B")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# 1) Confirm the rounding-to-zero issue: for each block, how many weights round to 0?
q = model.model.layers[0].self_attn.q_proj
W = q.weight.detach().float().cpu()
print(f"q_proj shape={tuple(W.shape)} std={W.std().item():.4f}")
# Quantize along dim=0 with block_size=64
W_for_block = W.transpose(0, 1).contiguous()  # (1024, 2048)
n_blocks = 2048 // 64
total_zeroed = 0
total_weights = 0
for b in range(n_blocks):
    block = W_for_block[:, b*64:(b+1)*64].reshape(-1)
    absmax = block.abs().amax()
    if absmax < 1e-12:
        continue
    scale = absmax / 7
    q_vals = torch.round(block / scale)
    n_zero = (q_vals == 0).sum().item()
    n_total = block.numel()
    total_zeroed += n_zero
    total_weights += n_total
print(f"per-block int4 (block=64): {total_zeroed}/{total_weights} = {100*total_zeroed/total_weights:.1f}% weights quantized to 0")

# 2) Per-channel INT4: each output channel gets its own scale (no block sharing)
# This is what GPTQ and modern 4-bit schemes do.
def per_channel_int4(W):
    """Per-output-channel symmetric INT4 quantize + dequantize."""
    # W is (out_features, in_features)
    absmax = W.abs().amax(dim=1, keepdim=True)  # (out, 1)
    scale = absmax / 7.0
    q = torch.clamp(torch.round(W / scale), -8, 7).to(torch.int8)
    return q.float() * scale  # dequant

W_dq = per_channel_int4(W)
diff = (W_dq - W).abs()
print(f"\nper-channel int4: max_abs={diff.max().item():.4f}  mean_abs={diff.mean().item():.4f}")
print(f"  rel_max={(diff / (W.abs() + 1e-9)).max().item():.4f}  rel_mean={(diff / (W.abs() + 1e-9)).mean().item():.4f}")
nz = (W_dq == 0).sum().item()
print(f"  zeroed: {nz}/{W.numel()} = {100*nz/W.numel():.1f}%")

# 3) PPL with per-channel quant (all layers)
print("\n--- PPL with per-channel int4 (no perm) ---")
chains = discover_chains(model, skip_modules=())
texts = [
    "The transformer architecture uses self-attention to model long-range dependencies in sequences.",
    "Quantization maps continuous values to a discrete grid; per-block scaling factors preserve precision.",
    "The Fisher information matrix measures how sensitive the model loss is to perturbations in parameters.",
]
joined = "\n\n".join(texts)
enc = tok(joined, return_tensors="pt")
input_ids = enc["input_ids"]

def ppl(m, ids, block=128):
    m.eval()
    nll, nt = 0.0, 0
    with torch.no_grad():
        for i in range(0, ids.shape[-1] - 1, block):
            chunk = ids[:, i:i+block+1].to(m.device)
            o = m(input_ids=chunk, use_cache=False).logits[:, :-1, :].float()
            t = chunk[:, 1:]
            nll += torch.nn.functional.cross_entropy(o.reshape(-1, o.size(-1)), t.reshape(-1), reduction="sum").item()
            nt += t.numel()
    return float(torch.tensor(nll / nt).exp().item()), nt

p_bf16, _ = ppl(model, input_ids)
print(f"BF16 PPL: {p_bf16:.3f}")

sd = {}
for name, mod in model.named_modules():
    if not isinstance(mod, torch.nn.Linear): continue
    W = mod.weight.detach().float().cpu()
    W_dq = per_channel_int4(W)
    sd[name + ".weight"] = W_dq.to(model.dtype).to("cuda")
ics = copy.deepcopy(model)
ics.load_state_dict(sd, strict=False)
p, _ = ppl(ics, input_ids)
print(f"ICS per-channel int4 PPL: {p:.3f}  ({p/p_bf16:.1f}x BF16)")
'''

t0 = time.time()
r = c.exec(script, timeout=600)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[-2000:])
print(f"\nwall time: {time.time()-t0:.1f}s")
