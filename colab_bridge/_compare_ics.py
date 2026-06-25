"""Compare ICS-dequantized weights against the original BF16 weights."""
import sys
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=300)

script = r'''
import sys
sys.path.insert(0, "/content/ICS")
import torch
from ics.export import load_ics_model, dequantized_state_dict
from transformers import AutoModelForCausalLM

# Load both
print("loading ICS dequantized state dict ...")
ics = load_ics_model("/content/qwen3-0.6b-ics-fresh-int4")
sd_ics = dequantized_state_dict(ics)
print(f"ICS sd has {len(sd_ics)} tensors")

print("loading BF16 reference ...")
ref = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", dtype=torch.bfloat16, device_map="cpu")
sd_ref = ref.state_dict()
print(f"ref sd has {len(sd_ref)} tensors")

# Pick a few layer pairs to compare
for layer in [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.5.self_attn.q_proj.weight",
    "model.layers.10.mlp.gate_proj.weight",
    "model.layers.27.self_attn.o_proj.weight",
]:
    if layer not in sd_ics:
        print(f"  {layer}: MISSING from ICS")
        continue
    w_ics = sd_ics[layer].float()
    w_ref = sd_ref[layer].float()
    if w_ics.shape != w_ref.shape:
        print(f"  {layer}: shape MISMATCH ics={tuple(w_ics.shape)} ref={tuple(w_ref.shape)}")
        continue
    diff = (w_ics - w_ref).abs()
    rel = diff / (w_ref.abs() + 1e-6)
    print(f"  {layer}: shape={tuple(w_ics.shape)}  max_abs_diff={diff.max().item():.4f}  mean_abs_diff={diff.mean().item():.4f}  rel_max={rel.max().item():.4f}  rel_mean={rel.mean().item():.4f}")

# Also check the meta to see if shapes are recorded correctly
print()
print("--- ics_meta.json layer entries (first 5) ---")
meta = ics["meta"]
for k in list(meta["layers"].keys())[:5]:
    print(f"  {k}: {meta['layers'][k]}")
print(f"  ... total {len(meta['layers'])} layers")
print(f"  permutations: {len(meta['permutations'])}")
'''

r = c.exec(script, timeout=300)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[-2000:])
