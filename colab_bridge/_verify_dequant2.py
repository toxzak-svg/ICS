"""Verify the dequant is correctly un-perming the chain."""
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
from ics.export import load_ics_model, dequantized_state_dict
from transformers import AutoModelForCausalLM

loaded = load_ics_model("/content/qwen3-0.6b-ics-gptq-int4")
sd = dequantized_state_dict(loaded)
print(f"sd has {len(sd)} layers")
print(f"chain_members: {len(loaded.get('chain_members', {}))} entries")
print(f"layer_perms: {len(loaded.get('layer_perms', {}))} entries")

ref = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", dtype=torch.bfloat16, device_map="cpu")
sd_ref = ref.state_dict()

# Pick one q_proj from layer 0
for name in [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
]:
    if name not in sd:
        print(f"  {name}: MISSING from dequant")
        continue
    w_ics = sd[name].float()
    w_ref = sd_ref[name].float()
    diff = (w_ics - w_ref).abs()
    print(f"  {name}: shape={tuple(w_ics.shape)} max_diff={diff.max().item():.4f} mean_diff={diff.mean().item():.4f}  ref_std={w_ref.std().item():.4f}")
    # Also compute the diff of the chain-permuted version (i.e., the saved weight before un-perm)
    # We can't easily get that, but we can check: is the diff the same as a single-layer GPTQ test?
    # In a single-layer test we got max 0.10 mean 0.014. So we should see similar here.
'''

r = c.exec(script, timeout=300)
print(r.stdout)
