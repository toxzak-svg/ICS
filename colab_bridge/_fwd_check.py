"""Forward pass diff with the corrected dequant."""
import sys, time
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=300)

script = r'''
import sys, importlib, copy
for mod_name in list(sys.modules):
    if mod_name.startswith("ics."):
        del sys.modules[mod_name]
if "ics" in sys.modules:
    del sys.modules["ics"]
sys.path.insert(0, "/content/ICS")

import torch
from ics.export import load_ics_model, dequantized_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

loaded = load_ics_model("/content/qwen3-0.6b-ics-gptq-int4")
sd = dequantized_state_dict(loaded)
print(f"loaded {len(sd)} layers")
print(f"chain_members: {len(loaded['chain_members'])}")

ref = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", dtype=torch.bfloat16, device_map="cuda")
tok = AutoTokenizer.from_pretrained("/content/Qwen3-0.6B")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Forward diff on a short text
text = "The quick brown fox jumps over the lazy dog."
enc = tok(text, return_tensors="pt").to("cuda")
with torch.no_grad():
    o_ref = ref(input_ids=enc["input_ids"]).logits

ics_model = copy.deepcopy(ref)
ics_model.load_state_dict(sd, strict=False)
ics_model.eval()
with torch.no_grad():
    o_ics = ics_model(input_ids=enc["input_ids"]).logits

diff = (o_ics - o_ref).abs()
print(f"\nForward diff: max={diff.max().item():.4f} mean={diff.mean().item():.4f}")
print(f"  next-token: ref={tok.decode([o_ref[0,-1].argmax().item()])!r}  ics={tok.decode([o_ics[0,-1].argmax().item()])!r}")

# Also check: is the model state consistent? Look at q/k/v for layer 0
print("\n--- Layer 0 weights (post-load) ---")
for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
    p = ref.model.layers[0].self_attn.__getattr__(f"{name}.weight")
    print(f"  ref {name}: shape={tuple(p.shape)} mean={p.float().mean().item():.5f} std={p.float().std().item():.4f}")
    p2 = ics_model.model.layers[0].self_attn.__getattr__(f"{name}.weight")
    if p.shape == p2.shape:
        d = (p - p2).abs().float()
        print(f"  ics {name}: max_diff={d.max().item():.5f} mean_diff={d.mean().item():.5f}")
    else:
        print(f"  ics {name}: shape mismatch {p2.shape}")
'''

r = c.exec(script, timeout=300)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[-2000:])
