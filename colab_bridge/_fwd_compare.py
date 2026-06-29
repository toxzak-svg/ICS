"""Check key alignment + forward pass diagnostic."""
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
from transformers import AutoModelForCausalLM, AutoTokenizer

ics = load_ics_model("/content/qwen3-0.6b-ics-fresh-int4")
sd_ics = dequantized_state_dict(ics)
ref = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", dtype=torch.bfloat16, device_map="cuda")
sd_ref = ref.state_dict()
tok = AutoTokenizer.from_pretrained("/content/Qwen3-0.6B")

# 1) Sanity: which keys are in sd_ics? Are k/v present?
ics_keys = sorted(sd_ics.keys())
ref_keys = sorted(sd_ref.keys())
print(f"ICS keys: {len(ics_keys)}")
print(f"Ref keys: {len(ref_keys)}")
print()

# Show the first 20 ICS keys (sorted)
print("first 20 ICS keys:")
for k in ics_keys[:20]:
    print(f"  {k}: {tuple(sd_ics[k].shape)}")
print()

# Are k_proj / v_proj in sd_ics?
for k in ics_keys:
    if "k_proj" in k or "v_proj" in k:
        print(f"  ICS has {k}: {tuple(sd_ics[k].shape)}")
        break
else:
    print("  No k_proj / v_proj in ICS (expected — GQA excludes them)")
print()

# 2) Run a forward pass on both: BF16 reference vs BF16+ICS overlay
ref.eval()
text = "The quick brown fox jumps over the lazy dog."
enc = tok(text, return_tensors="pt").to("cuda")
with torch.no_grad():
    out_ref = ref(input_ids=enc["input_ids"]).logits

# Now load ICS into a copy and forward
import copy
ics_model = copy.deepcopy(ref)
missing, unexpected = ics_model.load_state_dict(sd_ics, strict=False)
print(f"loaded: missing={len(missing)} unexpected={len(unexpected)}")
ics_model.eval()
with torch.no_grad():
    out_ics = ics_model(input_ids=enc["input_ids"]).logits

diff = (out_ics - out_ref).abs()
print(f"logits: shape={tuple(out_ics.shape)} max_abs_diff={diff.max().item():.4f} mean_abs_diff={diff.mean().item():.4f}")
rel = diff / (out_ref.abs() + 1e-6)
print(f"logits: rel_max={rel.max().item():.4f} rel_mean={rel.mean().item():.4f}")

# Also: how do the predictions compare?
pred_ref = out_ref[0, -1].argmax().item()
pred_ics = out_ics[0, -1].argmax().item()
print(f"next-token pred: ref={pred_ref} ({tok.decode([pred_ref])!r})  ics={pred_ics} ({tok.decode([pred_ics])!r})")
'''

r = c.exec(script, timeout=300)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[-2000:])
