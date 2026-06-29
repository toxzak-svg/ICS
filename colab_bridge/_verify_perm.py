"""Verify the dequant state dict is correct by re-applying the chain perm and forward."""
import sys
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
ref = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", dtype=torch.bfloat16, device_map="cuda")
tok = AutoTokenizer.from_pretrained("/content/Qwen3-0.6B")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Test 1: apply chain perm to the BF16 model, then load sd. The model should match the original.
# This verifies the dequant is correct AND the un-perm is correct.
ref_permed = copy.deepcopy(ref)
for chain_key, info in loaded["chain_members"].items():
    members = info["members"]
    targets = info["targets"]
    perm = torch.tensor(info["permutation"], dtype=torch.long)
    for m, t in zip(members, targets):
        w = ref_permed
        for p in m.split("."):
            w = getattr(w, p)
        w_data = w.weight.data
        if t == 0:
            # row perm: w[cperm] = w_orig => w_orig = zeros; w_orig[cperm] = w
            new_data = torch.zeros_like(w_data)
            new_data[perm] = w_data
            w_data.copy_(new_data)
        else:
            new_data = torch.zeros_like(w_data)
            new_data[:, perm] = w_data
            w_data.copy_(new_data)

# Now ref_permed is the chain-permuted version. Load sd into ref_permed.
# If sd represents the post-perm weights (which it should), the loaded model should
# give similar forward pass to the perm-applied ref.
sd_permed = {}
for k, v in sd.items():
    sd_permed[k] = v.to("cuda")
ref_permed.load_state_dict(sd_permed, strict=False)
ref_permed.eval()

text = "The quick brown fox jumps over the lazy dog."
enc = tok(text, return_tensors="pt").to("cuda")
with torch.no_grad():
    o_ref = ref(input_ids=enc["input_ids"]).logits
    o_permed = ref_permed(input_ids=enc["input_ids"]).logits
diff = (o_permed - o_ref).abs()
print(f"chain-permed model + loaded sd vs original: max={diff.max().item():.4f} mean={diff.mean().item():.4f}")
print(f"  next-token: ref={tok.decode([o_ref[0,-1].argmax().item()])!r}  perm+sd={tok.decode([o_permed[0,-1].argmax().item()])!r}")

# Now also: load sd into the FRESH ref (without perm). The model should be the original
# (modulo GPTQ error). If the un-perm in dequantized_state_dict is correct, this should
# also give similar output.
ics_model = copy.deepcopy(ref)
ics_model.load_state_dict({k: v.to("cuda") for k, v in sd.items()}, strict=False)
ics_model.eval()
with torch.no_grad():
    o_ics = ics_model(input_ids=enc["input_ids"]).logits
diff = (o_ics - o_ref).abs()
print(f"\nun-permed (dequant) model vs original: max={diff.max().item():.4f} mean={diff.mean().item():.4f}")
print(f"  next-token: ref={tok.decode([o_ref[0,-1].argmax().item()])!r}  ics={tok.decode([o_ics[0,-1].argmax().item()])!r}")
'''

r = c.exec(script, timeout=300)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[-2000:])
