"""Check the actual permutation lengths in the metadata + verify the k/v weights."""
import sys
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=300)

script = r'''
import sys, json
sys.path.insert(0, "/content/ICS")
import torch
from ics.export import load_ics_model, dequantized_state_dict
from transformers import AutoModelForCausalLM

ics = load_ics_model("/content/qwen3-0.6b-ics-fresh-int4")
meta = ics["meta"]

# Check permutation lengths
print("=== permutation lengths (first 10) ===")
for k, v in list(meta["permutations"].items())[:10]:
    perm = v["permutation"]
    print(f"  {k}: perm_length={len(perm)}  first_10={perm[:10]}")
print(f"  ... total {len(meta['permutations'])} perms")
print()

# Check shared_dim_size from chain discovery (we don't have it saved, but the
# perm length IS the shared_dim, so it should be 2048 for attn, 3072 for mlp)
import collections
lens = collections.Counter(len(v["permutation"]) for v in meta["permutations"].values())
print("perm length distribution:", dict(lens))
print()

# Compare the BF16 reference vs the dequantized model
ref = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", dtype=torch.bfloat16, device_map="cuda")
sd_ics = dequantized_state_dict(ics)

# Verify a single attention block: are k_proj / v_proj equal between ref and ics?
# (They are NOT in sd_ics, so the loaded model retains ref's BF16 values for them.)
for layer_idx in [0, 5, 10]:
    for proj in ["k_proj", "v_proj"]:
        name = f"model.layers.{layer_idx}.self_attn.{proj}.weight"
        w_ref = ref.state_dict()[name].float()
        print(f"  {name}: ref shape={tuple(w_ref.shape)} mean={w_ref.mean().item():.4f} std={w_ref.std().item():.4f}")
print()

# Now check: what does the dequantized q_proj look like vs a permuted version of ref's q_proj?
# We expect: dequant(q_proj) ≈ ref.q_proj[P, :] for the saved perm P
ch0 = list(meta["permutations"].keys())[0]  # e.g. "model.layers.0.self_attn.q_proj/model.layers.0.self_attn.o_proj"
print(f"checking chain: {ch0}")
perm0 = torch.tensor(meta["permutations"][ch0]["permutation"])
print(f"  perm length: {perm0.shape[0]}")
q_name = "model.layers.0.self_attn.q_proj.weight"
o_name = "model.layers.0.self_attn.o_proj.weight"
W_q_ref = ref.state_dict()[q_name].float()  # (2048, 1024)
W_q_ics = sd_ics[q_name].float()            # (2048, 1024) -- this is the dequantized post-perm version
W_o_ref = ref.state_dict()[o_name].float()  # (1024, 2048)
W_o_ics = sd_ics[o_name].float()            # (1024, 2048)

# Apply the perm to ref's q_proj: ref_permed = ref.q_proj[perm0]
W_q_ref_permed = W_q_ref[perm0, :]
print(f"  q_proj dequant vs ref[perm]: max_diff={(W_q_ics - W_q_ref_permed).abs().max().item():.4f} mean_diff={(W_q_ics - W_q_ref_permed).abs().mean().item():.4f}")
# This should be small (just INT4 quant noise) if the perm was applied to q_proj
print(f"  q_proj dequant vs ref (no perm): max_diff={(W_q_ics - W_q_ref).abs().max().item():.4f}")

# Same for o_proj
W_o_ref_permed = W_o_ref[:, perm0]
print(f"  o_proj dequant vs ref[:, perm]: max_diff={(W_o_ics - W_o_ref_permed).abs().max().item():.4f} mean_diff={(W_o_ics - W_o_ref_permed).abs().mean().item():.4f}")
print(f"  o_proj dequant vs ref (no perm): max_diff={(W_o_ics - W_o_ref).abs().max().item():.4f}")
'''

r = c.exec(script, timeout=300)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[-2000:])
