"""Debug the chain dim mismatch: dump the failing chain's members and module types."""
import sys
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=600)

script = r'''
import sys, traceback
sys.path.insert(0, '/content/ICS')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", quantization_config=bnb, device_map="cuda")
tok = AutoTokenizer.from_pretrained("/content/Qwen3-0.6B")

from ics.pipeline import discover_chains, ICSConfig, _get_module
from ics.permutation import find_permutation_composite

chains = discover_chains(model, skip_modules=ICSConfig().skip_modules)
print(f"discovered {len(chains)} chains")
print()
# Find the chain that fails (with the 1572864 W_A)
for i, ch in enumerate(chains):
    members = ch.members
    W_A_raw = _get_module(model, members[0]).weight
    if W_A_raw is not None and hasattr(W_A_raw, 'shape') and W_A_raw.shape[0] == 1572864:
        print(f"FOUND: chain {i}")
        print(f"  members: {members}")
        print(f"  shared_dim_size: {ch.shared_dim_size}")
        print(f"  kind: {ch.kind}")
        print(f"  fisher_source: {ch.fisher_source}")
        for m in members:
            mod = _get_module(model, m)
            w = mod.weight if hasattr(mod, 'weight') else None
            t = type(mod).__name__
            if w is not None and hasattr(w, 'shape'):
                print(f"    {m}: type={t} weight.shape={tuple(w.shape)} dtype={w.dtype}")
            else:
                print(f"    {m}: type={t} weight={w}")
        break

# Also dump all unique (member-name pattern) for producer positions to see what's getting through
print()
print("--- unique producer members (first 10 chars) ---")
seen = set()
for ch in chains:
    pfx = ch.members[0].rsplit(".", 1)[0] if "." in ch.members[0] else ch.members[0]
    if pfx not in seen:
        seen.add(pfx)
        print(f"  {ch.members[0]}")
print()
print("--- first 3 chains fully ---")
for i, ch in enumerate(chains[:3]):
    print(f"chain {i}: shared={ch.shared_dim_size} kind={ch.kind} members={ch.members}")
print()
print("--- chains 25-32 (around the failure point) ---")
for i, ch in enumerate(chains[25:32], 25):
    print(f"chain {i}: shared={ch.shared_dim_size} kind={ch.kind} members={ch.members[:3]}{'...' if len(ch.members)>3 else ''}")
'''

r = c.exec(script, timeout=300)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[-1500:])
