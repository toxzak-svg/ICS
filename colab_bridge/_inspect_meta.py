"""Inspect the saved meta to see what's actually in there."""
import sys
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=60)

script = r'''
import json
with open("/content/qwen3-0.6b-ics-gptq-int4/ics_meta.json") as f:
    meta = json.load(f)
print("top-level keys:", list(meta.keys()))
print("chain_members type:", type(meta.get("chain_members")))
if "chain_members" in meta:
    cm = meta["chain_members"]
    print(f"chain_members: {len(cm)} entries")
    if cm:
        for k in list(cm.keys())[:3]:
            print(f"  {k}: {cm[k]}")
print("permutations entries:", len(meta.get("permutations", {})))
print("layer_perms entries:", len(meta.get("layer_perms", {})))
print("quant_method:", meta.get("quant_method"))
'''
r = c.exec(script)
print(r.stdout)
