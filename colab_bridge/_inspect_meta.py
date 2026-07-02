"""Inspect saved perms and chain membership for layers 5 vs 27."""
import sys
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient
c = ColabClient('https://citizenship-compliant-laughing-statistical.trycloudflare.com', 'u8hpHpMt_9ivdocAtuFUXPvBEk6bUY-h')
r = c.exec("""
import sys, json; sys.path.insert(0, "/content/ICS")
from ics.export import load_ics_model
loaded = load_ics_model("/content/qwen3-0.6b-ics-gptq-int4")
meta = loaded["meta"]
print("meta keys:", list(meta.keys()))
print("n_perms:", len(loaded.get("layer_perms", {})))
print("n_chain_members:", len(loaded.get("chain_members", {})))
print()
print("layer_perms sample (layer 5 + 27 gate):")
for k in ['model.layers.5.mlp.gate_proj', 'model.layers.27.mlp.gate_proj']:
    if k in loaded['layer_perms']:
        p = loaded['layer_perms'][k]
        print(f"  {k}: len={len(p)}, first 8 = {p[:8]}, last 4 = {p[-4:]}")
print()
print("chain_members containing gate_proj (first 3 chains):")
chains = loaded.get("chain_members", {})
for ck, info in list(chains.items())[:3]:
    print(f"  chain: {ck}")
    print(f"    members: {info.get('members')}")
    print(f"    permutation: len={len(info.get('permutation', []))}, first 4 = {info.get('permutation', [])[:4]}")
    print(f"    gqa_sub_perm: {info.get('gqa_sub_perm')}")
    print(f"    gqa_sub_perm_members: {info.get('gqa_sub_perm_members')}")
    print(f"    member_perms keys: {list((info.get('member_perms') or {}).keys())}")
    print(f"    targets: {info.get('targets')}")
print()
# Find chain containing layer 27 gate
print("chain containing layer 27 gate:")
for ck, info in chains.items():
    if 'model.layers.27.mlp.gate_proj' in info.get('members', []):
        print(f"  chain: {ck}")
        print(f"    members: {info.get('members')}")
        print(f"    permutation len: {len(info.get('permutation', []))}")
        break
""")
print(r.stdout)
if r.error:
    print('ERR:', r.error[-1000:])