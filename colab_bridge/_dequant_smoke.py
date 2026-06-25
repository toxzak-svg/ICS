"""Step 1: verify smoke artifact dequant round-trip + check it produces finite tensors."""
import sys
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=300)

script = r'''
import sys
sys.path.insert(0, '/content/ICS')
import torch
from ics.export import load_ics_model, dequantized_state_dict

print("--- load_ics_model ---")
loaded = load_ics_model('/content/ICS/qwen3-0.6b-ics-smoke')
print("type:", type(loaded).__name__)
if isinstance(loaded, dict):
    print("keys:", list(loaded.keys())[:5])
    for k in list(loaded.keys())[:3]:
        v = loaded[k]
        if isinstance(v, torch.Tensor):
            print(f"  {k}: tensor {tuple(v.shape)} dtype={v.dtype} finite={torch.isfinite(v).all().item()}")
        else:
            print(f"  {k}: {type(v).__name__}")

print()
print("--- dequantized_state_dict ---")
sd = dequantized_state_dict(loaded)
print("returned:", type(sd).__name__)
if isinstance(sd, dict):
    print("num tensors:", len(sd))
    for k in list(sd.keys())[:5]:
        v = sd[k]
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype} finite={torch.isfinite(v).all().item()} mean={v.float().mean().item():.4f} std={v.float().std().item():.4f}")
        else:
            print(f"  {k}: {type(v).__name__} {v}")

print()
print("--- check bits/scales/zeros shapes are consistent ---")
scales = loaded.get('scales')
zeros = loaded.get('zeros')
bits = loaded.get('bits')
qdata = loaded.get('qdata')
print("scales:", None if scales is None else tuple(scales.shape))
print("zeros: ", None if zeros  is None else tuple(zeros.shape))
print("bits:  ", None if bits   is None else tuple(bits.shape))
print("qdata: ", None if qdata  is None else tuple(qdata.shape))
'''

r = c.exec(script)
print(r.stdout)
if r.error:
    print("ERROR:", r.error)
if r.stderr.strip():
    print("STDERR:", r.stderr)
