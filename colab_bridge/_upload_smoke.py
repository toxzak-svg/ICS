"""Investigate the existing qwen3-0.6b-ics-smoke artifact.

Load it, dequantize, run a forward pass. Check that:
  1. metadata round-trips (chains, dims, scales, zeros, bits are sane)
  2. dequantized state dict can be loaded into a Qwen3-0.6B HF model
  3. forward pass is finite (no NaN/Inf)
  4. the output is non-trivially close to the dense baseline (or at least not pure noise)
"""
import sys, time, os
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=600)

# Upload the smoke artifact from local to /content/ICS
print("--- uploading smoke artifact from local ---")
for fname in [
    'qwen3-0.6b-ics-smoke/model.safetensors',
    'qwen3-0.6b-ics-smoke/scales.safetensors',
    'qwen3-0.6b-ics-smoke/zeros.safetensors',
    'qwen3-0.6b-ics-smoke/bits.safetensors',
    'qwen3-0.6b-ics-smoke/ics_meta.json',
    'qwen3-0.6b-ics-smoke/synthetic_tokenizer.txt',
]:
    local = os.path.join(r'C:\Users\Zwmar\projects\ICS', fname)
    if not os.path.exists(local):
        print(f"  skip {fname} (missing)")
        continue
    res = c.upload(local, '/content/ICS/' + fname)
    print(f"  uploaded {fname}: {res}")

# Sanity check what's on the remote side
print("\n--- remote smoke artifact ---")
r = c.exec("""
import os
sm = '/content/ICS/qwen3-0.6b-ics-smoke'
for f in sorted(os.listdir(sm)):
    p = os.path.join(sm, f)
    print(f'  {f}: {os.path.getsize(p)} bytes')
""")
print(r.stdout)
