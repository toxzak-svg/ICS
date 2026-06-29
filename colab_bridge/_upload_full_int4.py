"""Upload the full-int4 artifact (the real one with 56 perms)."""
import sys, os, time
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=900)

art_dir = r'C:\Users\Zwmar\projects\ICS\qwen3-0.6b-ics-full-int4'
files = [
    'model.safetensors',
    'scales.safetensors',
    'zeros.safetensors',
    'bits.safetensors',
    'ics_meta.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'chat_template.jinja',
]

t_total = time.time()
for fname in files:
    local = os.path.join(art_dir, fname)
    if not os.path.exists(local):
        print(f"  skip {fname} (missing locally)")
        continue
    sz = os.path.getsize(local)
    print(f"  uploading {fname} ({sz/1e6:.1f} MB)...", end=' ', flush=True)
    t0 = time.time()
    res = c.upload(local, '/content/ICS/qwen3-0.6b-ics-full-int4/' + fname)
    print(f"  done in {time.time()-t0:.1f}s")
print(f"\ntotal upload time: {time.time()-t_total:.1f}s")
