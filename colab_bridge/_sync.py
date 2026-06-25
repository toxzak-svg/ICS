"""Sync one file from local to Colab runtime via the bridge."""
import sys, base64
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"

LOCAL = r"C:\Users\Zwmar\projects\ICS\scripts\test_pipeline_smoke.py"
REMOTE = "/content/ICS/scripts/test_pipeline_smoke.py"

with open(LOCAL, "rb") as f:
    content = f.read()
b64 = base64.b64encode(content).decode()
print(f"local size: {len(content)} bytes, b64: {len(b64)} chars")

# Single exec: decode + write. We send the b64 as a variable, not inlined.
script = f"""
import base64, hashlib
b64 = {b64!r}
data = base64.b64decode(b64)
with open({REMOTE!r}, "wb") as f:
    f.write(data)
with open({REMOTE!r}, "rb") as f:
    print("new hash:", hashlib.sha256(f.read()).hexdigest()[:12])
print("written bytes:", len(data))
"""

c = ColabClient(URL, TOK, timeout=300)
r = c.exec(script)
print(r)
print("stderr:", r.stderr)
