"""Sync the patched permutation.py to the runtime, then re-run the quant."""
import sys, base64, time
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=2400)

LOCAL = r"C:\Users\Zwmar\projects\ICS\ics\permutation.py"
REMOTE = "/content/ICS/ics/permutation.py"

with open(LOCAL, "rb") as f:
    content = f.read()
b64 = base64.b64encode(content).decode()
print(f"local size: {len(content)} bytes")

script = f"""
import base64, hashlib
b64 = {b64!r}
data = base64.b64decode(b64)
with open({REMOTE!r}, "wb") as f:
    f.write(data)
with open({REMOTE!r}, "rb") as f:
    h = hashlib.sha256(f.read()).hexdigest()[:12]
print("written bytes:", len(data), "hash:", h)
"""
r = c.exec(script)
print(r.stdout)
if r.error:
    print("SYNC ERROR:", r.error)
