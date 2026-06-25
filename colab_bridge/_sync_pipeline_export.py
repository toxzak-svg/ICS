"""Sync the updated pipeline.py + export.py to runtime."""
import sys, base64
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=300)

for local, remote in [
    (r"C:\Users\Zwmar\projects\ICS\ics\pipeline.py", "/content/ICS/ics/pipeline.py"),
    (r"C:\Users\Zwmar\projects\ICS\ics\export.py", "/content/ICS/ics/export.py"),
]:
    with open(local, "rb") as f:
        content = f.read()
    b64 = base64.b64encode(content).decode()
    script = f"""
import base64, hashlib
b64 = {b64!r}
data = base64.b64decode(b64)
with open({remote!r}, "wb") as f:
    f.write(data)
with open({remote!r}, "rb") as f:
    h = hashlib.sha256(f.read()).hexdigest()[:12]
print("written bytes:", len(data), "hash:", h, "to:", {remote!r})
"""
    r = c.exec(script)
    print(r.stdout)
    if r.error:
        print("ERR:", r.error[:500])
