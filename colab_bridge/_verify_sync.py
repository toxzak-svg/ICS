"""Verify the runtime's pipeline.py has the new function."""
import sys
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=60)

script = r'''
import hashlib, os
p = "/content/ICS/ics/pipeline.py"
with open(p, "rb") as f:
    print("runtime hash:", hashlib.sha256(f.read()).hexdigest()[:12])
print("size:", os.path.getsize(p))
with open(p) as f:
    content = f.read()
print("_dequantize_bnb_inplace in runtime?", "_dequantize_bnb_inplace" in content)
# Also show the line near the new function
idx = content.find("_dequantize_bnb_inplace")
if idx > 0:
    print("first 200 chars from match:")
    print(content[idx:idx+200])
'''
r = c.exec(script)
print(r.stdout)
