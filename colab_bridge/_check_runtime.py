"""Check the runtime's pipeline.py to verify it has the fix."""
import sys
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=60)

script = r'''
import hashlib, os
p = "/content/ICS/ics/pipeline.py"
print("hash:", hashlib.sha256(open(p,"rb").read()).hexdigest()[:12])
print("size:", os.path.getsize(p))
# Show lines 608-616
with open(p) as f:
    lines = f.readlines()
for i in range(605, min(620, len(lines))):
    print(f"  {i+1}: {lines[i].rstrip()}")
'''
r = c.exec(script)
print(r.stdout)
