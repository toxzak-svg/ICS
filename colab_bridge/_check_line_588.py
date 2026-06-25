"""Check the actual line 588 in the runtime."""
import sys
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=60)

script = r'''
import hashlib
with open("/content/ICS/ics/pipeline.py", "rb") as f:
    print("hash:", hashlib.sha256(f.read()).hexdigest()[:12])
with open("/content/ICS/ics/pipeline.py") as f:
    lines = f.readlines()
for i in range(584, 600):
    print(f"  {i+1}: {lines[i].rstrip()}")
print("---")
# Find the perms[key] = perm and chain_members[key] = lines
for i, line in enumerate(lines, 1):
    if "perms[key] = perm" in line:
        print(f"  perms[key] = perm at line {i}")
    if "chain_members[key] =" in line:
        print(f"  chain_members[key] = at line {i}")
    if "chain_members = " in line or "chain_members: " in line:
        print(f"  chain_members assignment at line {i}: {line.rstrip()}")
'''
r = c.exec(script)
print(r.stdout)
