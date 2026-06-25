"""Verify the runtime uses the block-row-major storage."""
import sys
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=60)

script = r'''
import hashlib
with open("/content/ICS/ics/pipeline.py", "rb") as f:
    h = hashlib.sha256(f.read()).hexdigest()[:12]
print("hash:", h)
# Also check ics/__pycache__
import os
for f in os.listdir("/content/ICS/ics/__pycache__"):
    print("cache:", f, os.path.getsize(f"/content/ICS/ics/__pycache__/{f}"))
# Find the qdata layout code
with open("/content/ICS/ics/pipeline.py") as f:
    content = f.read()
idx = content.find("block_q = Q[:, start:end]")
if idx > 0:
    print("FOUND block_q = Q[:, start:end] line")
    print(content[max(0,idx-200):idx+200])
else:
    print("NOT FOUND block_q = Q[:, start:end]")
    # Maybe the old Q.reshape(-1) is still there
    idx2 = content.find("Q.reshape(-1).contiguous()")
    if idx2 > 0:
        print("FOUND old Q.reshape(-1) line at", idx2)
'''

r = c.exec(script)
print(r.stdout)
