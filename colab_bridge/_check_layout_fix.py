"""Check the runtime's pipeline.py for the qdata layout fix."""
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
    content = f.read()
# Find the qdata_pieces loop
idx = content.find("qdata_pieces = []")
if idx > 0:
    print(content[idx:idx+500])
else:
    print("NOT FOUND")
    # Find the old Q.reshape(-1)
    idx = content.find("Q.reshape(-1)")
    print(content[max(0,idx-100):idx+200])
'''
r = c.exec(script)
print(r.stdout)
