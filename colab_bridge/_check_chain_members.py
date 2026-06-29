"""Check the runtime file for chain_members."""
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
print("chain_members field present:", "chain_members: dict" in content)
print("chain_members in return:", "chain_members=chain_members" in content)
print("chain_members in pipeline build:", "chain_members[key] =" in content)
'''
r = c.exec(script)
print(r.stdout)
