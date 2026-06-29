"""Check the GPTQ path in the runtime's pipeline.py."""
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
# Find the QuantizedTensor construction in the GPTQ path
import re
for m in re.finditer(r"QuantizedTensor\(", content):
    start = m.start()
    print(f"--- at line ~{content[:start].count(chr(10))+1} ---")
    print(content[start:start+400])
    print()
'''
r = c.exec(script)
print(r.stdout)
