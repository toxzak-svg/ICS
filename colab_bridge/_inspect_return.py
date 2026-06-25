"""Inspect the return ICSResult block in runtime pipeline.py."""
import sys
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=60)

script = r'''
with open("/content/ICS/ics/pipeline.py") as f:
    content = f.read()
idx = content.find("return ICSResult(")
print(content[idx:idx+400])
print("---")
# also check the chain_members build
idx2 = content.find("chain_members[key] =")
print(content[idx2:idx2+250])
'''
r = c.exec(script)
print(r.stdout)
