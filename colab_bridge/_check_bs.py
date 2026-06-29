"""Check the block_size in the saved meta."""
import sys
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=60)

script = r'''
import json
with open("/content/qwen3-0.6b-ics-gptq-int4/ics_meta.json") as f:
    meta = json.load(f)
print("meta block_size:", meta["block_size"])
print("meta gptq_group_size:", meta.get("gptq_group_size"))
# pick a few layer entries
for name in list(meta["layers"].keys())[:3]:
    print(f"  {name}: {meta['layers'][name]}")
'''
r = c.exec(script)
print(r.stdout)
