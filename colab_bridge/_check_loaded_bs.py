"""Check the loaded QT's block_size."""
import sys
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=60)

script = r'''
import sys, importlib
for mod_name in list(sys.modules):
    if mod_name.startswith("ics."):
        del sys.modules[mod_name]
if "ics" in sys.modules:
    del sys.modules["ics"]
sys.path.insert(0, "/content/ICS")
import torch
from ics.export import load_ics_model
loaded = load_ics_model("/content/qwen3-0.6b-ics-gptq-int4")
for name in list(loaded["layers"].keys())[:3]:
    qt = loaded["layers"][name]
    print(f"  {name}: block_size={qt.block_size} original_shape={qt.original_shape} scales.shape={tuple(qt.scales.shape)}")
'''

r = c.exec(script)
print(r.stdout)
