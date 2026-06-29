"""Clear __pycache__ and re-import."""
import sys
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=60)

script = r'''
import os, shutil, sys
# Clear pycache
for root, dirs, files in os.walk("/content/ICS"):
    for d in dirs:
        if d == "__pycache__":
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)
            print("removed:", os.path.join(root, d))
print("cache cleared")
# Now do a fresh import
for mod_name in list(sys.modules):
    if mod_name.startswith("ics."):
        del sys.modules[mod_name]
if "ics" in sys.modules:
    del sys.modules["ics"]
import ics.pipeline
# verify the ICSResult has chain_members
from dataclasses import fields
flds = [f.name for f in fields(ics.pipeline.ICSResult)]
print("ICSResult fields:", flds)
print("chain_members" in flds)
'''
r = c.exec(script)
print(r.stdout)
