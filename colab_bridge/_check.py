"""Simple bridge check."""
import sys
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=60)

script = r'''
print("bridge alive")
import os
print("Qwen3.5-2B exists:", os.path.exists("/content/Qwen3.5-2B"))
# Check if any python pipeline is running
import subprocess
r = subprocess.run(["ps", "aux"], capture_output=True, text=True)
for line in r.stdout.splitlines():
    if "run_pipeline" in line or "colab_quantize" in line:
        print("PIPELINE:", line)
'''
try:
    r = c.exec(script, timeout=30)
    print(r.stdout)
except Exception as e:
    print("ERR:", e)
