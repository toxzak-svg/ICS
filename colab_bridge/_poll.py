"""Poll pipeline log."""
import sys, time
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=60)

for i in range(20):
    script = f'''
import os, time
log = "/content/pipeline.log"
if os.path.exists(log):
    with open(log) as f:
        content = f.read()
    print(f"--- t={i*30}s, log size={{len(content)}} ---")
    print(content[-2500:])
else:
    print("no log yet")
import subprocess
r = subprocess.run(["ps", "aux"], capture_output=True, text=True)
for line in r.stdout.splitlines():
    if "run_pipeline" in line:
        print("PROC:", line)
        break
art = "/content/qwen3.5-2b-ics-gptq-int4"
if os.path.exists(art):
    print("ART exists:", os.listdir(art))
'''
    r = c.exec(script, timeout=30)
    print(f"\n=== poll {i} (t={i*30}s) ===")
    print(r.stdout)
    if "[done_save]" in r.stdout or "[done_pipeline]" in r.stdout:
        print("\n*** PIPELINE COMPLETE ***")
        break
    time.sleep(30)
