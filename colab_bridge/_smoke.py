"""Install ICS deps and run the three smoke test suites."""
import sys, time
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"

c = ColabClient(URL, TOK, timeout=600)

# 1. Install requirements
print("--- installing requirements ---")
t0 = time.time()
r = c.exec("""
import subprocess
r = subprocess.run(['pip', 'install', '-q', '-r', '/content/ICS/requirements.txt'], capture_output=True, text=True, timeout=300)
print('stdout tail:', r.stdout[-300:])
print('stderr tail:', r.stderr[-300:])
""", timeout=400)
print(r.stdout)
if r.error:
    print("ERROR:", r.error)
print(f"pip took {time.time()-t0:.1f}s\n")

# 2. Smoke tests, sequentially with separators
for script in ['scripts/test_correctness.py', 'scripts/test_pipeline_smoke.py', 'scripts/test_gqa.py']:
    print(f"--- running {script} ---")
    t0 = time.time()
    r = c.exec(f"""
import subprocess
r = subprocess.run(['python', '/content/ICS/{script}'], capture_output=True, text=True, timeout=300)
print('exit:', r.returncode)
print('stdout:')
print(r.stdout)
if r.stderr.strip():
    print('STDERR:')
    print(r.stderr)
""", timeout=400)
    # the print above goes via stdout
    print(r.stdout)
    print(f"({time.time()-t0:.1f}s)")
    print()
