"""Run the int4-only ICS quant on Qwen3-0.6B (match the existing artifact)."""
import sys, time
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=2400)  # 40 min ceiling for the full quant

script = r'''
import subprocess, time, os, sys
os.environ["PYTHONUNBUFFERED"] = "1"
t0 = time.time()
# int4-only: match qwen3-0.6b-ics-full-int4's int4-fraction=1.0 settings
r = subprocess.run(
    [
        "python", "/content/ICS/scripts/colab_quantize_ics.py",
        "--model", "/content/Qwen3-0.6B",
        "--output", "/content/qwen3-0.6b-ics-fresh-int4",
        "--int4-fraction", "1.0",
        "--int2-fraction", "0.0",
        "--int1-fraction", "0.0",
        "--max-calibration-samples", "64",
        "--max-calibration-length", "256",
        "--block-size", "64",
        "--method", "composite",
    ],
    capture_output=True, text=True, timeout=2400,
)
print("--- quant stdout ---")
print(r.stdout)
print("--- quant stderr (last 800 chars) ---")
print(r.stderr[-800:])
print(f"\nreturn code: {r.returncode}")
print(f"elapsed: {time.time()-t0:.1f}s")
if os.path.exists("/content/qwen3-0.6b-ics-fresh-int4"):
    print("\noutput artifact:")
    for f in sorted(os.listdir("/content/qwen3-0.6b-ics-fresh-int4")):
        p = os.path.join("/content/qwen3-0.6b-ics-fresh-int4", f)
        print(f"  {f}: {os.path.getsize(p)/1e6:.2f} MB")
'''

t0 = time.time()
r = c.exec(script, timeout=2400)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[:2000])
print(f"\nwall time: {time.time()-t0:.1f}s")
