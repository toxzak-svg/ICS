"""Download Qwen3-0.6B (BF16) and Q4_K_M GGUF in parallel; install llama-cpp-python."""
import sys, time
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=1500)

# Start both downloads in a single exec; the model is ~1.2 GB and the GGUF ~500 MB.
script = r'''
import subprocess, time, os
from concurrent.futures import ThreadPoolExecutor

def download_qwen3():
    from huggingface_hub import snapshot_download
    t0 = time.time()
    p = snapshot_download(
        repo_id="Qwen/Qwen3-0.6B",
        local_dir="/content/Qwen3-0.6B",
        allow_patterns=[
            "*.safetensors", "*.json", "*.txt", "*.jinja", "tokenizer*", "vocab*", "merges*", "generation_config*"
        ],
    )
    print(f"[qwen3] downloaded to {p} in {time.time()-t0:.1f}s")
    sz = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)
    print(f"[qwen3] total size: {sz/1e9:.2f} GB")
    return p

def download_gguf():
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    p = hf_hub_download(
        repo_id="unsloth/Qwen3-0.6B-GGUF",
        filename="Qwen3-0.6B-Q4_K_M.gguf",
        local_dir="/content/gguf",
    )
    print(f"[gguf] downloaded to {p} in {time.time()-t0:.1f}s")
    print(f"[gguf] size: {os.path.getsize(p)/1e6:.1f} MB")
    return p

def install_llama_cpp():
    import subprocess
    t0 = time.time()
    # Try the binary first (much faster than compiling)
    r = subprocess.run(["pip", "install", "-q", "llama-cpp-python"], capture_output=True, text=True, timeout=600)
    print(f"[llama-cpp] install took {time.time()-t0:.1f}s")
    print("[llama-cpp] stdout tail:", r.stdout[-300:])
    print("[llama-cpp] stderr tail:", r.stderr[-300:])
    try:
        import llama_cpp
        print("[llama-cpp] version:", llama_cpp.__version__)
    except Exception as e:
        print("[llama-cpp] import failed:", e)

with ThreadPoolExecutor(max_workers=3) as ex:
    futs = [ex.submit(download_qwen3), ex.submit(download_gguf), ex.submit(install_llama_cpp)]
    for f in futs:
        try:
            f.result()
        except Exception as e:
            print("task error:", e)
'''

t0 = time.time()
r = c.exec(script, timeout=1500)
print(r.stdout)
if r.error:
    print("ERROR:", r.error)
if r.stderr.strip():
    print("STDERR:", r.stderr[-1000:])
print(f"wall time: {time.time()-t0:.1f}s")
