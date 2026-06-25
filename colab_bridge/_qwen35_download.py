"""Run ICS+GPTQ on Qwen3.5-2B + benchmark PPL."""
import sys, time
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=2400)

script = r'''
import sys, os, time, importlib, copy, shutil
# Clear pycache first
for root, dirs, files in os.walk("/content/ICS"):
    for d in dirs:
        if d == "__pycache__":
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)
for mod_name in list(sys.modules):
    if mod_name.startswith("ics."):
        del sys.modules[mod_name]
if "ics" in sys.modules:
    del sys.modules["ics"]
sys.path.insert(0, "/content/ICS")
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ics.pipeline import quantize_model, ICSConfig
from ics.export import save_ics_model, load_ics_model, dequantized_state_dict
from scripts.colab_quantize_ics import DEFAULT_CALIBRATION as CALIB

# Download Qwen3.5-2B
print("=== downloading Qwen3.5-2B ===", flush=True)
t0 = time.time()
from huggingface_hub import snapshot_download
qwen35_path = snapshot_download(
    repo_id="Qwen/Qwen3.5-2B",
    local_dir="/content/Qwen3.5-2B",
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja", "tokenizer*", "vocab*", "merges*", "generation_config*"],
)
print(f"downloaded in {time.time()-t0:.1f}s to {qwen35_path}", flush=True)

import os
for f in sorted(os.listdir(qwen35_path)):
    p = os.path.join(qwen35_path, f)
    print(f"  {f}: {os.path.getsize(p)/1e6:.1f} MB")

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
print("\n=== loading model (bnb 4-bit) ===", flush=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(qwen35_path, quantization_config=bnb_cfg, device_map="cuda")
tok = AutoTokenizer.from_pretrained(qwen35_path)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

# Quick look at the architecture: how many layers? linear_attn vs self_attn?
total_layers = len(model.model.layers)
print(f"\ntotal layers: {total_layers}")
n_linear_attn = 0
n_self_attn = 0
for i, layer in enumerate(model.model.layers[:5]):
    has_linear_attn = hasattr(layer, 'linear_attn')
    has_self_attn = hasattr(layer, 'self_attn')
    if has_linear_attn: n_linear_attn += 1
    if has_self_attn: n_self_attn += 1
    print(f"  layer {i}: linear_attn={has_linear_attn}  self_attn={has_self_attn}")
# Check the last few layers too
for i, layer in enumerate(model.model.layers[-3:]):
    has_linear_attn = hasattr(layer, 'linear_attn')
    has_self_attn = hasattr(layer, 'self_attn')
    print(f"  layer {total_layers-3+i}: linear_attn={has_linear_attn}  self_attn={has_self_attn}")
print(f"first 5: linear_attn={n_linear_attn}, self_attn={n_self_attn}")
'''

t0 = time.time()
r = c.exec(script, timeout=2400)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[-3000:])
print(f"\nwall time: {time.time()-t0:.1f}s")
