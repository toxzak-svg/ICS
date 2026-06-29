"""Run ICS+GPTQ on Qwen3.5-2B as a background subprocess and poll."""
import sys, time
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=120)

# 1. Write the pipeline as a script to the runtime
script_setup = r'''
import os, subprocess
# Make sure cache is clean
for root, dirs, files in os.walk("/content/ICS"):
    for d in dirs:
        if d == "__pycache__":
            import shutil
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)

# Write the pipeline script to /content/run_pipeline.py
script = """
import sys, os, time, importlib
for mod_name in list(sys.modules):
    if mod_name.startswith(\"ics.\"):
        del sys.modules[mod_name]
if \"ics\" in sys.modules:
    del sys.modules[\"ics\"]
sys.path.insert(0, \"/content/ICS\")
os.environ[\"PYTHONUNBUFFERED\"] = \"1\"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ics.pipeline import quantize_model, ICSConfig
from ics.export import save_ics_model
from scripts.colab_quantize_ics import DEFAULT_CALIBRATION as CALIB

print(\"[start]\", flush=True)
bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=\"nf4\", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(\"/content/Qwen3.5-2B\", quantization_config=bnb_cfg, device_map=\"cuda\")
tok = AutoTokenizer.from_pretrained(\"/content/Qwen3.5-2B\")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print(f\"[loaded] {len(model.model.layers)} layers\", flush=True)

cfg = ICSConfig(
    block_size=64, int4_fraction=1.0, int2_fraction=0.0, int1_fraction=0.0,
    method=\"composite\", quant_method=\"gptq\",
    gptq_group_size=128, gptq_percdamp=0.01, gptq_blocksize=128,
    max_calibration_length=256, calibration_texts=CALIB[:24],
    skip_modules=(\"lm_head\", \"embed_tokens\", \"norm\", \"mtp.\"),
)

t0 = time.time()
print(\"[start_pipeline]\", flush=True)
result = quantize_model(model, tok, cfg, device=\"cuda\", show_progress=True)
elapsed = time.time() - t0
print(f\"[done_pipeline] {elapsed:.1f}s  perms={len(result.perms)}  quant={len(result.quant)}\", flush=True)

save_ics_model(result, \"/content/qwen3.5-2b-ics-gptq-int4\", tokenizer=tok, source_model_dir=\"/content/Qwen3.5-2B\")
print(\"[done_save]\", flush=True)
"""

with open("/content/run_pipeline.py", "w") as f:
    f.write(script)

# Start it as a background nohup process
r = subprocess.run(
    ["nohup", "python3", "-u", "/content/run_pipeline.py", ">", "/content/pipeline.log", "2>&1", "&"],
    capture_output=True, text=True,
)
print("started, pid info:")
import time
time.sleep(2)
# Check if it's running
r2 = subprocess.run(["ps", "aux"], capture_output=True, text=True)
for line in r2.stdout.splitlines():
    if "run_pipeline" in line or "python" in line.lower() and "qwen" in line.lower():
        print("PROC:", line)
print("\\nlog start:")
with open("/content/pipeline.log") as f:
    print(f.read()[:1000])
'''

r = c.exec(script_setup)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[:500])
