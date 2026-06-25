"""Debug: print result.chain_members right before save."""
import sys, time
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=1500)

script = r'''
import sys, os, time, importlib, copy
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
from ics.export import save_ics_model

# Small calibration
CALIB = [
    "The quick brown fox jumps over the lazy dog.",
    "In transformer architectures, attention computes a weighted sum.",
    "Quantization maps continuous values to a discrete grid.",
    "The Fisher information matrix measures sensitivity to perturbations.",
    "The Eiffel Tower was completed in 1889 and stands 330 meters tall.",
    "Renewable energy sources include solar, wind, hydro, and geothermal.",
    "The human brain contains roughly 86 billion neurons.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose.",
]

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", quantization_config=bnb_cfg, device_map="cuda")
tok = AutoTokenizer.from_pretrained("/content/Qwen3-0.6B")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

cfg = ICSConfig(
    block_size=64, int4_fraction=1.0, int2_fraction=0.0, int1_fraction=0.0,
    method="composite", quant_method="gptq",
    gptq_group_size=128, gptq_percdamp=0.01, gptq_blocksize=128,
    max_calibration_length=128, calibration_texts=CALIB,
    skip_modules=("lm_head", "embed_tokens", "norm", "mtp."),
)

t0 = time.time()
print("=== quantize_model ===", flush=True)
result = quantize_model(model, tok, cfg, device="cuda", show_progress=True)
print(f"\n[pipeline] done in {time.time()-t0:.1f}s", flush=True)
print(f"result.chain_members: type={type(result.chain_members).__name__} len={len(result.chain_members) if result.chain_members else 'None'}")
if result.chain_members:
    first_key = list(result.chain_members.keys())[0]
    print(f"  first key: {first_key}")
    print(f"  first value: {result.chain_members[first_key]}")
print(f"result.layer_perms: type={type(result.layer_perms).__name__} len={len(result.layer_perms) if result.layer_perms else 'None'}")

# Inspect the dataclass
from dataclasses import fields
print(f"ICSResult fields: {[f.name for f in fields(result)]}")
'''

t0 = time.time()
r = c.exec(script, timeout=1500)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[-3000:])
print(f"\nwall time: {time.time()-t0:.1f}s")
