"""Inspect bnb Linear4bit weights + dequantize path."""
import sys
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=300)

script = r'''
import sys
sys.path.insert(0, "/content/ICS")
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import bitsandbytes as bnb

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", quantization_config=bnb_cfg, device_map="cuda")

q = model.model.layers[0].self_attn.q_proj
g = model.model.layers[0].mlp.gate_proj
d = model.model.layers[0].mlp.down_proj

print("=== q_proj (Linear4bit) ===")
print("type:", type(q).__name__)
print("weight:", type(q.weight).__name__)
print("weight.shape:", tuple(q.weight.shape), "dtype:", q.weight.dtype)
print("out_features (config):", q.out_features, "in_features:", q.in_features)

print()
print("=== gate_proj (Linear4bit) ===")
print("weight:", type(g.weight).__name__)
print("weight.shape:", tuple(g.weight.shape), "dtype:", g.weight.dtype)
print("out_features (config):", g.out_features, "in_features:", g.in_features)

print()
print("=== dequantize_4bit on q.weight.data ===")
try:
    w_q = bnb.functional.dequantize_4bit(q.weight.data, q.weight.quant_state)
    print("q dequantized shape:", tuple(w_q.shape), "dtype:", w_q.dtype)
except Exception as e:
    print("q dequantize error:", e)

print()
print("=== dequantize_4bit on g.weight.data ===")
try:
    w_g = bnb.functional.dequantize_4bit(g.weight.data, g.weight.quant_state)
    print("g dequantized shape:", tuple(w_g.shape), "dtype:", w_g.dtype)
except Exception as e:
    print("g dequantize error:", e)

print()
print("=== can we cast module.weight in-place to fp16? ===")
try:
    # This is what bitsandbytes does internally for compute
    fp16_w = bnb.functional.dequantize_4bit(g.weight.data, g.weight.quant_state).to(torch.float16)
    print("fp16 shape:", tuple(fp16_w.shape))
except Exception as e:
    print("err:", e)
'''

r = c.exec(script, timeout=300)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[-2000:])
