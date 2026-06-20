"""Quick load test for Qwen3-0.6B."""
import time, torch, sys
t0 = time.time()
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3-0.6B', torch_dtype='auto', device_map='cpu'
)
t = time.time() - t0
cfg = model.config
print(f'Loaded in {t:.1f}s', flush=True)
print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M', flush=True)
print(f'Heads: {cfg.num_attention_heads}, KV: {cfg.num_key_value_heads}', flush=True)
hd = cfg.hidden_size // cfg.num_attention_heads
print(f'q: {model.model.layers[0].self_attn.q_proj.weight.shape}', flush=True)
print(f'k: {model.model.layers[0].self_attn.k_proj.weight.shape}', flush=True)
print(f'v: {model.model.layers[0].self_attn.v_proj.weight.shape}', flush=True)
print(f'o: {model.model.layers[0].self_attn.o_proj.weight.shape}', flush=True)
print(f'GQA shared dim (o_in): {model.model.layers[0].self_attn.o_proj.weight.shape[1]}', flush=True)
print(f'q_out={cfg.hidden_size}, k_out={cfg.num_key_value_heads*hd}, v_out={cfg.num_key_value_heads*hd}', flush=True)
print('OK', flush=True)
sys.exit(0)
