"""Debug the Fisher/weight mismatch in the quant step."""
import sys
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=600)

script = r'''
import sys
sys.path.insert(0, "/content/ICS")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ics.pipeline import discover_chains, ICSConfig, compute_fisher, _dequantize_bnb_inplace, _get_module, fisher_to_per_channel

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", quantization_config=bnb_cfg, device_map="cuda")
tok = AutoTokenizer.from_pretrained("/content/Qwen3-0.6B")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

chains = discover_chains(model, skip_modules=ICSConfig().skip_modules)
print(f"chains: {len(chains)}")
ch0 = chains[0]
print(f"chain 0: members={ch0.members} shared={ch0.shared_dim_size} fisher_source={ch0.fisher_source}")

# Run fisher on just chain 0 to debug
fisher = compute_fisher(
    model, tok,
    ["The quick brown fox.", "In transformer architectures, attention computes a weighted sum."],
    layer_filter=lambda n, m: n in ch0.members,
    max_length=64, device="cuda", show_progress=False,
)
print(f"\nfisher computed for {len(fisher)} layers:")
for name, fs in fisher.items():
    print(f"  {name}: fisher.shape={tuple(fs.fisher.shape)}  n_samples={fs.n_samples}")

# Dequantize
n = _dequantize_bnb_inplace(model)
print(f"\ndequantized {n} layers")

# Now check the post-dequant layer shapes
for n in ch0.members:
    mod = _get_module(model, n)
    w = mod.weight
    print(f"  {n}: type={type(mod).__name__} weight.shape={tuple(w.shape)} dtype={w.dtype}")

# Also check the W.shape[1] for o_proj (the failing dim)
o = _get_module(model, ch0.members[1])
W = o.weight.detach().float()
print(f"\no_proj weight.shape = {tuple(W.shape)}, W.shape[1] = {W.shape[1]}")
'''

r = c.exec(script, timeout=300)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[-2500:])
