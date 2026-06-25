"""Clear __pycache__ and re-run GPTQ + verify dequant is correct."""
import sys, time
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=1800)

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

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", quantization_config=bnb_cfg, device_map="cuda")
tok = AutoTokenizer.from_pretrained("/content/Qwen3-0.6B")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

cfg = ICSConfig(
    block_size=64, int4_fraction=1.0, int2_fraction=0.0, int1_fraction=0.0,
    method="composite", quant_method="gptq",
    gptq_group_size=128, gptq_percdamp=0.01, gptq_blocksize=128,
    max_calibration_length=256, calibration_texts=CALIB[:32],
    skip_modules=("lm_head", "embed_tokens", "norm", "mtp."),
)

t0 = time.time()
print("=== quantize_model ===", flush=True)
result = quantize_model(model, tok, cfg, device="cuda", show_progress=True)
print(f"\n[pipeline] done in {time.time()-t0:.1f}s")
save_ics_model(result, "/content/qwen3-0.6b-ics-gptq-int4", tokenizer=tok, source_model_dir="/content/Qwen3-0.6B")

# Verify the dequant
print("\n=== verify ===")
loaded = load_ics_model("/content/qwen3-0.6b-ics-gptq-int4")
print(f"loaded chain_members: {len(loaded['chain_members'])}")
print(f"loaded layer_perms: {len(loaded['layer_perms'])}")

ref = AutoModelForCausalLM.from_pretrained("/content/Qwen3-0.6B", dtype=torch.bfloat16, device_map="cpu")
sd_ref = ref.state_dict()
sd = dequantized_state_dict(loaded)
for name in ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.gate_proj.weight", "model.layers.5.self_attn.o_proj.weight"]:
    w_ics = sd[name].float()
    w_ref = sd_ref[name].float()
    diff = (w_ics - w_ref).abs()
    print(f"  {name}: max_diff={diff.max().item():.4f} mean_diff={diff.mean().item():.4f}")

# PPL
texts = [
    "The transformer architecture uses self-attention to model long-range dependencies in sequences.",
    "Quantization maps continuous values to a discrete grid; per-block scaling factors preserve precision.",
    "The Fisher information matrix measures how sensitive the model loss is to perturbations in parameters.",
    "Climate change is driving more frequent extreme weather events across the globe.",
    "The human brain contains roughly 86 billion neurons connected by trillions of synapses.",
    "In 1969, the Apollo 11 mission landed the first humans on the Moon, marking a milestone in human history.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
    "Renewable energy sources include solar, wind, hydro, and geothermal power generation.",
]
joined = "\n\n".join(texts)
enc = tok(joined, return_tensors="pt")
input_ids = enc["input_ids"]

def ppl(m, ids, block=128):
    m.eval()
    nll, nt = 0.0, 0
    with torch.no_grad():
        for i in range(0, ids.shape[-1] - 1, block):
            chunk = ids[:, i:i+block+1].to(m.device)
            o = m(input_ids=chunk, use_cache=False).logits[:, :-1, :].float()
            t = chunk[:, 1:]
            nll += torch.nn.functional.cross_entropy(o.reshape(-1, o.size(-1)), t.reshape(-1), reduction="sum").item()
            nt += t.numel()
    return float(torch.tensor(nll / nt).exp().item()), nt

p_bf16, n = ppl(ref.to("cuda"), input_ids)
print(f"\nBF16 PPL: {p_bf16:.3f}  ({n} tokens)")

ics_model = copy.deepcopy(ref.to("cuda"))
ics_model.load_state_dict(sd, strict=False)
p, n = ppl(ics_model, input_ids)
print(f"ICS+GPTQ int4 PPL: {p:.3f}  ({n} tokens)  hit = {p/p_bf16:.2f}x BF16")
'''

t0 = time.time()
r = c.exec(script, timeout=1800)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[-3000:])
print(f"\nwall time: {time.time()-t0:.1f}s")
