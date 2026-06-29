"""Run ICS+GPTQ on Qwen3.5-2B + benchmark PPL."""
import sys, time
sys.path.insert(0, r'C:\Users\Zwmar\projects\ICS\colab_bridge')
from client import ColabClient

URL = "https://never-lecture-across-pocket.trycloudflare.com"
TOK = "KmzfnkVQWEHm1ydq-yx0J8oSeItYS5a2"
c = ColabClient(URL, TOK, timeout=3600)  # 60 min ceiling for 2B

script = r'''
import sys, os, time, importlib, copy, shutil
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
from ics.pipeline import quantize_model, ICSConfig, discover_chains
from ics.export import save_ics_model, load_ics_model, dequantized_state_dict
from scripts.colab_quantize_ics import DEFAULT_CALIBRATION as CALIB

# Load
print("=== loading Qwen3.5-2B (bnb 4-bit) ===", flush=True)
bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained("/content/Qwen3.5-2B", quantization_config=bnb_cfg, device_map="cuda")
tok = AutoTokenizer.from_pretrained("/content/Qwen3.5-2B")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print(f"loaded: {len(model.model.layers)} layers", flush=True)

# Quick chain discovery preview
chains = discover_chains(model, skip_modules=("lm_head", "embed_tokens", "norm", "mtp."))
print(f"discovered {len(chains)} chains", flush=True)
from collections import Counter
kinds = Counter(c.kind for c in chains)
print(f"kinds: {dict(kinds)}", flush=True)

# Show a few example chains
print(f"\nexample chains:", flush=True)
for c in chains[:5]:
    print(f"  {c.kind}: {c.members}  shared={c.shared_dim_size}", flush=True)
if len(chains) > 5:
    print(f"  ... and {len(chains)-5} more", flush=True)

# Run pipeline
cfg = ICSConfig(
    block_size=64, int4_fraction=1.0, int2_fraction=0.0, int1_fraction=0.0,
    method="composite", quant_method="gptq",
    gptq_group_size=128, gptq_percdamp=0.01, gptq_blocksize=128,
    max_calibration_length=256, calibration_texts=CALIB[:24],  # 24 samples
    skip_modules=("lm_head", "embed_tokens", "norm", "mtp."),
)

t0 = time.time()
print("\n=== quantize_model ===", flush=True)
result = quantize_model(model, tok, cfg, device="cuda", show_progress=True)
print(f"\n[pipeline] done in {time.time()-t0:.1f}s", flush=True)
print(f"perms: {len(result.perms)}  quant: {len(result.quant)}  layer_perms: {len(result.layer_perms)}", flush=True)

# Save
save_ics_model(result, "/content/qwen3.5-2b-ics-gptq-int4", tokenizer=tok, source_model_dir="/content/Qwen3.5-2B")
print("[save] done", flush=True)

# Compression report
art = "/content/qwen3.5-2b-ics-gptq-int4"
for f in sorted(os.listdir(art)):
    p = os.path.join(art, f)
    print(f"  {f}: {os.path.getsize(p)/1e6:.1f} MB", flush=True)

# PPL benchmark
print("\n=== PPL benchmark ===", flush=True)
loaded = load_ics_model(art)
sd = dequantized_state_dict(loaded)
print(f"loaded {len(sd)} dequantized layers", flush=True)

ref = AutoModelForCausalLM.from_pretrained("/content/Qwen3.5-2B", dtype=torch.bfloat16, device_map="cuda")

# Use a few different eval sets
texts = [
    "The transformer architecture uses self-attention to model long-range dependencies in sequences.",
    "Quantization maps continuous values to a discrete grid; per-block scaling factors preserve precision.",
    "The Fisher information matrix measures how sensitive the model loss is to perturbations in parameters.",
    "Climate change is driving more frequent extreme weather events across the globe.",
    "The human brain contains roughly 86 billion neurons connected by trillions of synapses.",
    "In 1969, the Apollo 11 mission landed the first humans on the Moon, marking a milestone in human history.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
    "Renewable energy sources include solar, wind, hydro, and geothermal power generation.",
    "DNA encodes genetic information using four nucleotide bases: adenine, thymine, guanine, and cytosine.",
    "The Gated DeltaNet is a linear attention model with a recurrence over a compressed state, designed to replace softmax attention for long sequences.",
]
joined = "\\n\\n".join(texts)
enc = tok(joined, return_tensors="pt")
input_ids = enc["input_ids"]
print(f"eval tokens: {input_ids.shape[-1]}", flush=True)

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

p_bf16, n = ppl(ref, input_ids)
print(f"BF16 PPL: {p_bf16:.3f}  ({n} tokens)", flush=True)

ics_model = copy.deepcopy(ref)
ics_model.load_state_dict(sd, strict=False)
p, n = ppl(ics_model, input_ids)
print(f"ICS+GPTQ int4 PPL: {p:.3f}  ({n} tokens)  hit = {p/p_bf16:.2f}x BF16", flush=True)

# Short text forward diff
text = "The quick brown fox jumps over the lazy dog."
enc2 = tok(text, return_tensors="pt").to("cuda")
with torch.no_grad():
    o_ref = ref(input_ids=enc2["input_ids"]).logits
    o_ics = ics_model(input_ids=enc2["input_ids"]).logits
diff = (o_ics - o_ref).abs()
print(f"\\nshort forward diff: max={diff.max().item():.4f} mean={diff.mean().item():.4f}", flush=True)
print(f"  next-token: ref={tok.decode([o_ref[0,-1].argmax().item()])!r}  ics={tok.decode([o_ics[0,-1].argmax().item()])!r}", flush=True)
'''

t0 = time.time()
r = c.exec(script, timeout=3600)
print(r.stdout)
if r.error:
    print("ERROR:", r.error[-3000:])
print(f"\nwall time: {time.time()-t0:.1f}s")
