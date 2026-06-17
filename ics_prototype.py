!pip install -q transformers accelerate bitsandbytes

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'google/gemma-4-12b'
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='auto',
    load_in_8bit=True
)

def apply_ics(model):
    w_q = model.model.layers[0].self_attn.q_proj.weight.data.clone()
    w_k = model.model.layers[0].self_attn.k_proj.weight.data.clone()
    perm_q = torch.argsort(w_q.abs().sum(dim=1), descending=True)
    w_q = w_q[perm_q]
    model.model.layers[0].self_attn.q_proj.weight.data = w_q
    w_k = w_k[perm_q]
    model.model.layers[0].self_attn.k_proj.weight.data = w_k
    return model

model = apply_ics(model)

inputs = tokenizer('Hello', return_tensors='pt').to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=10)
print(tokenizer.decode(out[0], skip_special_tokens=True))