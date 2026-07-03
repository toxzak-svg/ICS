"""
Isomorphic Channel Sorting (ICS) — Colab T4 run script.

Targets gemma-4-27b (or any HF causal LM) on the cheapest possible GPU
(Colab free T4, 15GB VRAM). Loads the model in 4-bit (NF4) via
BitsAndBytes, computes activation-Fisher Information on a small
calibration set, finds the joint permutation for each linear chain,
applies it, then quantizes per-block with INT4/INT2/INT1 mixed
precision. Saves the result to a directory of safetensors.

Usage in Colab:
    !git clone <this-repo>
    %cd ICS
    !pip install -q -r requirements.txt
    !python scripts/colab_quantize_ics.py \\
        --model google/gemma-4-27b \\
        --output ./gemma-4-27b-ics \\
        --max-calibration-samples 64 \\
        --max-calibration-length 256

Notes on T4 fitting:
    27B params in NF4 is ~14GB. Activation Fisher requires storing
    forward activations and computing backward gradients, which adds
    ~2-4GB depending on sequence length. We use gradient checkpointing
    + small max_length (256) to keep peak VRAM around 13-14GB.
    If you OOM, drop --max-calibration-length to 128 first.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import torch

# Make `ics` importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ics.pipeline import ICSConfig, quantize_model
from ics.export import save_ics_model


# A small but diverse calibration set. We use a mix of wikitext-style
# prose, code, and chat to get a stable Fisher estimate. ~64 samples
# is enough for the Fisher sum to converge at block granularity.
DEFAULT_CALIBRATION = [
    "The quick brown fox jumps over the lazy dog.",
    "In transformer architectures, multi-head attention computes a weighted sum of value vectors.",
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    "Quantization maps continuous weights to a discrete grid, with a scaling factor per block.",
    "The capital of France is Paris. The capital of Japan is Tokyo.",
    "Neural network training uses backpropagation to compute gradients via the chain rule.",
    "import torch\nx = torch.randn(3, 3)\nprint(x @ x.T)",
    "Once upon a time, in a small village by the sea, there lived a fisherman.",
    "The Fisher Information matrix measures the sensitivity of the loss to parameter perturbations.",
    "Recipe: mix flour, water, and yeast. Let it rise for an hour, then bake.",
    "In 1969, the Apollo 11 mission landed the first humans on the Moon.",
    "Climate change is driving more frequent extreme weather events worldwide.",
    "The Pythagorean theorem states that a^2 + b^2 = c^2 for right triangles.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
    "Bitcoin is a decentralized digital currency, without a central authority.",
    "Mount Everest, the highest mountain on Earth, stands at 8,849 meters.",
    "The human brain contains approximately 86 billion neurons.",
    "DNA encodes genetic information using four nucleotide bases: A, T, G, C.",
    "Shakespeare wrote 37 plays and 154 sonnets during his lifetime.",
    "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
    "Black holes have such strong gravity that nothing, not even light, can escape.",
    "The periodic table organizes chemical elements by atomic number.",
    "Machine learning models can overfit to training data if not regularized properly.",
    "The Eiffel Tower was completed in 1889 and stands 330 meters tall.",
    "Albert Einstein developed the theory of relativity in the early 20th century.",
    "The Amazon rainforest produces about 6% of the world's oxygen.",
    "Quantum computers use qubits that can exist in superposition of states.",
    "The Great Wall of China is over 21,000 kilometers long.",
    "DNA sequencing has become orders of magnitude cheaper over the past two decades.",
    "Renewable energy sources include solar, wind, hydro, and geothermal.",
    "The Mona Lisa was painted by Leonardo da Vinci between 1503 and 1519.",
    "Plate tectonics explains the movement of the Earth's continental plates.",
    "The human genome project was completed in 2003 after 13 years of work.",
    "Shakespeare's plays are typically categorized as comedies, tragedies, or histories.",
    "The Fibonacci sequence appears throughout nature, in sunflowers and pinecones.",
    "Marie Curie was the first person to win Nobel Prizes in two different sciences.",
    "The Internet was originally developed as ARPANET in the late 1960s.",
    "Gravity on Mars is about 38% of Earth's gravity.",
    "The Amazon River is the largest river by discharge volume in the world.",
    "Machine translation has improved dramatically with the advent of neural networks.",
    "The Milky Way galaxy contains an estimated 100 to 400 billion stars.",
    "Antibiotics work by killing or slowing the growth of bacteria.",
    "The Sahara Desert covers most of North Africa and is the largest hot desert.",
    "The Hubble Space Telescope has been operating since 1990.",
    "The Roman Empire reached its greatest extent in 117 AD under Trajan.",
    "Tides are caused by the gravitational pull of the Moon and Sun on Earth's oceans.",
    "The French Revolution began in 1789 with the storming of the Bastille.",
    "The Taj Mahal was built by Mughal emperor Shah Jahan in memory of his wife.",
    "Octopuses have three hearts and nine brains.",
    "The Andes is the longest mountain range in the world.",
    "Penguins are flightless birds that are highly adapted to life in water.",
    "The Panama Canal connects the Atlantic and Pacific Oceans.",
    "Bananas are berries, but strawberries are not.",
    "The human heart beats about 100,000 times per day.",
    "Light travels about 300,000 kilometers in one second.",
    "The Sahara was once a lush, green region with lakes and grasslands.",
    "An octopus can change color and texture to camouflage itself.",
    "The first successful powered flight was by the Wright brothers in 1903.",
    "Bananas contain a small amount of natural radiation due to their potassium content.",
    "Cows have best friends and can become stressed when separated from them.",
    "Honey never spoils and can last for thousands of years if stored properly.",
    "A group of flamingos is called a 'flamboyance'.",
    "The unicorn is the national animal of Scotland.",
    "Sea otters hold hands while sleeping so they don't drift apart.",
    "A jiffy is an actual unit of time: 1/100th of a second.",
    "Cleopatra lived closer in time to the moon landing than to the building of the Great Pyramid.",
    "Oxford University is older than the Aztec Empire.",
    "There are more ways to arrange a deck of cards than there are atoms on Earth.",
    "A bolt of lightning is about 5 times hotter than the surface of the sun.",
    "Cats can rotate their ears 180 degrees.",
    "The shortest war in history lasted 38 to 45 minutes.",
    "Sloths can hold their breath longer than dolphins can.",
    "Wombat poop is cube-shaped.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ICS quantize a HF causal LM on Colab T4")
    p.add_argument("--model", default="google/gemma-4-27b", help="HF model name or path")
    p.add_argument("--output", default="./gemma-4-27b-ics", help="output directory")
    p.add_argument("--max-calibration-samples", type=int, default=64)
    p.add_argument("--max-calibration-length", type=int, default=256)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--int4-fraction", type=float, default=0.5)
    p.add_argument("--int2-fraction", type=float, default=0.4)
    p.add_argument("--int1-fraction", type=float, default=0.1)
    p.add_argument(
        "--quant-method",
        choices=["block", "gptq", "per_row_int4"],
        default="per_row_int4",
        help="quantization backend; per_row_int4 is the current outlier-robust default",
    )
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--method", choices=["composite", "sinkhorn_hungarian"], default="composite")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-grad-ckpt", action="store_true",
                   help="disable gradient checkpointing (more VRAM, faster)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available; running on CPU. This will be very slow for 27B.")
        device = "cpu"
    else:
        device = "cuda"
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\n=== ICS quantize: {args.model} ===\n")

    # 1. Load model in 4-bit (NF4) to fit in T4 VRAM
    print("[load] importing transformers + bitsandbytes...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"[load] loading {args.model} in 4-bit...", flush=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # eager is safest across HF versions
    )
    print(f"[load] model loaded in {time.time() - t0:.1f}s", flush=True)

    # Enable gradient checkpointing for Fisher backward pass
    if not args.no_grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("[load] gradient checkpointing ON", flush=True)

    # BitsAndBytes 4-bit requires this for backward
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # 2. Build calibration set
    cal = DEFAULT_CALIBRATION[: args.max_calibration_samples]
    if len(cal) < args.max_calibration_samples:
        # repeat to hit the count
        cal = (cal * (args.max_calibration_samples // len(cal) + 1))[: args.max_calibration_samples]

    config = ICSConfig(
        block_size=args.block_size,
        int4_fraction=args.int4_fraction,
        int2_fraction=args.int2_fraction,
        int1_fraction=args.int1_fraction,
        quant_method=args.quant_method,
        alpha=args.alpha,
        beta=args.beta,
        method=args.method,
        calibration_texts=cal,
        max_calibration_length=args.max_calibration_length,
    )

    # 3. Run the pipeline
    t0 = time.time()
    result = quantize_model(model, tokenizer, config, device=device)
    print(f"\n[pipeline] finished in {time.time() - t0:.1f}s", flush=True)

    # 4. Save
    print(f"\n[save] writing to {args.output}...", flush=True)
    save_ics_model(
        result,
        args.output,
        tokenizer=tokenizer,
    )
    print(f"[save] done", flush=True)

    # 5. Compression report
    n_quantized = len(result.quant)
    total_dense_params = 0
    total_packed_bytes = 0
    for qt in result.quant.values():
        n = 1
        for s in qt.original_shape:
            n *= s
        total_dense_params += n
        total_packed_bytes += qt.qdata.numel()  # int8 storage
    total_dense_bytes = total_dense_params * 2  # BF16 baseline
    print(f"\n=== Compression report ===")
    print(f"Quantized layers:  {n_quantized}")
    print(f"Dense params:      {total_dense_params / 1e9:.2f} B")
    print(f"Dense size (BF16): {total_dense_bytes / 1e9:.2f} GB")
    print(f"Packed size:       {total_packed_bytes / 1e9:.2f} GB")
    print(f"Compression ratio: {total_dense_bytes / max(total_packed_bytes, 1):.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
