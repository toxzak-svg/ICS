# Isomorphic Channel Sorting (ICS)

Offline topological weight permutation + variable-bit block quantization for mobile NPU deployment.

Replaces runtime weight patching (FABQ-RC and friends) with a compile-time
sort that yields a single contiguous INT4/INT2/INT1 memory stream. Drop-in
on Apple CoreML, Qualcomm QNN, and Android NNAPI — no custom kernels.

## Theory (one paragraph)

For two sequential linear layers $W_A, W_B$ sharing a channel dim, the
identity $X (W_A P^T)(P W_B) = X W_A W_B$ holds for any permutation $P$.
The trick is to pick $P$ such that high-Fisher channels and weight
outliers cluster into the same dense blocks, so the per-block scaling
factor $\alpha$ stops crushing normal weights. Block quantization then
assigns INT4 to the dense blocks, INT2/INT1 to the tail.

The activation-Fisher vector $F[i] = E[(\partial L / \partial a_i)^2]$ is
shared between a layer's output channels and the next layer's input
channels (same activation tensor), so there is no real "two-sided"
conflict on the activation side. The remaining joint optimization is
on the weight-magnitude side: $W_A$'s column outliers vs $W_B$'s row
outliers. We use a 1-D composite sort by default (O(N log N)) and a
Sinkhorn-Knopp + Hungarian relaxation (O(N^3)) for the cleaner version.

## Pipeline

```
load model (4-bit NF4)  →  compute activation-Fisher  →  find joint permutation  →  apply  →  quantize per-block  →  save
```

1. **Load** the model in 4-bit (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`) so a 27B model fits in 15GB T4 VRAM.
2. **Fisher**: forward + backward on 64-256 short calibration strings. Captures $(\partial L / \partial a_i)^2$ per output channel of every `nn.Linear`.
3. **Permutation**: for each chain (attention block: q/k/v/o; MLP block: gate/up/down), find a single permutation that minimizes the joint quantization cost. Block-aware rebalance keeps high-Fisher channels out of block-edge padding.
4. **Apply**: bake the perm into the weights. For an MLP: `gate[rows]=P, up[rows]=P, down[cols]=P`. Element-wise SiLU and product commute with the perm. For non-GQA attention: `q[rows]=k[rows]=v[rows]=o[cols]=P`. GQA attention is skipped until the permutation is constrained to preserve Q/KV head groups.
5. **Quantize**: assign INT4 / INT2 / INT1 per block based on the block's total Fisher score.
6. **Save**: safetensors directory with `model.safetensors` (packed int8), `scales.safetensors`, `zeros.safetensors`, `bits.safetensors`, and `ics_meta.json` for reconstruction.

## Run on Colab T4

```python
# Colab cell 1: install
!git clone <this-repo>
%cd ICS
!pip install -q -r requirements.txt
```

```python
# Colab cell 2: quantize
!python scripts/colab_quantize_ics.py \
    --model google/gemma-4-27b \
    --output ./gemma-4-27b-ics \
    --max-calibration-samples 64 \
    --max-calibration-length 256
```

The script will:

- Load gemma-4-27b in NF4 (≈14GB on T4)
- Enable gradient checkpointing
- Compute Fisher on 64 short calibration strings
- Find + apply the permutation for every attention + MLP block
- Quantize with mixed INT4/INT2/INT1
- Save the result to `./gemma-4-27b-ics/`

Expected runtime on T4: ~30-60 min for 27B depending on `--max-calibration-samples`.

## Run locally (CPU, no model)

```bash
# Algorithm correctness
python scripts/test_correctness.py

# Pipeline smoke (uses a tiny fake transformer, no model download)
python scripts/test_pipeline_smoke.py
```

## Run Qwen3.5-2B on Colab

The active Qwen target for the bridge pipeline is **Qwen/Qwen3.5-2B**. Use the
Colab runner for the real quantization job; it downloads the upstream model,
runs the ICS+GPTQ path, saves the artifact, and can upload staged outputs to
Hugging Face.

```bash
python scripts/colab_quantize_qwen35.py \
    --model Qwen/Qwen3.5-2B \
    --output ./qwen3.5-2b-ics \
    --max-calibration-samples 64 \
    --max-calibration-length 256
```

For the managed Colab bridge workflow, use:

```bash
python colab_bridge/pipeline_qwen35.py
```

That bridge script targets `toxzak/Qwen3.5-2B-ICS-INT4` for the quantized model
artifact and `toxzak/ics-quantization-artifacts` for snapshots, logs, and
benchmark outputs.

## Legacy Qwen3-0.6B local smoke

The 0.6B runner is a CPU-friendly smoke/debug path only. It exercises the real
Hugging Face model path without requiring a GPU and defaults to one calibration
sample and one selected chain so CPU-only machines can validate mechanics
without attempting a full-model quantization job.

First make sure the tokenizer files are cached with the model snapshot:

```bash
hf download Qwen/Qwen3-0.6B \
    --include tokenizer.json \
    --include tokenizer_config.json \
    --include vocab.json \
    --include merges.txt \
    --include generation_config.json
```

Then run the one-chain smoke pipeline:

```bash
python scripts/run_qwen3_06b_pipeline.py \
    --local-files-only \
    --offline \
    --max-calibration-samples 1 \
    --max-calibration-length 8 \
    --max-chains 1 \
    --no-save
```

Expected local smoke result:

- Qwen3-0.6B loads from the cached snapshot.
- The real tokenizer returns non-empty token IDs.
- 56 chains are discovered, 1 chain is selected.
- Fisher, permutation, apply, and quantization complete.
- The smoke run produces 1 permutation and 2 quantized layers.

Use `--output ./qwen3-0.6b-ics-smoke` without `--no-save` to write the
safetensors export. The runner still has `--synthetic-tokenizer` for cache-only
debugging, but use the real tokenizer for any meaningful calibration run.

## What this codebase does

| file | purpose |
| --- | --- |
| `ics/fisher.py` | per-channel activation-Fisher via forward+backward hooks |
| `ics/permutation.py` | composite-score + Sinkhorn-Hungarian permutations; chain + fan-out applications |
| `ics/quantize.py` | variable-bit block quantize / dequantize (INT4 / INT2 / INT1) |
| `ics/pipeline.py` | chain discovery, joint perm search, application, quant orchestration |
| `ics/export.py` | safetensors save/load with perm metadata |
| `scripts/colab_quantize_ics.py` | end-to-end CLI: load → fisher → perm → quant → save |
| `scripts/colab_quantize_qwen35.py` | Qwen3.5-2B Colab quantization entrypoint |
| `colab_bridge/pipeline_qwen35.py` | Qwen3.5-2B bridge orchestration with staged HF upload and PPL eval |
| `scripts/run_qwen3_06b_pipeline.py` | legacy CPU-friendly Qwen3-0.6B smoke runner with offline/cache support |
| `scripts/test_correctness.py` | 10 unit tests on synthetic data, including chain identity at FP32 noise |
| `scripts/test_pipeline_smoke.py` | end-to-end pipeline on a 64-dim fake transformer plus Fisher regressions |
| `scripts/test_gqa.py` | Qwen-style grouped-query attention chain discovery and apply smoke tests |
| `scripts/test_load_qwen.py` | quick Qwen3-0.6B load and shape probe |

## Tests

```
$ python scripts/test_correctness.py
ALL 10 TESTS PASSED
  chain forward-pass max abs diff: 5.245e-06
  MLP fan-out-chain forward diff: 7.629e-06
  attention fan-out forward diff: 1.431e-06
  full pipeline: forward pass max diff: 1.526e-05

$ python scripts/test_pipeline_smoke.py
ALL 7 TESTS PASSED
  end-to-end forward diff: 3.725e-09

$ python scripts/test_gqa.py
ALL 2 TESTS PASSED

$ python scripts/run_qwen3_06b_pipeline.py --local-files-only --offline --max-calibration-samples 1 --max-calibration-length 8 --max-chains 1 --no-save
[pipeline] permutations: 1
[pipeline] quantized layers: 2
```

The forward-pass residuals are FP32 roundoff — the math
$X(W_A P^T)(P W_B) = X W_A W_B$ is exact.

## Quality status

`scripts/benchmark_perplexity.py` still measures perplexity, but the report now
uses baseline-relative **quality** as the decision signal. The dense baseline is
quality `1.0`; each candidate gets `baseline_ppl / candidate_ppl`, plus a
`pass`/`fail` status against `--min-quality-score` (default `0.90`).

The harness compares:

- Hugging Face dense baseline (`--dtype fp16`, `bf16`, or `fp32`)
- ICS-dequantized weights loaded back into the HF model
- llama.cpp `Q4_K_M` GGUF via `llama-perplexity`

Tiny local smoke command:

```bash
python scripts/benchmark_perplexity.py \
    --local-files-only \
    --offline \
    --dtype fp16 \
    --quant-method per_row_int4 \
    --max-eval-tokens 128 \
    --max-calibration-samples 1 \
    --max-calibration-length 8 \
    --max-chains 1 \
    --q4-gguf <qwen3-0.6b-q4_k_m.gguf> \
    --llama-ctx 16 \
    --min-quality-score 0.90 \
    --output-json perplexity_results_qwen3_06b.json
```

Use `quality_summary` in the output JSON as the gate from now on. The raw
perplexity values are diagnostic evidence, not the final acceptance label.
When rebuilding `--ics-output`, `--quant-method per_row_int4` is the current
outlier-robust path; `block` is the older mixed INT4/INT2/INT1 path, and `gptq`
is the Hessian-based diagnostic path.

For a sequential method sweep and publishable package, use:

```bash
python scripts/benchmark_matrix.py \
    --model Qwen/Qwen3-0.6B \
    --local-files-only \
    --offline \
    --dtype fp16 \
    --methods block,gptq,per_row_int4 \
    --output-dir benchmarks/qwen3_06b_harness \
    --scratch-dir .pytest_cache/ics_harness
```

The matrix runner writes `manifest.json`, `RESULTS.md`, and
`package_manifest.json` under `benchmarks/qwen3_06b_harness`, while large
method artifacts and zip packages remain under `.pytest_cache/ics_harness`.

Current Qwen3-0.6B tiny smoke result is recorded in
`benchmarks/qwen3_06b_perplexity_smoke.json`:

| model | perplexity | quality | status | tokens | note |
| --- | ---: | ---: | --- | ---: | --- |
| fp16 | 92.9362 | 1.0000 | baseline | 49 | HF dense baseline |
| ics_dequantized | 4702612.5630 | 0.0000 | fail | 49 | one-chain smoke artifact; not full-model ICS |
| q4_k_m | 216.8099 | 0.4287 | fail | 16 | llama.cpp Q4_K_M GGUF |

These are not Qwen3.5-2B report numbers and not reportable WikiText-2 numbers.
They use the built-in tiny eval text so the harness can complete on CPU. For
meaningful reporting, pass a fixed dataset text file with `--eval-text-file`,
raise `--max-eval-tokens`, and run the full Qwen3.5-2B ICS artifact instead of
the one-chain 0.6B smoke artifact. The acceptance question should be "does the
candidate preserve enough baseline-relative quality?", not "is the raw PPL
finite?".

See `docs/PRIOR_ART.md` for the current prior-art map and novelty caveats.

## Caveats / honest gaps

- **Activation-Fisher requires gradients.** 4-bit-loaded models need BitsAndBytes ≥ 0.43 and `model.enable_input_require_grads()`.
- **Qwen GQA attention is skipped.** Grouped-query attention is not invariant under arbitrary channel permutations because Q heads and repeated KV heads have fixed grouping. The current chain discovery keeps MLP chains for Qwen-style models and skips GQA attention until a head-group-preserving permutation is implemented.
- **CPU Qwen smoke is intentionally bounded.** The local runner defaults to one chain and a low-memory Fisher objective. Full-model calibration should run on a GPU with a real calibration set.
- **27B on T4 is tight.** If you OOM during Fisher backward, drop `--max-calibration-length` from 256 to 128 first.
- **One-shot calibration.** The Fisher is computed once on a small text mix. For domain-specific deployments, swap `DEFAULT_CALIBRATION` in `scripts/colab_quantize_ics.py` for in-domain text.
- **No kernel-side custom op.** The export is the dense (dequantized) layout plus the metadata needed to reconstruct the packed layout at deploy time. Wiring this to a custom NPU kernel is out of scope for this repo — that's a separate piece of work tied to CoreML/QNN/NNAPI.
- **Supersedes the buggy prototype.** The `ics_prototype.py` script in the project root permutes Q/K by row-magnitude with no inverse-perm on O and no block awareness. The current `ics/` package supersedes it.

## What this is not

- Not a fine-tuning framework. The permutation is locked post-calibration. Re-training the model after ICS permute will diverge from the original training trajectory — this is the limit of the offline-permutation trick.
- Not a general integer-arithmetic library. The packed int8 layout is a storage convention; runtime dequantization for verification is provided but the deploy-time NPU kernel is a separate problem.
- Not a weight-only PTQ that competes on perplexity with GPTQ/AWQ. ICS is a *topology* change, not a better quantizer. Combined with INT4/INT2/INT1, the accuracy hit will be in the same ballpark as any mixed-precision block-quant scheme (~0.3-1.0 PPL on wikitext for 27B), but the gain is the elimination of FABQ-RC's runtime sparse-gather.

## License

MIT.
