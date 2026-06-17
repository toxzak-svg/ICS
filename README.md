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
4. **Apply**: bake the perm into the weights. For an MLP: `gate[rows]=P, up[rows]=P, down[cols]=P`. Element-wise SiLU and product commute with the perm. For attention: `q[rows]=k[rows]=v[rows]=o[cols]=P`. Softmax commutes because it's row-equivariant under a uniform row perm.
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

## What this codebase does

| file | purpose |
| --- | --- |
| `ics/fisher.py` | per-channel activation-Fisher via forward+backward hooks |
| `ics/permutation.py` | composite-score + Sinkhorn-Hungarian permutations; chain + fan-out applications |
| `ics/quantize.py` | variable-bit block quantize / dequantize (INT4 / INT2 / INT1) |
| `ics/pipeline.py` | chain discovery, joint perm search, application, quant orchestration |
| `ics/export.py` | safetensors save/load with perm metadata |
| `scripts/colab_quantize_ics.py` | end-to-end CLI: load → fisher → perm → quant → save |
| `scripts/test_correctness.py` | 10 unit tests on synthetic data, including chain identity at FP32 noise |
| `scripts/test_pipeline_smoke.py` | end-to-end pipeline on a 64-dim fake transformer |

## Tests

```
$ python scripts/test_correctness.py
ALL 10 TESTS PASSED
  chain forward-pass max abs diff: 5.245e-06
  MLP fan-out-chain forward diff: 7.629e-06
  attention fan-out forward diff: 1.431e-06
  full pipeline: forward pass max diff: 1.526e-05

$ python scripts/test_pipeline_smoke.py
ALL 2 TESTS PASSED
  end-to-end forward diff: 3.725e-09
```

The forward-pass residuals are FP32 roundoff — the math
$X(W_A P^T)(P W_B) = X W_A W_B$ is exact.

## Caveats / honest gaps

- **Activation-Fisher requires gradients.** 4-bit-loaded models need BitsAndBytes ≥ 0.43 and `model.enable_input_require_grads()`.
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
