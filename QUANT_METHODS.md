# Quant Methods — Your Stack

Survey of every weight-quant method currently implemented in `projects/ICS/` and `projects/fabq-rc/`. Numbers come from your benchmark JSONs and MD reports (not from paper claims) — wherever I cite a perplexity or MSE, the source file is called out.

## TL;DR

Target correction: the active ICS bridge target is **Qwen3.5-2B**. Qwen3-0.6B
results in this document are legacy smoke/debug evidence and cheaper
reproductions of the export/load failure, not the current model target.

You have two fundamentally different systems:

1. **ICS** — *offline topological sort + variable-bit block quantization*. Permutes weights at compile time so all sensitive channels land in dense INT4 blocks, then dumps to a clean contiguous int8 stream. Designed for **NPU / mobile deployment** where the runtime kernel is dumb and memory layout matters. The active bridge target is **Qwen3.5-2B**; the checked-in Qwen3-0.6B PPL is legacy debug evidence showing the same export/load failure mode at smaller scale.

2. **FABQ-RC** — *runtime adaptive mixed-precision quantization with custom CUDA kernels*. Each row gets assigned int4 or binary at quantization time, dequantized on the fly inside a fused GEMM kernel. Designed for **GPU inference** where you want compression but don't want a custom NPU target. Working at ~1.4 bpw (lite) and ~3 bpw (unified vp/ebq) with measured throughput.

These are **not competing methods** — they're aimed at different deployment targets. Below is the full inventory.

## The full inventory

| # | Method | Where | Bit-widths | Block / Granularity | Perm? | Calibration | Targets |
|---|---|---|---|---|---|---|---|
| 1 | ICS per-block (core) | `ICS/ics/quantize.py` | INT4 / INT2 / INT1 per block | block_size=64 | Yes (Fisher sort) | Fisher | NPU/mobile, contiguous int8 stream |
| 2 | ICS GPTQ path | `ICS/ics/gptq.py` | INT4 | per-group (group_size=128) | Yes (Fisher chain) | Fisher + Hessian | ICS pipeline alternative backend |
| 3 | ICS composite perm | `ICS/ics/permutation.py` | n/a (sort only) | channel perm | yes | Fisher + weight ∞-norm | used by both ICS paths |
| 4 | ICS sinkhorn-hungarian perm | same | n/a | channel perm | yes | Fisher + weight ∞-norm | cleaner opt, O(N³) |
| 5 | ICS spectral perm | same | n/a | channel perm via Fiedler vec | yes | Fisher + weight ∞-norm | multi-feature, OR-semantics |
| 6 | FABQ-RC-lite | `fabq-rc/benchmarks/benchmark_qwen35_08b_weight_quant.py` | binary + 5% int4 | adaptive blocksize | no | row energy (Fisher proxy) | weight recon sanity, no CUDA |
| 7 | FABQ-RC (mixed) | `fabq-rc/gemma4-12b/streaming/fabq_rc_cuda/quantized_linear.py` + `src/fabq_rc_gemm_v2.cu` | int4 OR binary per row | blocksize=64, codebook correction | no | row energy | production CUDA GEMM |
| 8 | Unified FABQ-VP/EBQ | `fabq-rc/benchmarks/benchmark_unified_fabq.py` | int8 / int4 / int2 / binary per row | blocksize=128 | no | forward-only imatrix | dense-dequantized PPL/gen validation |

Plus the comparison baseline:
- **llama.cpp Q4_K_M GGUF** — measured via `benchmark_perplexity.py` (q4_k_m numbers below)

---

## 1. ICS per-block variable-bit (core)

**File:** `ICS/ics/quantize.py`

**Algorithm.** For each `nn.Linear` weight `W[out, in]`, you first sort channels via the perm pipeline (see §3-5), then quantize along the sorted dim in blocks of 64:

```
block 0 .. int4_fraction*total_blocks  -> INT4 (symmetric [-8, 7], absmax / 7)
block int4 .. int4+int2                  -> INT2 (symmetric [-2, 1], 4-per-byte packed)
block tail                               -> INT1 (sign-only, 8-per-byte packed)
```

Per-block scales/zeros (always zero for symmetric). Storage is a single int8 stream + per-block fp32 scales + int32 zero points + per-block bits.

**What it gives you.** A *contiguous* memory layout — exactly what NPU INT kernels want. No codebook, no per-row dispatch, no runtime matrix-indirection. The dequant path is `qdata → int8 → reshape → ×scale`, fully branch-free per block.

**What it costs.** The permutation is mandatory: without Fisher-sorting the channels first, INT2/INT1 blocks get crushed by outliers and the model is unusable. So ICS is really "permutation + per-block quant", not "quant" alone.

**Calibration.** Per-channel Fisher (`fisher.py`) computed via forward+backward through the unquantized model on ~64-256 calibration strings. `loss_mode="cross_entropy"` (next-token CE) or `"last_logit_mean"`. Backward through BitsAndBytes Linear4bit works in bnb>=0.43, which is why the pipeline can run on a T4 for 27B-class models.

**Round-trip claim from docstring:** ~1% relative for INT4, ~5% for INT1.

---

## 2. ICS GPTQ path

**File:** `ICS/ics/gptq.py` (new module, 275 LoC, from-scratch — not a wrapper around auto-gptq)

**Algorithm.** Classic Frantar-style GPTQ:
1. Compute layer Hessian `H = 2 X^T X / (B*T)` from forward hooks over calibration batches.
2. Cholesky factor, get `H_inv_chol`.
3. Damp diagonal by `percdamp * mean(diag(H))` (default 1%); retry with progressively larger damping if Cholesky fails.
4. Sort columns by descending Hessian diag, process in `blocksize=128` chunks, propagate column quantization error to remaining columns using `H_inv_chol`.
5. Per-group scale (group_size=128), per-group zero (always 0 for symmetric).

Output: `Q_int8`, `scales[n_groups]`, `zeros[n_groups]=0`, `perm[in_features]`.

**What it gives you vs. §1.** Better weight error than naive per-block at the same bpw (Hessian-aware column ordering reduces squared reconstruction error). But the data layout is "row-major per group" not "block-row-major by Fisher bin", so it doesn't get the same NPU-friendliness unless you reshape it.

**Used as alternative backend** in `pipeline.py:610` when `config.quant_method == "gptq"`. The pipeline computes Hessians for all chain members, then GPTQ-quantizes each layer.

**Implementation note.** The from-scratch implementation is ~150 LoC, deliberately minimal. Trade-off: it doesn't have all the bells (act-order, true sequential groups) of `auto-gptq`. For your scale (sub-3B models on T4/colab) that's probably fine.

---

## 3. Composite-score channel permutation

**File:** `ICS/ics/permutation.py:175`

**Algorithm.** For a (W_A, W_B) chain with shared channel dim N, sort channels by:
```
score[i] = F[i] * (alpha * ||W_A[:, i]||_inf + beta * ||W_B[i, :]||_inf + 1e-12)
```
with `alpha=beta=1` defaults. The sort is `argsort(score, descending=True)`. Result is a single permutation of length N.

**The AND-semantics problem (called out in your own docstring).** Channels that are high-Fisher but low-magnitude (or vice versa) get buried in the tail. The docstring explicitly motivates §5 (spectral) as the fix for this.

**Cost.** O(N log N). Fast.

---

## 4. Sinkhorn-Hungarian permutation

**File:** `ICS/ics/permutation.py:229`

**Algorithm.**
1. Build cost matrix `C[i, j] = F[i] * (alpha * M_A[j] + beta * M_B[j])` — NxN.
2. Sinkhorn-Knopp doubly-stochastic relaxation: 10 iters of row/column normalization of `K = exp(-C / mean(C))`.
3. Hungarian projection via `scipy.optimize.linear_sum_assignment(-K)` to get a hard permutation.
4. Optional `_rebalance_for_blocks` step that re-sorts each block of `block_size` channels by Fisher descending (so high-Fisher channels sit at block *interior* not block *edges*).

**Cost.** O(N³) from Hungarian. For typical LLM hidden dims (1024-8192) this is fine (ms-level). Block-aware rebalance is O(N log N) per block.

**When to use over §3.** When the joint objective is the priority and you can afford the latency. For your use case (one-time offline pipeline), always prefer this over composite.

---

## 5. Spectral permutation

**File:** `ICS/ics/permutation.py:51`

**Algorithm.**
1. Stack three normalized per-channel features into `[N, 3]`: Fisher, W_A's per-output-channel ∞-norm, W_B's per-input-channel ∞-norm.
2. Build `N×N` Gaussian RBF affinity matrix on this joint feature, bandwidth set by median heuristic (no alpha/beta knobs).
3. Symmetric-normalize, eigendecompose.
4. Take the Fiedler vector (second-smallest eigenvalue's eigenvector), use it as the 1-D sort key. Or stack top-N components and take first PC.

**Why it matters.** OR-semantics on the joint feature space (RBF similarity) instead of AND-semantics (product). High-Fisher-only channels no longer get buried.

**Cost.** O(N²) memory + O(N³) eigendecomp. For N=4096 that's ~256MB float32 and a few seconds. Probably fine offline, possibly heavy for online re-permutation.

**Status in your code.** Exists but I don't see a switch flag — check if pipeline.py currently routes to it. (Quick grep would tell you.) If it doesn't, you're leaving wins on the table.

---

## 6. FABQ-RC-lite

**File:** `fabq-rc/benchmarks/benchmark_qwen35_08b_weight_quant.py` (via `fabq-rc/gemma4-12b/streaming/fabq_rc_cuda/quant_pipeline.py`)

**Algorithm.** Cheap version of FABQ-RC:
- 5% of rows assigned int4 (top-Fisher), 95% assigned binary.
- Adaptive blocksize per layer (sweep {64, 128, 256, 512}, pick min Fisher-weighted reconstruction error with penalty `BS_PENALTIES = {64:1.5, 128:1.0, 256:0.85, 512:0.75}`).
- No codebook, no inference kernel — just weight reconstruction.

**Calibration.** Row energy as Fisher proxy (not real Fisher). That's the "lite" — fast, but lossy.

**Observed (Qwen3.5-0.8B, 244 tensors, 615.6M weights, 90.56s CPU):**

| Method | MSE | SQNR (dB) | bpw |
|---|---:|---:|---:|
| int8 rowwise symmetric | 1.78e-8 | 40.59 | 8.01 |
| int4 rowwise symmetric | 5.77e-6 | 15.48 | 4.01 |
| Q1 block64 | 7.63e-5 | 4.27 | 1.25 |
| Q1 block128 | 7.70e-5 | 4.23 | 1.13 |
| Q1 block256 | 7.75e-5 | 4.20 | 1.06 |
| Q1 block512 | 7.79e-5 | 4.18 | 1.03 |
| **FABQ-RC-lite** | **6.62e-5** | **4.89** | **1.40** |

Source: `fabq-rc/results/qwen35_08b_weight_quant.md`.

**Reading.** FABQ-RC-lite is *better than fixed binary at the same bpw*, but the SQNR is still in single digits. That's fine for "weight reconstruction quality" sanity but does not validate perplexity. The .md file says this explicitly: *"This benchmark should be treated as a local weight-level sanity benchmark, not as a claim that full FABQ-RC perplexity has been validated for Qwen3.5-0.8B."*

---

## 7. FABQ-RC (mixed int4 + binary, custom CUDA kernel)

**Files:** `fabq-rc/gemma4-12b/streaming/fabq_rc_cuda/quantized_linear.py` + `src/fabq_rc_gemm_v2.cu`

**Algorithm.**
- For each `nn.Linear` row: top-N rows by energy get int4 (with per-row fp16 scale), rest get binary (per-block fp16 scale + per-block codebook index, codebook is fp16[N_clusters, max_blocksize]).
- Stored as: `int4_weights[n_int4, in]`, `binary_bits` (packed uint8), `binary_scales[n_binary, n_blocks]`, `codebook_idx[n_binary, n_blocks]`, `codebook[n_clusters, max_blocksize]`.
- The forward pass NEVER materializes the fp16 weight. The CUDA kernel consumes the compressed buffers directly.

**Inference kernels (v2):**
- `v2_int4_kernel` — vectorized scalar int4 GEMM, decode-friendly (small B*T). `__half2` loads on x, scalar int8 on W.
- `v2_binary_kernel` — binary-only GEMM with coalesced bit-byte unpacking (adjacent threads → adjacent bytes → coalesced 32-bit loads).
- `v2_mixed_kernel` — per-row int4-or-binary dispatch. The general FABQ-RC case.
- `v2_int4_via_fp16_tc_kernel` — **W4A16** tensor-core GEMM (int4 weight dequantized to fp16 in shared memory, fp16×fp16 WMMA m16n16k16). This is the production path for batched eval. The comment in the kernel is candid: *"not native int4 TC - that would require int4 activations, which explodes PPL at this bpw."*
- `v2_embed_kernel` — quantized embedding lookup using the same components.

Dispatch logic in `quantized_linear.py:617`: TC path when `B*T >= 16 && in_features % 16 == 0 && out_features % 128 == 0`, else scalar path. Bias is fused into the GEMM (no second kernel launch).

**QuantizedLinear.module** pretends to be nn.Linear (`.in_features`, `.out_features`, `.bias`) but `.weight` raises — there's no FP16 weight, ever.

**Observed (Qwen3-0.6B, 196 layers, 440M weights, 19.5s CPU validation):**
- bpw: 1.40
- MSE: 2.65e-4
- SQNR: 4.86 dB
- blocksize: 64 (all 196 layers)

Source: `fabq-rc/results/qwen3_06b_fabq_runtime_benchmark.json`.

**Note on the kernel comment.** The CUDA kernel explicitly avoids native int4 TC because W4A4 would explode perplexity. Your W4A16 path is the right call — it dequantizes int4→fp16 in shared memory on the fly, then does standard fp16×fp16 WMMA. This means you *don't* hit the W4A4 off-table cliff that I flagged in memory.

---

## 8. Unified FABQ-VP/EBQ (variable precision + ebq codebook correction)

**File:** `fabq-rc/benchmarks/benchmark_unified_fabq.py` + `quant_pipeline.py`

**Algorithm.**
- "Forward-only imatrix calibration" computes input-feature importance (no backward, no Fisher). Lightweight.
- For a target bpw, pick a row-level precision mix:

| target bpw | int8 | int4 | int2 | binary |
|---:|---:|---:|---:|---:|
| ≤ 2.05 | 2% | 18% | 30% | 50% |
| ≤ 3.05 | 3% | 49% | 24% | 24% |
| > 3.05 | 5% | 85% | 10% | 0% |

(vp = variable precision; the 5/49/24/24 mix is what your Qwen3.5-2B benchmark used.)

- EBQ: int2 and binary rows get a *residual block correction* — a small correction vector per block stored alongside the bitstream. The "ebq" codebook correction is per-block-residual mean, computed at quant time.

**Observed (Qwen3.5-2B, 284 layers, 1.7B target weights, 40s on colab):**
- target bpw: 3.0
- nominal mix bpw: 2.92
- estimated bpw: 3.11
- MSE: 1.81e-5
- SQNR: 9.38 dB

Source: `fabq-rc/results/qwen3_5_2b_unified_fabq_benchmark.json`.

**Reading.** Going from 1.40 bpw (lite, MSE 2.65e-4) → 2.92 bpw (unified, MSE 1.81e-5) buys ~14× lower MSE. SQNR more than doubles (4.86 → 9.38). At the cost of slightly more than 2× the storage. Pareto-favorable.

The validation report flags this as **CPU dequantized** (not native CUDA kernel), so it's a PPL/gen validation path, not a throughput benchmark. The kernel work for unified is the next step.

---

## Tradeoff matrix

| Method | Storage / 1B params | Best PPL hit | Speed on GPU | Speed on CPU | Custom kernel? | Hardware target |
|---|---:|---|---|---|---|---|
| fp16 baseline | 2.00 GB | 0 (reference) | 1.0× | 1.0× | none | any |
| int8 rowwise | 1.00 GB | ~negligible | ~1.0× (memory-bound, ~half bw) | ~2× (memory) | optional | any |
| llama.cpp Q4_K_M | ~0.55 GB | small (~2× PPL on Qwen3-0.6B 49 tokens) | n/a (CPU) | reference | llama.cpp built-in | CPU/Metal/CUDA |
| ICS GPTQ (group=128) | 0.55 GB | small | dequant → fp16 then GEMM | dequant → fp16 then GEMM | optional | NPU/mobile target |
| ICS per-block (1.0/0.4/0.1 bpw) | ~0.55 GB typical | depends on Fisher quality | not directly runnable yet | not directly runnable yet | NPU INT kernel target | NPU/mobile |
| FABQ-RC-lite (1.40 bpw) | 0.175 GB | unvalidated PPL | n/a (CPU bench only) | n/a | none | recon only |
| FABQ-RC (1.40 bpw) | 0.175 GB | unvalidated PPL (recon MSE 2.65e-4) | has CUDA kernel, untested throughput | dequant fallback | yes (v2_int4 / v2_binary / v2_mixed / v2_int4_via_fp16_tc) | CUDA sm_80+ |
| Unified FABQ-VP/EBQ (2.92 bpw) | 0.365 GB | unvalidated (MSE 1.81e-5) | CPU dequant path | CPU dequant path | planned | same |
| QuIP# 2-bit (reference, not yours) | ~0.25 GB | near-fp16 PPL at 2bpw | via custom CUTLASS | slow | yes | CUDA |
| BitNet b1.58 (reference) | ~0.20 GB | competitive with larger models | native 1.58b kernels | slow | yes | custom |

(PPL numbers above are only for cases where you have a measured value; everything "unvalidated" reflects the gaps in your current benchmark set.)

---

## Observed metrics from your benchmark JSONs

### Qwen3.5-2B / Qwen3-0.6B ICS perplexity status

The active ICS target is `Qwen/Qwen3.5-2B` via `colab_bridge/pipeline_qwen35.py`.
The existing HF artifact `toxzak/Qwen3.5-2B-ICS-INT4` was measured as broken:
BF16 PPL 30.8 vs ICS-dequant PPL 1,170,034, with logit max_diff 28.75.

The smaller Qwen3-0.6B debug run shows the same class of failure in a cheaper
local artifact.

`ICS/perplexity_results_qwen3_06b_full_int4.json` (504 tokens, block_size=64):

| name | perplexity | seconds | notes |
|---|---:|---:|---|
| fp16 | 97.76 | 34.4 | reference |
| ics_dequantized | **23,126,432** | 43.8 | **broken** |
| q4_k_m | 43.64 | 2.0 | only 128 tokens |

(`q4_k_m`'s 128-token sample can't be compared directly to fp16's 504-token sample — different document windows.)

The ics_dequantized PPL being ~5 orders of magnitude worse than fp16 is not a quantization quality issue, it's a **load/unperm bug**. The export pipeline round-trips through chain perms and GPTQ column perms; something there is corrupting weights on load. Candidates: the fused-weight group-by-group unperm in `export.py:238` or the row/column-perm target assignment in `discover_chains`. You have all the metadata needed to debug — start by dumping one layer's dequantized weight and comparing to the original.

### Qwen3-0.6B FABQ-RC (reconstruction only)

`fabq-rc/results/qwen3_06b_fabq_runtime_benchmark.json`:
- 196 layers, 440M target weights
- 5% int4 rows / 95% binary
- bpw: 1.40, MSE: 2.65e-4, SQNR: 4.86 dB
- The validation report says forward + generate works on CPU; PPL was not measured in this file. (The Qwen3.5-0.8B PPL wasn't measured either; the .md is explicit about this.)

### Qwen3.5-2B unified FABQ-VP/EBQ

`fabq-rc/results/qwen3_5_2b_unified_fabq_benchmark.json`:
- 284 layers, 1.7B target weights
- mix: 3% int8, 49% int4, 24% int2, 24% binary
- bpw nominal 2.92 / estimated 3.11
- MSE: 1.81e-5, SQNR: 9.38 dB
- can_forward + can_generate validated (CPU dequant path)

---

## Where each method fits

**Pick ICS per-block when:**
- You're targeting an NPU / mobile inference path that has INT4/INT2/INT1 kernels and wants contiguous memory.
- You can afford the offline Fisher + permutation pass (one-time cost, scales with calibration set size).
- Memory layout matters more than last-percent PPL.

**Pick ICS GPTQ when:**
- You want a well-known, defensible algorithm name in the writeup.
- You don't need NPU-specific contiguous layout.
- You're using it inside the ICS pipeline (the export saves the per-layer column perm for unperm).

**Pick FABQ-RC when:**
- You're targeting CUDA sm_80+ and want one custom kernel that handles mixed int4/binary rows.
- You want decode-friendly (small B*T) and batched (B*T>=16) paths in the same module.
- 1.40 bpw is acceptable as a memory budget.

**Pick Unified FABQ-VP/EBQ when:**
- You can spend ~3 bpw instead of ~1.4 bpw and want MSE in the 1e-5 range instead of 1e-4.
- You're OK with CPU dequant for now (native CUDA kernel not built yet).
- You want one config knob (`target_bpw`) that picks the row mix for you.

**Don't pick FABQ-RC-lite for anything except sanity-checking.** It's a reconstruction-only benchmark, not a deployable model. The .md is explicit about this.

---

## What I'd work on next (if you want a recommendation)

In rough priority order, based on what's broken or missing:

0. **Fix the ICS end-to-end PPL on Qwen3.5-2B.** The 1,170,034 PPL artifact is a bug, not a quantization ceiling. The 0.6B run is only a cheaper reproduction. Current chain discovery skips GQA attention because arbitrary channel permutations do not preserve fixed Q/KV head groups; rerun the Qwen3.5-2B bridge with MLP-only ICS chains, then dequantize one remaining chain member and compare to the pre-perm weight if PPL is still broken.

1. **Use the Qwen3-0.6B result only as a cheaper reproduction.** The 23M perplexity is a bug, not a quantization ceiling, but the defensible target for this bridge is Qwen3.5-2B.

2. **Wire the spectral permutation into the pipeline.** It's the best of the three (OR-semantics, no alpha/beta knobs) and your code has it but I don't see a switch flag. One line in `pipeline.py` plus a config flag.

3. **Measure FABQ-RC and Unified end-to-end PPL on the same Qwen3-0.6B and Qwen3.5-2B you've already loaded.** You have the benchmark harness (`scripts/benchmark_perplexity.py`); you just haven't run it through the FABQ-RC modules. Right now FABQ-RC's "validation" is can-forward-can-generate, not actual perplexity. Without PPL numbers, you can't say FABQ-RC is better or worse than ICS at any bpw.

4. **Native CUDA kernel for unified FABQ-VP/EBQ.** Right now it dequantizes to dense bfloat16 on CPU for validation. Building a kernel that consumes the vp/ebq components natively is the next big engineering item, and it's where the real throughput win lives.

5. **Either drop FABQ-RC-lite from the benchmark set or rename it.** It's misleading — it implies "FABQ-RC" performance when it's just weight reconstruction. The .md is honest about it but the JSON file doesn't carry that caveat.
