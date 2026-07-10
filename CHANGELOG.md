
## 2026-07-10

Daily sync.


## 2026-07-09

Daily sync — local commits pulled/pushed if any; CHANGELOG refreshed.

## 2026-07-08

Daily sync.


## 2026-07-07

Daily sync.


## 2026-07-06

Daily sync.


## 2026-07-05

Daily sync.


## 2026-07-04

Daily sync.


## 2026-07-03

Daily sync.


## 2026-07-02

Daily sync.


## 2026-06-30

Daily sync.


## 2026-06-29

Daily sync.


## 2026-06-28

Daily sync.


## 2026-06-27

- **GQA-aware sub-perm in pipeline**: `discover_chains` now detects GQA structure (k/v with smaller output dim than q, where `shared % k_dim == 0`) and includes them in the chain with a derived sub-perm. New `_derive_gqa_sub_perm` constructs a strict permutation of length `k_dim` from the main chain perm via collision-resolved nearest-unused search. **Pre-fix code excluded k/v entirely** — root cause of catastrophic PPL (~1M). Pre-fix behavior: `softmax(q_permuted · k_original · v_original)` had mismatched layouts because q heads were reordered across KV group boundaries without reordering the KV heads themselves.
- **`PermutationResult.gqa_sub_perm`**: new field, set by `quantize_model` after the main perm is found. Length `k_dim` for GQA chains, `None` otherwise.
- **`ics/export.py`**: dequant reads `member_perms` map for per-member perm dispatch — GQA members get the sub-perm, all others get the main perm. Falls back gracefully for older artifacts without `member_perms` or `gqa_sub_perm_members` (uses main perm for everything, preserving pre-GQA artifact compatibility).
- **`scripts/test_gqa.py`**: 4 tests (chain discovery with k/v, sub-perm strict-perm invariant across identity/swap/adversarial/real perms, apply_chain with sub-perm keeps forward finite + bounded, end-to-end strict perm after composite-score perm) all green. Full local suite: 24/24 tests pass.
- **Fisher normalization follows GQA sub-perm**: for GQA members the Fisher vector is reordered by `gqa_sub` before bit-width assignment, so the bit allocation matches the permuted weight rows.

﻿## 2026-06-25

- **`ics/pipeline.py`** (+221/-X): pipeline expansion (new stages, expanded surface).
- **`ics/export.py`** (+86/-X): export pipeline additions (likely row/col quant metadata handling).
- **`ics/permutation.py`** (+6/-X): small refactor.
- New module **`ics/gptq.py`** - GPTQ quantization support.
- New directories **`colab_bridge/`**, **`colab_MiniMax/`** - Colab integration assets (likely for remote quant runs).

## 2026-06-26

- Corrected the public docs/page framing: the active bridge pipeline target is **Qwen3.5-2B**, not Qwen3-0.6B. The Qwen3-0.6B material remains documented only as a legacy CPU/local smoke and cheaper debug reproduction.
- **`ics/quantize.py`**: fixed `_int4_block_quantize` to use symmetric q range [-7, 7] with scale = absmax/7 instead of asymmetric [-8, 7]. The asymmetric range produced a 14% norm excess on the negative side and caused the dequantized model to diverge catastrophically (PPL ~1M instead of expected ~80-100). Note: this fix did NOT resolve the catastrophic PPL — the per-weight INT4 noise level is correct but the model's forward pass still produces garbage. Root cause still TBD.
- **`ics/gptq.py`**: same symmetric q range fix as `quantize.py` (qmin = -7 instead of -8).

### Bridge session log (2026-06-26 ~01:35-02:32 ET, before tunnel died)
- Synced `ics/`, `scripts/` to `/content/ICS/` on Colab.
- Installed `bitsandbytes==0.49.2` (was missing on fresh kernel).
- All 19 local tests pass on Colab after fix (10 correctness + 2 GQA + 7 pipeline smoke).
- Authenticated HF using env token; user = `toxzak`. Repos: `toxzak/Qwen3-0.6B-ICS-INT4` (private, was empty), `toxzak/Qwen3.5-2B-ICS-INT4` (public, had broken artifact from 2026-06-25 14:15 UTC), `toxzak/ics-quantization-artifacts` (private).
- Downloaded Qwen3-0.6B (1.5 GB) and Qwen3.5-2B (4.6 GB).
- **Confirmed existing `Qwen3.5-2B-ICS-INT4` artifact on HF is broken**: BF16 PPL 30.8 vs ICS-dequant PPL 1,170,034 (delta +3.8M%, logit max_diff 28.75).
- **Re-ran full pipeline on Qwen3-0.6B** (56 chains, 140 quantized layers, 67s pipeline). Uploaded artifact to `toxzak/Qwen3-0.6B-ICS-INT4`.
  - BEFORE the INT4 fix: BF16 PPL 79.89, ICS PPL 900,629 (catastrophic).
  - AFTER the INT4 fix: BF16 PPL 79.89, ICS PPL 1,250,178 (still catastrophic, slightly worse).
  - Per-weight dequant vs BF16 diff: mean 0.012-0.016 (INT4 noise level), max 0.55 on outliers. Looks correct at the weight level.
  - Layer-by-layer forward-pass hidden state diff: L0=0, L1=0.77, L2=3.28, L3=1824. Catastrophic blowup by layer 3.
  - Per-chain loading test: each chain individually produces INT4-level noise; catastrophic behavior only emerges when many chains loaded together. Cumulative INT4 noise compounds non-linearly.
- Tunnel unregistered at ~02:32 ET (kernel disconnected). Need new URL+token to continue.

## 2026-06-24

Daily sync.


## 2026-06-23

Daily sync.



## 2026-06-24

Daily sync.


## 2026-06-20

- Added a local Qwen3-0.6B smoke runner that can use cached Hugging Face snapshots, offline mode, bounded chain selection, and a low-memory Fisher objective.
- Fixed activation-Fisher collection so hooks retain gradients from the real loss graph while avoiding unnecessary parameter-gradient allocation.
- Added GQA-aware chain discovery for Qwen-style attention and quantization along the correct row/column dimension.
- Saved quantization dimension metadata in exports so dequantization can reconstruct row- and column-quantized tensors correctly.
- Added Qwen tokenizer loading guidance and verified the real tokenizer path with a one-chain Qwen3-0.6B pipeline smoke run.
- Expanded smoke tests for Fisher gradients, empty tokenization, chain filtering, and parameter-gradient suppression.
- Added a perplexity benchmark harness for HF dense, ICS-dequantized, and llama.cpp Q4_K_M GGUF comparisons, plus a tiny Qwen3-0.6B smoke result.

## 2026-06-24

Daily sync.


## 2026-06-17

Daily backup: 9 files changed (0 modified, 9 added, 0 deleted).

## 2026-06-24

Daily sync.


## 2026-06-21

- Daily auto-sync: refreshed .gitignore; updated scripts/benchmark_perplexity.py; added .env.example, colab_hf_github_pipeline.ipynb, enchmarks/qwen3_06b_full_quant_ppl.json, local Qwen eval transcript, and ssh_tunnels_and_how_to_dig_them (1).ipynb.

## 2026-07-01

Daily sync.
