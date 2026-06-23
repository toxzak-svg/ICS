# Changelog

All notable changes to this project will be documented in this file.

## 2026-06-23

Daily sync.



## 2026-06-20

- Added a local Qwen3-0.6B smoke runner that can use cached Hugging Face snapshots, offline mode, bounded chain selection, and a low-memory Fisher objective.
- Fixed activation-Fisher collection so hooks retain gradients from the real loss graph while avoiding unnecessary parameter-gradient allocation.
- Added GQA-aware chain discovery for Qwen-style attention and quantization along the correct row/column dimension.
- Saved quantization dimension metadata in exports so dequantization can reconstruct row- and column-quantized tensors correctly.
- Added Qwen tokenizer loading guidance and verified the real tokenizer path with a one-chain Qwen3-0.6B pipeline smoke run.
- Expanded smoke tests for Fisher gradients, empty tokenization, chain filtering, and parameter-gradient suppression.
- Added a perplexity benchmark harness for HF dense, ICS-dequantized, and llama.cpp Q4_K_M GGUF comparisons, plus a tiny Qwen3-0.6B smoke result.

## 2026-06-17

Daily backup: 9 files changed (0 modified, 9 added, 0 deleted).

## 2026-06-21

- Daily auto-sync: refreshed .gitignore; updated scripts/benchmark_perplexity.py; added .env.example, colab_hf_github_pipeline.ipynb, enchmarks/qwen3_06b_full_quant_ppl.json, local Qwen eval transcript, and ssh_tunnels_and_how_to_dig_them (1).ipynb.

