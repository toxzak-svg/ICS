# ICS Benchmark Results

- Model: `Qwen/Qwen3-0.6B`
- Evaluation label: `smoke`
- Git commit: `4038de3112f62b7507d04ad2b7af32c6632ecd4b`
- Dirty worktree: `True`

| method | baseline PPL | ICS PPL | quality | status | artifact MiB | est. BPW | tensors | base unchanged | note |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| block | 92.9362 | 26449.9832 | 0.0035 | fail | 15.2443 | 20.3257 | 4 | 307 | 4 dequantized tensors loaded; 307 base tensors unchanged |
| gptq | 92.9362 | 73.1526 | 1.2704 | pass | 17.1276 | 22.8368 | 4 | 307 | 4 dequantized tensors loaded; 307 base tensors unchanged |
| per_row_int4 | 92.9362 | 131.0001 | 0.7094 | fail | 17.1264 | 22.8352 | 4 | 307 | 4 dequantized tensors loaded; 307 base tensors unchanged |

## Caveats

- PPL is the measured quantity; acceptance uses baseline-relative quality.
- Built-in eval text is a smoke check, not paper-grade evidence.
- gptq is the local ICS GPTQ backend, not a new GPTQ algorithm claim.
- External AWQ/RPTQ/QuaRot/SpinQuant baselines are intentionally out of v1.

## Commands

### block

```powershell
C:\Python314\python.exe scripts/benchmark_perplexity.py --model Qwen/Qwen3-0.6B --device cpu --dtype fp16 --quant-method block --ics-output .pytest_cache\ics_harness\artifacts\qwen06_block --output-json benchmarks\qwen3_06b_harness\runs\qwen06_block.json --max-eval-tokens 2048 --max-calibration-samples 24 --max-calibration-length 256 --min-quality-score 0.9 --rebuild-ics --local-files-only --offline
```

### gptq

```powershell
C:\Python314\python.exe scripts/benchmark_perplexity.py --model Qwen/Qwen3-0.6B --device cpu --dtype fp16 --quant-method gptq --ics-output .pytest_cache\ics_harness\artifacts\qwen06_gptq --output-json benchmarks\qwen3_06b_harness\runs\qwen06_gptq.json --max-eval-tokens 2048 --max-calibration-samples 24 --max-calibration-length 256 --min-quality-score 0.9 --rebuild-ics --local-files-only --offline
```

### per_row_int4

```powershell
C:\Python314\python.exe scripts/benchmark_perplexity.py --model Qwen/Qwen3-0.6B --device cpu --dtype fp16 --quant-method per_row_int4 --ics-output .pytest_cache\ics_harness\artifacts\qwen06_per_row_int4 --output-json benchmarks\qwen3_06b_harness\runs\qwen06_per_row_int4.json --max-eval-tokens 2048 --max-calibration-samples 24 --max-calibration-length 256 --min-quality-score 0.9 --rebuild-ics --local-files-only --offline
```
