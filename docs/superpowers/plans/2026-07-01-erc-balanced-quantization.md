# ERC Balanced Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a balanced Error-Bounded Residual Cell guard that promotes unsafe INT1/INT2 blocks to INT4 before export.

**Architecture:** Keep the existing ICS permutation path unchanged. Add an optional post-assignment promotion pass in `ics.quantize` that scores each proposed low-bit block with Fisher-weighted reconstruction error, then promotes only blocks over a configurable budget. Store promotion metadata in `QuantizedTensor` and export it for diagnostics.

**Tech Stack:** Python, PyTorch, existing script-style tests.

---

### Task 1: Quantizer Behavior

**Files:**
- Modify: `ics/quantize.py`
- Test: `scripts/test_erc_quantize.py`

- [ ] **Step 1: Write failing tests**

Create tests that call the new `balance_bit_widths_by_error` API. One test uses a block with large outliers and a tight budget and expects promotion from INT1 to INT4. Another uses a generous budget and expects the low-bit assignment to stay unchanged.

- [ ] **Step 2: Run tests and verify red**

Run: `python scripts/test_erc_quantize.py`
Expected: fails because `balance_bit_widths_by_error` does not exist.

- [ ] **Step 3: Implement minimal API**

Add `erc_promoted` to `QuantizedTensor`, implement Fisher-weighted block error scoring, and call it from `quantize_blockwise` when requested.

- [ ] **Step 4: Run tests and verify green**

Run: `python scripts/test_erc_quantize.py`
Expected: pass.

### Task 2: Pipeline and Export Integration

**Files:**
- Modify: `ics/pipeline.py`
- Modify: `ics/export.py`
- Test: `scripts/test_erc_quantize.py`

- [ ] **Step 1: Write failing pipeline/config/export tests**

Assert `ICSConfig` exposes ERC knobs and exported metadata records `erc_promoted` for a saved tensor.

- [ ] **Step 2: Run tests and verify red**

Run: `python scripts/test_erc_quantize.py`
Expected: fails on missing config/export behavior.

- [ ] **Step 3: Wire implementation**

Add config fields, pass Fisher/error budget to `quantize_blockwise`, serialize `erc_promoted`, and load it back with backward compatibility.

- [ ] **Step 4: Run focused regression tests**

Run: `python scripts/test_erc_quantize.py`, `python scripts/test_correctness.py`, and `python scripts/test_pipeline_smoke.py`.

