# Prior Art Notes for ICS

This file is a packaging aid, not a novelty claim. It records the closest
quantization and outlier-handling work found while preparing the current
`per_row_int4` path.

## Current ICS Position

ICS is best described as a topology-preserving channel permutation plus an
export format for block or row-scaled quantized weights. The current practical
quality path is `quant_method="per_row_int4"`: one symmetric INT4 scale per
output row after optional chain permutation. The motivation is empirical:
per-group INT4 with absmax scaling can collapse on late transformer layers when
one large weight outlier dominates a 128-column group scale.

What is likely not novel:

- Using activation/Fisher-style statistics to decide what matters.
- Equivalent transformations that move quantization difficulty across weights,
  activations, or channels.
- Outlier mitigation via scaling, rotation, clipping, splitting, or reordering.
- Per-channel or per-row scaling itself.

What may remain distinctive enough to test:

- Restricting the transformation to exact chain-compatible permutations, so
  deployment can keep a contiguous static weight stream.
- Combining chain permutation with row-scaled INT4 as a conservative fallback
  when lower-bit block layouts fail.
- Targeting NPU-friendly static export rather than runtime sparse gathers or
  learned runtime rotations.

## Closest Work

- GPTQ: one-shot second-order weight quantization for transformer models. It is
  the main prior for Hessian-aware weight reconstruction, but it optimizes
  quantized weights rather than an exact channel permutation topology.
  Source: https://arxiv.org/abs/2210.17323

- SmoothQuant: migrates activation outlier difficulty into weights through an
  equivalent offline transformation, enabling W8A8 LLM inference. It is close in
  spirit because it uses mathematical equivalence to reshape quantization
  difficulty before deployment.
  Source: https://arxiv.org/abs/2211.10438

- AWQ: activation-aware weight quantization that identifies salient channels
  from activation statistics and protects them through scaling instead of
  hardware-inefficient mixed precision. This is close to ICS's use of
  activation/Fisher signals for deciding which channels or blocks matter.
  Source: https://arxiv.org/abs/2306.00978

- RPTQ: reorder-based post-training quantization. This is the closest
  high-level prior for channel reordering: it rearranges channels and quantizes
  them in clusters to reduce range differences, then fuses reorder overhead into
  adjacent operations.
  Source: https://arxiv.org/abs/2304.01089

- Outlier Channel Splitting: duplicates channels containing outliers and halves
  their values, preserving network function while reducing outlier magnitude.
  This is older non-LLM-specific evidence that exact functional transforms can
  improve post-training quantization without retraining.
  Source: https://arxiv.org/abs/1901.09504

- OmniQuant: optimizes quantization parameters through learnable weight clipping
  and learnable equivalent transformations. It is a strong prior for calibrated
  equivalent transformations, especially in low-bit LLM settings.
  Source: https://arxiv.org/abs/2308.13137

- QuaRot: applies rotations that preserve the model function while removing
  outliers, enabling end-to-end 4-bit inference. This is close in invariance
  logic but uses rotations rather than pure permutations.
  Source: https://arxiv.org/abs/2404.00456

- SpinQuant: learns rotations for quantization, improving on random rotations in
  difficult LLM cases. It reinforces that outlier-shaping transformations are a
  crowded area, and that stronger baselines are rotation-based.
  Source: https://arxiv.org/abs/2405.16406

- VPTQ: low-bit vector post-training quantization using second-order
  optimization, residual/outlier handling, and codebooks. It is less directly
  comparable to ICS but relevant as an aggressive low-bit compression baseline.
  Source: https://arxiv.org/abs/2409.17066

## Publication Implication

The defensible framing is not "new quantization beats GPTQ/AWQ." The safer
claim is narrower: ICS explores whether exact chain-compatible channel
permutations can make a static, contiguous, NPU-friendly quantized layout usable
without runtime sparse gathers. The current evidence still needs full-model,
fixed-dataset validation against GPTQ, AWQ, SmoothQuant-style scaling, and a
rotation baseline such as QuaRot or SpinQuant.
