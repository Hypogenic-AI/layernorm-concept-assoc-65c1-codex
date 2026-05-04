# Outline: LayerNorm and Concept Association in Creative Tasks

## Title
- LayerNorm broadens representation geometry without improving creative surface diversity in GPT-2

## Abstract
- Context: creative generation often collapses toward stereotyped outputs.
- Gap: prior work studies LayerNorm geometry or creative diversity, but not their direct link.
- Approach: compare `gpt2` against `schaeff/gpt2-small_LNFree300` on NoveltyBench creativity prompts and Only Connect Wall geometry.
- Key evidence: semantic dispersion increases; anisotropy drops; effective rank rises; lexical diversity does not improve reliably.
- Takeaway: LayerNorm shapes representation geometry, but removing it alone does not yield more novel surface outputs.

## Introduction
- Hook: creative failure can come from architecture, not only decoding.
- Importance: if normalization narrows concept space, creativity interventions must go beyond sampling tricks.
- Gap: no direct matched-model test of LayerNorm and creative/associative collapse.
- Approach: paired GPT-2 vs LN-free GPT-2 study across output metrics and hidden-state geometry.
- Quantitative preview: +0.0285 embedding dispersion, -0.0273 anisotropy, +1.208 effective rank, no significant gain in distinctness or AUC.
- Contributions:
  - We test a direct architecture-level hypothesis about LayerNorm and creative collapse.
  - We connect surface generation metrics to internal geometry metrics in one protocol.
  - We show strong geometric changes without matching lexical-diversity gains.
  - We release a reproducible local CPU-based pipeline with saved generations and statistics.

## Related Work
- Theme 1: LayerNorm as optimization and expressivity mechanism.
  - Xiong et al. (2020), Brody et al. (2023), Zhu et al. (2025).
- Theme 2: degeneration and diversity in open-ended generation.
  - Holtzman et al. (2020), Zhang et al. (2025).
- Theme 3: creativity and association benchmarks.
  - Chen and Ding (2023), Stevenson et al. (2022), Haase et al. (2025), Naeini et al. (2023).
- Positioning: unlike prior work, evaluate matched normalized vs LN-free models on both creative outputs and association geometry.

## Methodology
- Problem statement and hypotheses H1-H3.
- Models: `gpt2` and `schaeff/gpt2-small_LNFree300`.
- Datasets:
  - 20 NoveltyBench creativity prompts.
  - 62 OCW validation walls.
- Generation protocol:
  - 12 samples per prompt, temperature 0.9, top-p 0.95, max_new_tokens 80, seed 42.
- Metrics:
  - Creative: `distinct_1`, `distinct_2`, embedding dispersion, bigram Jaccard, repetition, sentence adherence.
  - Geometry: AUC, similarity gap, anisotropy, effective rank.
- Statistics:
  - prompt- or wall-level paired tests, Shapiro-Wilk gate, paired t-test or Wilcoxon, BH-FDR.
- Implementation:
  - Python stack, CPU runtime 262.65s, CUDA unavailable despite 4 RTX A6000 GPUs due to driver mismatch.

## Results
- Table 1: creative generation metrics with means, 95% bootstrap CIs, delta, FDR p-values.
- Table 2: OCW geometry metrics with means, 95% bootstrap CIs, delta, FDR p-values.
- Figure 1: generation metrics plot.
- Figure 2: OCW geometry plot.
- Main findings:
  - significant gain in embedding dispersion only on generation side.
  - large significant drop in anisotropy and rise in effective rank.
  - no significant AUC or similarity-gap gain.
  - directional checks: 15/20 prompts higher dispersion; 62/62 walls lower anisotropy and higher rank.

## Discussion
- Interpretation: LayerNorm compresses residual-space geometry.
- But broader geometry does not automatically translate into controllable lexical novelty.
- Alternative explanations: noisier LN-free checkpoint, non-instruction-tuned GPT-2, indirect OCW embedding probe.
- Limitations: model family, benchmark mismatch, sparse adherence metric, CPU-only execution.
- Broader implication: creativity bottlenecks are partly architectural but remain mediated by training distribution and decoding.

## Conclusion
- Restate contribution and answer: partial support for hypothesis.
- Key takeaway: LayerNorm is a geometry-shaping factor, not a single direct cause of creative-output collapse.
- Future work: stronger model families, human or strong-LLM judging, decoder interaction studies, task-conditioned probes.
