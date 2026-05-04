# Research Plan: LayerNorm and Concept Association in Creative Tasks

## Motivation & Novelty Assessment

### Why This Research Matters
Creative generation tasks expose a persistent gap between the number of plausible concept combinations a language model could in principle represent and the smaller set of high-probability outputs it actually emits. If part of that gap is architectural rather than purely a consequence of decoding or alignment, then understanding LayerNorm's role could inform both model design and interventions aimed at improving novelty without destroying coherence.

### Gap in Existing Work
The collected literature separates into two mostly disconnected lines: LayerNorm papers explain optimization and attention geometry, while creativity/diversity papers measure output collapse, semantic association, or distractor sensitivity. None of the gathered work directly tests whether LayerNorm changes representational geometry in a way that measurably alters creative output diversity or concept-association structure.

### Our Novel Contribution
This project links these literatures with a direct architectural comparison between a vanilla GPT-2 model and a matched LN-free GPT-2 variant. The contribution is not a new benchmark, but a cross-level analysis: we test whether removing LayerNorm changes both internal geometry on an association benchmark and external diversity on creative prompts.

### Experiment Justification
- Experiment 1: NoveltyBench creativity generations. Needed to test whether LN status changes the diversity and stereotypy of open-ended creative outputs under fixed decoding.
- Experiment 2: OCW association geometry. Needed to test whether LN status changes semantic grouping structure in hidden representations, which addresses the user's geometry/association hypothesis directly.
- Experiment 3: Robustness across decoding seeds and prompt instances. Needed to distinguish stable architecture effects from one-off sampling noise.

## Research Question
Does LayerNorm in GPT-2-class transformers contribute to representational and generative collapse toward more stereotyped outputs, reducing diversity and weakening non-obvious concept associations in creative tasks?

## Background and Motivation
Prior work suggests three relevant facts. First, LayerNorm changes transformer behavior beyond optimization, affecting residual-space projection and token selectability. Second, open-ended generation often collapses toward high-probability, low-diversity continuations. Third, creativity and association benchmarks show that LLMs can be competent on average while still under-expressing exceptional or unusual associations. This project tests whether those observations connect through LayerNorm.

## Hypothesis Decomposition
- H1: Under matched sampling settings, the LN-free GPT-2 variant will produce more diverse responses than vanilla GPT-2 on creative prompts.
- H2: The LN-free GPT-2 variant will show less collapsed hidden-state geometry, operationalized as lower anisotropy and higher effective rank on association stimuli.
- H3: If LayerNorm contributes to concept-association collapse, same-group vs different-group clue separability on OCW will change measurably between the two models.
- Alternative explanation A: Any diversity increase is just model degradation or noisier decoding, not more useful creativity.
- Alternative explanation B: Any geometry change does not translate into output behavior.
- Alternative explanation C: Effects are specific to GPT-2 and do not generalize to modern aligned LLMs.

## Proposed Methodology

### Approach
Use a controlled, local-model comparison between `gpt2` and `schaeff/gpt2-small_LNFree300`, a released LN-free GPT-2 variant produced by post-training LN removal. Hold decoding settings fixed, run repeated generations on creative prompts, and pair output metrics with internal geometry metrics on the Only Connect Wall dataset. This design is feasible in one session, directly tied to the hypothesis, and avoids confounding architecture effects with provider-level RLHF differences.

### Experimental Steps
1. Verify and document environment, GPU availability, random seeds, and package versions.
2. Load the NoveltyBench creativity subset and OCW validation/test partitions.
3. Run repeated generations from both models on the 20 creativity prompts with fixed sampling parameters.
4. Compute creative-output metrics: distinct-1/2, self-BLEU proxy via pairwise n-gram overlap, sentence-embedding dispersion, repetition rate, and sentence-count adherence.
5. Encode OCW clues in a neutral carrier prompt with both models and extract final-layer hidden representations.
6. Compute geometry metrics on OCW clues: same-group vs different-group cosine-separability AUC, within-minus-between similarity gap, anisotropy, and effective rank.
7. Aggregate per-prompt and per-wall statistics, run paired statistical tests, estimate effect sizes, and generate figures.

### Baselines
- Primary baseline: vanilla `gpt2`
- Architectural comparison model: `schaeff/gpt2-small_LNFree300`
- Decoding control: fixed temperature/top-p across both models
- Statistical control: per-prompt and per-wall paired comparisons

### Evaluation Metrics
- Creative diversity:
  - `distinct_1`, `distinct_2`: lexical diversity across generations
  - Mean pairwise embedding distance: semantic spread across generations
  - Mean pairwise n-gram overlap: stereotypy/self-similarity proxy
  - Repetition rate: degeneracy proxy
  - Sentence-count adherence: coarse prompt-following/utility proxy
- Association geometry:
  - Same-vs-different group AUROC from cosine similarity
  - Mean within-group minus between-group similarity gap
  - Embedding anisotropy: mean pairwise cosine among clue embeddings
  - Effective rank / participation ratio of clue-embedding covariance

### Statistical Analysis Plan
- Unit of analysis for generation metrics: prompt.
- Unit of analysis for geometry metrics: wall.
- Null hypotheses:
  - H0-1: No difference in prompt-level diversity metrics between vanilla and LN-free models.
  - H0-2: No difference in wall-level geometry metrics between vanilla and LN-free models.
- Tests:
  - Shapiro-Wilk on paired differences to assess normality.
  - Paired t-test if normality is not rejected; otherwise Wilcoxon signed-rank.
  - Two-sided alpha = 0.05.
  - Benjamini-Hochberg correction within each metric family.
- Effect sizes:
  - Cohen's d for paired differences when using t-tests.
  - Rank-biserial correlation for Wilcoxon tests.
- Uncertainty:
  - 95% bootstrap confidence intervals over prompts/walls for headline metrics.

## Expected Outcomes
Support for the hypothesis would look like: LN-free GPT-2 shows higher diversity and semantic spread on creativity prompts, together with less anisotropic or more separable OCW clue geometry. Refutation would look like negligible differences, or diversity gains that coincide with obvious quality collapse and weaker association structure.

## Timeline and Milestones
1. Planning and environment verification: 20-30 min
2. Dependency installation and pilot runs: 20-30 min
3. Script implementation: 45-75 min
4. Experiment execution: 45-90 min
5. Analysis, figures, and report: 45-60 min
6. Validation and reruns of key checks: 15-25 min

## Potential Challenges
- GPT-2 is not instruction-tuned, so absolute creativity quality may be low.
- The LN-free model may be noisier because post-training LN removal can degrade perplexity.
- OCW clue embeddings from a causal LM are an indirect association measure.
- Sentence-embedding metrics can overestimate diversity if outputs become incoherent.

Mitigations:
- Emphasize paired, same-family comparison rather than absolute model quality.
- Include coarse adherence/repetition metrics to separate diversity from collapse.
- Report limitations explicitly and avoid overgeneralizing beyond GPT-2-class models.

## Success Criteria
- Both models load and run reproducibly on local hardware.
- At least one full creative-output comparison and one association-geometry comparison complete with saved raw results.
- Statistical tests and figures are generated from actual runs.
- REPORT.md states a defensible answer to the hypothesis with explicit limitations.
