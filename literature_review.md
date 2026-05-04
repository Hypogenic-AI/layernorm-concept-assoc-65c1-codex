# Literature Review: LayerNorm and Concept Association in Creative Tasks

## Review Scope

### Research Question
Does LayerNorm contribute to generative collapse toward safe, high-probability outputs, reducing creative or incongruent concept association in language-model creative tasks?

### Inclusion Criteria
- Transformer or LLM papers that analyze LayerNorm function, placement, or removal
- Papers on degeneration, diversity collapse, or open-ended generation quality
- Papers that operationalize creativity through divergent thinking, semantic association, or distractor-heavy association tasks
- Resources with runnable code or downloadable benchmarks when available

### Exclusion Criteria
- Purely vision-only normalization papers without transferable transformer insight
- General creativity commentary without measurable benchmarks
- Papers with no clear relevance to either LayerNorm or creative/diversity evaluation

### Time Frame
- Foundations: 2019-2020
- Primary emphasis: 2022-2025

### Sources
- arXiv
- GitHub repositories linked from papers/project pages
- manual fallback search after the local paper-finder backend stalled

## Search Log

| Date | Query | Source | Notes |
|------|-------|--------|-------|
| 2026-05-04 | `layer normalization concept association creativity generative models` | local paper-finder helper | backend hung; switched to manual search |
| 2026-05-04 | layer norm / transformer / creativity / degeneration queries | arXiv web search | used for paper acquisition |
| 2026-05-04 | benchmark and dataset searches | GitHub + project pages + HF | used for code and data acquisition |

## Screening Results

| Paper | Decision | Reason |
|------|----------|--------|
| Xiong et al. 2020 | Include | foundational LN placement and stability analysis |
| Brody et al. 2023 | Include | direct claim that LN changes attention expressivity |
| Zhu et al. 2025 | Include | strongest recent normalization-removal result |
| Holtzman et al. 2019 | Include | canonical degeneration paper |
| Zhang et al. 2025 (NoveltyBench) | Include | direct benchmark for response diversity collapse |
| Chen and Ding 2023 | Include | direct creativity probe via DAT |
| Stevenson et al. 2022 | Include | AUT-based human vs GPT comparison |
| Haase et al. 2025 | Include | current multi-model creativity variability study |
| Naeini et al. 2023 | Include | distractor-heavy association benchmark |

## Key Papers

### On Layer Normalization in the Transformer Architecture
- **Authors**: Xiong et al.
- **Year**: 2020
- **Source**: arXiv / widely cited transformer optimization paper
- **Key Contribution**: explains why LN placement changes gradient behavior; Pre-LN avoids unstable gradients seen in Post-LN.
- **Methodology**: mean-field analysis plus transformer experiments comparing Pre-LN and Post-LN.
- **Datasets Used**: multiple transformer application settings; the paper is more about optimization dynamics than creativity.
- **Results**: Pre-LN removes the need for careful warm-up while preserving performance.
- **Code Available**: not central for this project.
- **Relevance to Our Research**: if LN affects gradient flow and residual scaling, it may also affect how sharply models concentrate probability mass during training.

### On the Expressivity Role of LayerNorm in Transformers' Attention
- **Authors**: Brody, Alon, Yahav
- **Year**: 2023
- **Source**: Findings of ACL 2023
- **Key Contribution**: argues LN changes what attention can represent, not just how it optimizes.
- **Methodology**: geometric decomposition of LN into projection and scaling; theory plus majority and unselectable-key experiments.
- **Datasets Used**: synthetic majority task, language-modeling-oriented experiments.
- **Results**: projection helps queries attend uniformly when needed; scaling prevents “unselectable” keys.
- **Code Available**: Yes, cloned in `code/layer_norm_expressivity_role/`.
- **Relevance to Our Research**: highly relevant mechanistically. If LN shapes attention geometry toward regularized, comparable token representations, it is a plausible contributor to homogenized association structure.

### Transformers without Normalization
- **Authors**: Zhu et al.
- **Year**: 2025
- **Source**: arXiv
- **Key Contribution**: shows DyT can replace normalization layers across transformer families.
- **Methodology**: substitute LN/RMSNorm with `tanh(alpha x)` plus affine parameters, keeping most training settings fixed.
- **Datasets Used**: ImageNet, LibriSpeech-style speech setup, diffusion models, and language settings.
- **Results**: normalized transformers can often be matched or exceeded without explicit normalization.
- **Code Available**: project page/code referenced in the paper.
- **Relevance to Our Research**: strongest experimental justification for running LN-free or LN-reduced creative-task ablations.

### The Curious Case of Neural Text Degeneration
- **Authors**: Holtzman et al.
- **Year**: 2019
- **Source**: ICLR 2020
- **Key Contribution**: high-probability decoding leads to bland repetition; nucleus sampling better matches human text diversity.
- **Methodology**: compares beam search, top-k, pure sampling, and nucleus sampling against human text distributions.
- **Datasets Used**: open-ended long-form generation with GPT-2 and human reference text.
- **Results**: beam search is inappropriate for open-ended generation; truncating the unreliable tail preserves quality and diversity.
- **Code Available**: not needed here.
- **Relevance to Our Research**: this is the clearest prior evidence for “collapse toward human-intuitive/high-probability outputs,” though it addresses decoding rather than architecture.

### NoveltyBench: Evaluating Language Models for Humanlike Diversity
- **Authors**: Zhang et al.
- **Year**: 2025
- **Source**: COLM 2025
- **Key Contribution**: formal benchmark for multiple distinct, high-quality generations.
- **Methodology**: 1,100 prompts, equivalence-class partitioning, `distinct_k` and `utility_k` metrics.
- **Datasets Used**: NB-Curated and NB-WildChat.
- **Results**: frontier models are much less diverse than humans; larger models in a family are often less novel than smaller ones.
- **Code Available**: Yes, cloned in `code/novelty-bench/`.
- **Relevance to Our Research**: primary benchmark for testing the hypothesis.

### Probing the Creativity of Large Language Models: Can Models Produce Divergent Semantic Association?
- **Authors**: Chen, Ding
- **Year**: 2023
- **Source**: arXiv
- **Key Contribution**: evaluates LLMs with DAT, a semantic-distance creativity measure.
- **Methodology**: compare multiple LLMs and decoding strategies on generation of unrelated nouns; compute cosine-distance-based DAT score.
- **Datasets Used**: DAT framework, with human comparisons.
- **Results**: GPT-4 exceeds most humans on greedy DAT score; stochastic decoding can raise creativity scores for smaller models but reduces stability.
- **Code Available**: Yes, linked in paper; related task code cloned via DAT repo.
- **Relevance to Our Research**: supplies a direct concept-association evaluation axis.

### Putting GPT-3's Creativity to the (Alternative Uses) Test
- **Authors**: Stevenson et al.
- **Year**: 2022
- **Source**: arXiv / ICCC
- **Key Contribution**: compares GPT-3 against humans on AUT.
- **Methodology**: evaluate originality, usefulness, surprise, flexibility, and semantic-distance proxies on alternative object uses.
- **Datasets Used**: Guilford’s Alternative Uses Test with human comparison data.
- **Results**: humans still outperform GPT-3 overall on creative output quality.
- **Code Available**: not central here.
- **Relevance to Our Research**: AUT remains a useful complementary benchmark to DAT and NoveltyBench.

### Has the Creativity of Large-Language Models peaked?
- **Authors**: Haase, Hanel, Pokutta
- **Year**: 2025
- **Source**: arXiv
- **Key Contribution**: evaluates 14 LLMs on DAT and AUT with inter- and intra-model variability.
- **Methodology**: repeated prompting across models and prompt formulations.
- **Datasets Used**: DAT and AUT.
- **Results**: average AUT performance can exceed human average, but truly exceptional creativity remains rare; output variability is large.
- **Code Available**: not inspected here.
- **Relevance to Our Research**: suggests repeated sampling is necessary and single-run evaluation will be noisy.

### Only Connect Wall Dataset
- **Authors**: Naeini et al.
- **Year**: 2023
- **Source**: NeurIPS Datasets and Benchmarks
- **Key Contribution**: creative association benchmark with built-in distractors and fixation pressure.
- **Methodology**: 618 wall puzzles; grouping task plus connection-naming task; compares embeddings, PLMs, and LLMs.
- **Datasets Used**: OCW, OCW-Randomized, OCW-WordNet.
- **Results**: even strong LLMs underperform expert humans; more in-context examples do not fix the core difficulty.
- **Code Available**: Yes, cloned in `code/ocw/`.
- **Relevance to Our Research**: excellent concept-association task that may expose whether LN contributes to brittle, stereotyped grouping behavior.

## Common Methodologies

- **Normalization analysis**: mean-field theory, geometric decomposition, and LN-removal substitutions.
- **Creativity measurement**: DAT semantic distance, AUT ratings/semantic proxies, distractor-heavy puzzle solving, and diversity-over-multiple-generations metrics.
- **Diversity evaluation**: repetition statistics, equivalence clustering, `distinct_k`, `utility_k`, and human preference judgments.

## Standard Baselines

- **Architectural baselines**: Pre-LN transformer, Post-LN transformer, RMSNorm variants, LN-free / DyT variants.
- **Decoding baselines**: greedy, top-k, temperature sampling, nucleus sampling.
- **Creativity-task baselines**: human responses, GPT-family models, and embedding-based association methods.

## Evaluation Metrics

- **Diversity**: `distinct_k`, `utility_k`, self-BLEU-style overlap, repetition rate.
- **Semantic creativity**: DAT score based on pairwise cosine distance among generated words.
- **Divergent thinking**: AUT ratings for originality, usefulness, surprise, flexibility, plus semantic distance.
- **Association puzzles**: exact grouping accuracy, AMI/ARI/FMS/Wasserstein for grouping, and exact match/ROUGE/BERTScore for named connections.

## Datasets in the Literature

- **NoveltyBench**: open-ended generation diversity benchmark; best direct fit for the hypothesis.
- **DAT**: prompt-based semantic-association measure; operationalizes remote association.
- **AUT**: classic divergent-thinking benchmark; useful but requires careful scoring.
- **OCW**: red-herring-rich concept grouping benchmark; useful for fixation-style errors.

## Gaps and Opportunities

- **Direct gap**: none of the collected papers directly test whether LayerNorm itself causes lower creativity or more stereotyped concept association.
- **Mechanistic gap**: LN papers explain expressivity and stability, but do not evaluate output diversity on creative tasks.
- **Benchmark gap**: creativity papers compare models and decoding strategies, but rarely architecture-level normalization ablations.
- **Experimental opportunity**: evaluate matched models with LN, RMSNorm, DyT, and LN-removed variants on NoveltyBench, DAT, and OCW.

## Recommendations for Our Experiment

- **Recommended datasets**: NoveltyBench for open-ended diversity, OCW for associative problem solving, DAT scoring via the cloned task code, and AUT if a curated prompt set is added later.
- **Recommended baselines**: standard LN model, LN-free or DyT replacement, and a decoding control condition using greedy vs nucleus sampling.
- **Recommended metrics**: `distinct_k`, `utility_k`, DAT score, OCW grouping metrics, repetition statistics, and intra-prompt variance across repeated samples.
- **Methodological considerations**:
  - Hold decoding constant when testing architectural effects, otherwise LN effects will be confounded with sampling effects.
  - Evaluate repeated generations per prompt; single samples will hide diversity differences.
  - Compare both quality and novelty, because higher novelty alone can drift into nonsense.
  - Include an inference-only LN removal condition when feasible, because `code/removing-layer-norm/` suggests that some LN effects may persist less strongly at inference than assumed.
