# REPORT: LayerNorm and Concept Association in Creative Tasks

## 1. Executive Summary
This study tested whether LayerNorm contributes to generative collapse toward stereotyped, human-intuitive outputs by comparing vanilla `gpt2` against the released LN-free checkpoint `schaeff/gpt2-small_LNFree300`. The key result is mixed: removing LayerNorm substantially expanded hidden-state geometry on an association benchmark, but it did not produce a reliable gain in lexical diversity on creative generation prompts.

In concrete terms, LN removal raised semantic spread across repeated creative generations, reduced representational anisotropy, and nearly doubled effective rank on Only Connect Wall clue embeddings. At the same time, it slightly reduced `distinct_1` and `distinct_2`, slightly increased a lexical-overlap proxy, and did not significantly improve association separability. The practical implication is that LayerNorm appears to compress representation geometry, but architectural removal alone is not enough to turn that broader geometry into clearly more novel creative surface outputs in this GPT-2-family setting.

## 2. Research Question & Motivation
### Hypothesis
Layer normalization may contribute to collapse toward high-probability outputs by regularizing the residual stream into a geometry that favors safe, stereotyped concept associations over more incongruent or creative ones.

### Why this matters
The user’s framing points to two forms of collapse: sampling/decoding collapse and architectural collapse. Decoding and RLHF are already known contributors to bland or intuitive outputs, but if LayerNorm itself also narrows the usable representation geometry, then interventions aimed at creativity need to go beyond decoding tricks.

### Literature gap
The local literature review identified a direct gap: LayerNorm papers explain optimization and attention geometry, while creativity papers measure diversity or semantic association, but none of the gathered work directly tests whether LayerNorm changes creative-output variability or association structure. This project fills that gap with a matched local-model comparison.

## 3. Experimental Setup
### Models tested
- Baseline: `gpt2`
- Architectural variant: `schaeff/gpt2-small_LNFree300`

### Datasets
- NoveltyBench creativity subset: 20 prompts from `datasets/novelty_bench/curated.jsonl`
- Only Connect Wall validation split: 62 walls from `datasets/ocw/dataset/validation.json`

### Experimental protocol
1. Generate 12 responses per creativity prompt from each model with fixed sampling.
2. Measure lexical diversity, semantic spread, overlap, repetition, and sentence-count adherence.
3. Encode OCW clue strings with each model’s final hidden layer and compute geometry/association metrics per wall.
4. Run paired statistical tests across prompts or walls, then correct within each metric family using Benjamini-Hochberg FDR.

### Generation settings
- `temperature=0.9`
- `top_p=0.95`
- `max_new_tokens=80`
- `seed=42`

### Metrics
Creative generation:
- `distinct_1`, `distinct_2`
- Mean pairwise sentence-embedding cosine distance (`embedding_dispersion`)
- Mean pairwise bigram Jaccard overlap (`pairwise_bigram_jaccard`; higher means more overlap, so worse diversity)
- Mean repetition rate
- Sentence-count adherence when the prompt explicitly requested a sentence count

Association geometry:
- `association_auc`: AUROC for same-group vs different-group clue pairs using cosine similarity
- `similarity_gap`: within-group minus between-group cosine similarity
- `anisotropy`: mean pairwise cosine among clue embeddings
- `effective_rank`: participation-ratio style covariance rank

### Tools and environment
- Python `3.12.8`
- `torch 2.11.0+cu130`
- `transformers 5.7.0`
- `sentence-transformers 5.4.1`
- `scikit-learn 1.8.0`
- `scipy 1.17.1`
- `pandas 3.0.2`
- `matplotlib 3.10.9`
- `seaborn 0.13.2`

### Hardware
The workspace exposed four NVIDIA RTX A6000 GPUs (`49 GB` each), but Torch could not use CUDA because the installed build expected a newer driver than the container provided. As a result, the study executed on CPU. Total experiment runtime was `262.65s`.

### Reproducibility
- Environment file: `pyproject.toml`
- Main script: `src/run_layernorm_study.py`
- Metadata and run arguments: `results/layernorm_study/metadata.json`
- Raw generations: `results/layernorm_study/raw/`

## 4. Results
### 4.1 Creative generation metrics

| Metric | GPT-2 | LN-free GPT-2 | Delta (LN-free - GPT-2) | FDR p-value | Interpretation |
|---|---:|---:|---:|---:|---|
| `distinct_1` | 0.463 | 0.451 | -0.0117 | 0.258 | Slight lexical diversity drop |
| `distinct_2` | 0.899 | 0.884 | -0.0151 | 0.115 | Slight lexical diversity drop |
| `embedding_dispersion` | 0.803 | 0.831 | +0.0285 | 0.024 | Significant semantic spread increase |
| `pairwise_bigram_jaccard` | 0.00627 | 0.00806 | +0.00179 | 0.105 | More lexical overlap, not significant after FDR |
| `mean_repetition_rate` | 0.0277 | 0.0256 | -0.00206 | 0.766 | No meaningful change |
| `sentence_adherence` | 0.222 | 0.167 | -0.0556 | 0.600 | No reliable gain in coarse utility proxy |

Additional directional checks:
- LN-free GPT-2 had higher semantic spread on `15/20` prompts.
- LN-free GPT-2 beat vanilla on `distinct_1` for only `8/20` prompts and on `distinct_2` for `6/20`.

### 4.2 OCW association-geometry metrics

| Metric | GPT-2 | LN-free GPT-2 | Delta (LN-free - GPT-2) | FDR p-value | Interpretation |
|---|---:|---:|---:|---:|---|
| `association_auc` | 0.5271 | 0.5294 | +0.00233 | 0.706 | No significant separability gain |
| `similarity_gap` | 0.000581 | 0.001046 | +0.000465 | 0.299 | Small, non-significant improvement |
| `anisotropy` | 0.9915 | 0.9642 | -0.0273 | 8.5e-26 | Strong reduction in collapse/homogenization |
| `effective_rank` | 1.352 | 2.560 | +1.208 | 1.5e-11 | Strong increase in representational dimensionality |

Directional checks:
- LN-free GPT-2 had lower anisotropy on `62/62` walls.
- LN-free GPT-2 had higher effective rank on `62/62` walls.
- AUC improved on `33/62` walls, so the strong geometry change did not consistently translate into much better group separability.

### 4.3 Figures and artifacts
- Generation metric plot: `figures/layernorm_study/generation_metrics.png`
- OCW geometry plot: `figures/layernorm_study/ocw_metrics.png`
- Metric CSVs:
  - `results/layernorm_study/generation_metrics.csv`
  - `results/layernorm_study/ocw_metrics.csv`
- Statistical tests:
  - `results/layernorm_study/generation_stats.csv`
  - `results/layernorm_study/ocw_stats.csv`

## 5. Analysis & Discussion
### What the results show
The strongest evidence is internal, not surface-level. Removing LayerNorm consistently produced less anisotropic and higher-rank clue embeddings on OCW, which supports the user’s idea that LayerNorm affects residual-space geometry in a way that can compress representational variation. This result aligns with the literature suggesting that normalization changes attention/residual geometry rather than merely stabilizing training.

The generative results are more ambiguous. LN removal increased semantic spread across repeated responses, which is consistent with the hypothesis that the model accesses a broader region of concept space. However, the same model did not become more lexically distinct by `distinct_n`, and it showed slightly more pairwise bigram overlap. On these prompts, broader embedding dispersion did not cleanly become broader lexical creativity.

### Interpretation relative to the hypothesis
The hypothesis was partially supported.
- Supported: LayerNorm removal clearly changed geometry in the direction of less collapse.
- Partially supported: semantic response spread increased under fixed decoding.
- Not supported strongly enough: this did not become a robust gain in lexical novelty or OCW grouping performance.

The most defensible interpretation is that LayerNorm contributes to a compressed representation geometry, but the path from geometry to visibly creative output is mediated by other factors: pretrained distribution shape, small-model capacity, and prompt-following weakness in GPT-2.

### Alternative explanations
- The LN-free checkpoint may simply be noisier after post-training LN removal, which can raise embedding spread without improving useful creativity.
- GPT-2 is not instruction-tuned, so NoveltyBench prompts are a weak surface test of creativity quality.
- The clue-embedding OCW protocol measures association geometry indirectly from a causal LM rather than from a model trained specifically for semantic grouping.

### Qualitative notes
The raw outputs support the quantitative story. Both models produced weak prompt following, but the LN-free model more often wandered into semantically broader or more abrupt topic shifts. That matches the higher embedding-dispersion result, but it does not obviously look like better controlled creativity.

## 6. Limitations
- Model family limitation: this study used GPT-2 small and one LN-free derivative, not modern RLHF-tuned LLMs.
- Quality proxy limitation: sentence-count adherence is coarse, and only 3 prompts exposed that metric.
- Benchmark mismatch: NoveltyBench was designed for stronger instruction-following models, so absolute scores here should not be compared to frontier-model leaderboard results.
- Geometry task limitation: OCW embeddings were extracted from clue strings directly, not from a supervised grouping head or a task-trained association model.
- Hardware limitation: GPU acceleration was unavailable because the Torch CUDA build and installed driver were mismatched.

These limitations mean the study should be read as a mechanistic GPT-2-family probe, not a final statement about LayerNorm in modern aligned LLMs.

## 7. Conclusions & Next Steps
Removing LayerNorm from GPT-2 substantially reduced representational anisotropy and increased effective rank, which supports the idea that LayerNorm compresses concept geometry. But that broader geometry did not straightforwardly produce more lexically diverse or more accurate creative associations under fixed decoding. In this experiment, LayerNorm looked more like a geometry-shaping factor than a single direct cause of creative-output collapse.

Recommended follow-up experiments:
1. Repeat the same protocol on a stronger open-weight model family with matched normalized and de-normalized variants.
2. Add a quality-controlled evaluation layer, such as human or strong-LLM judging, to separate useful novelty from incoherence.
3. Test decoding interactions directly: compare LN and LN-free models under greedy, temperature, and nucleus sampling to estimate architecture-by-decoder interactions.
4. Replace direct clue-string embeddings with task-conditioned OCW prompts or a learned association probe to test whether geometric expansion becomes more useful under explicit reasoning.

## References
1. Xiong et al. (2020). *On Layer Normalization in the Transformer Architecture.*
2. Brody, Alon, Yahav (2023). *On the Expressivity Role of LayerNorm in Transformers' Attention.*
3. Zhu et al. (2025). *Transformers without Normalization.*
4. Holtzman et al. (2019). *The Curious Case of Neural Text Degeneration.*
5. Zhang et al. (2025). *NoveltyBench: Evaluating Language Models for Humanlike Diversity.*
6. Chen and Ding (2023). *Probing the Creativity of Large Language Models: Can Models Produce Divergent Semantic Association?*
7. Naeini et al. (2023). *Only Connect Wall Dataset.*
