# Resources Catalog

## Summary

This document catalogs the papers, datasets, and repositories gathered for the project on LayerNorm and concept association in creative tasks.

## Papers

Total papers downloaded: 9

| Title | Authors | Year | File | Key Info |
|------|---------|------|------|---------|
| On Layer Normalization in the Transformer Architecture | Xiong et al. | 2020 | `papers/2002.04745_xiong2020_layernorm_transformer.pdf` | Foundational LN placement/stability paper |
| On the Expressivity Role of LayerNorm in Transformers' Attention | Brody et al. | 2023 | `papers/2305.02582_brody2023_layernorm_expressivity.pdf` | LN changes attention geometry and selectability |
| Transformers without Normalization | Zhu et al. | 2025 | `papers/2503.10622_zhu2025_transformers_without_normalization.pdf` | DyT replacement for LN/RMSNorm |
| NoveltyBench | Zhang et al. | 2025 | `papers/2504.05228_zhang2025_noveltybench.pdf` | Direct diversity-collapse benchmark |
| The Curious Case of Neural Text Degeneration | Holtzman et al. | 2019 | `papers/1904.09751_holtzman2019_text_degeneration.pdf` | Canonical degeneration/decoding paper |
| Probing the Creativity of Large Language Models | Chen and Ding | 2023 | `papers/2310.11158_chen2023_llm_divergent_association.pdf` | DAT-based semantic creativity probe |
| Putting GPT-3's Creativity to the (Alternative Uses) Test | Stevenson et al. | 2022 | `papers/2206.08932_stevenson2022_gpt3_alternative_uses.pdf` | AUT benchmark against humans |
| Has the Creativity of Large-Language Models peaked? | Haase et al. | 2025 | `papers/2504.12320_haase2025_llm_creativity_peaked.pdf` | DAT/AUT variability across 14 LLMs |
| Only Connect Wall Dataset paper | Naeini et al. | 2023 | `papers/2306.11167_naeini2023_only_connect_wall.pdf` | Distractor-rich concept association benchmark |

See `papers/README.md` for more detail.

## Datasets

Total datasets downloaded: 2

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| NoveltyBench | `code/novelty-bench/data/` | 2,300 JSONL records | open-ended diversity evaluation | `datasets/novelty_bench/` | best direct test of collapse toward stereotyped outputs |
| Only Connect Wall (OCW) | Toronto-hosted dataset archive | 618 walls | creative grouping and connection naming | `datasets/ocw/dataset/` | strong benchmark for concept association under distractors |

See `datasets/README.md` for loading and download instructions.

## Code Repositories

Total repositories cloned: 5

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| divergent-association-task | https://github.com/jayolson/divergent-association-task | DAT scoring | `code/divergent-association-task/` | requires external embeddings |
| novelty-bench | https://github.com/novelty-bench/novelty-bench | diversity benchmark and evaluation code | `code/novelty-bench/` | includes packaged benchmark data |
| layer_norm_expressivity_role | https://github.com/tech-srl/layer_norm_expressivity_role | mechanistic LN experiments | `code/layer_norm_expressivity_role/` | some experiments need Gurobi |
| ocw | https://github.com/TaatiTeam/OCW | OCW evaluation and data utilities | `code/ocw/` | supports extra ablation datasets |
| removing-layer-norm | https://github.com/submarat/removing-layer-norm | LN-free GPT-2 fine-tuning/eval | `code/removing-layer-norm/` | high compute requirement |

See `code/README.md` for repo-specific notes and entry points.

## Resource Gathering Notes

### Search Strategy
- Started with the local `paper-finder` helper in diligent mode.
- Fell back to manual search when the local backend stalled.
- Used arXiv/project pages for papers and GitHub project repos for code/data.
- Prioritized resources that jointly cover LayerNorm mechanics and creativity/diversity benchmarks.

### Selection Criteria
- Direct relevance to LayerNorm role, removal, or placement
- Direct relevance to output diversity, degeneration, or creative semantic association
- Availability of code or reusable benchmark data
- Preference for benchmarks that support repeated generation and quantitative comparison

### Challenges Encountered
- The local paper-finder backend did not return promptly.
- Semantic Scholar API returned HTTP 429 rate limits.
- The Hugging Face `TaatiTeam/OCW` loader was incompatible with the installed `datasets` version because it still uses a legacy dataset script.
- `wget` was unavailable in the container, so `curl` was used for archive download.

### Gaps and Workarounds
- No paper directly links LayerNorm to creative collapse; this remains the central experimental gap.
- DAT participant data were not auto-downloaded from OSF, but scoring code is available locally and sufficient for model-side generation studies.
- AUT prompt/data assets were not bundled in an immediately reusable repo here, so AUT is documented as a follow-up benchmark rather than a downloaded primary dataset.

## Recommendations for Experiment Design

1. **Primary datasets**: Use NoveltyBench for generation diversity and OCW for concept association with distractors; add DAT scoring as a lightweight third axis.
2. **Baseline methods**: Compare a standard LN model against DyT or LN-free variants, with decoding held fixed.
3. **Evaluation metrics**: Use `distinct_k`, `utility_k`, DAT score, OCW grouping/connection metrics, plus repetition and sample variance.
4. **Code to adapt/reuse**: Start with `code/novelty-bench/` for evaluation scaffolding and `code/removing-layer-norm/` for LN-removal baselines; use `code/divergent-association-task/` and `code/ocw/` as task-specific evaluators.
