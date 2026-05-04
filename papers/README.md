# Downloaded Papers

1. [On Layer Normalization in the Transformer Architecture](2002.04745_xiong2020_layernorm_transformer.pdf)
   - Authors: Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, Tie-Yan Liu
   - Year: 2020
   - arXiv: 2002.04745
   - Why relevant: canonical analysis of Pre-LN vs Post-LN training dynamics; useful when designing LayerNorm ablations.

2. [On the Expressivity Role of LayerNorm in Transformers' Attention](2305.02582_brody2023_layernorm_expressivity.pdf)
   - Authors: Shaked Brody, Uri Alon, Eran Yahav
   - Year: 2023
   - arXiv: 2305.02582
   - Why relevant: argues LayerNorm changes the geometry and expressivity of attention, not only optimization stability.

3. [Transformers without Normalization](2503.10622_zhu2025_transformers_without_normalization.pdf)
   - Authors: Jiachen Zhu, Xinlei Chen, Kaiming He, Yann LeCun, Zhuang Liu
   - Year: 2025
   - arXiv: 2503.10622
   - Why relevant: strong recent evidence that LayerNorm can be replaced with DyT while preserving performance, making direct LN-removal ablations practical.

4. [NoveltyBench: Evaluating Language Models for Humanlike Diversity](2504.05228_zhang2025_noveltybench.pdf)
   - Authors: Yiming Zhang, Harshita Diddee, Susan Holm, Hanchen Liu, Xinyue Liu, Vinay Samuel, Barry Wang, Daphne Ippolito
   - Year: 2025
   - arXiv: 2504.05228
   - Why relevant: directly measures mode collapse and diversity loss in open-ended generation.

5. [The Curious Case of Neural Text Degeneration](1904.09751_holtzman2019_text_degeneration.pdf)
   - Authors: Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi
   - Year: 2019
   - arXiv: 1904.09751
   - Why relevant: foundational paper on bland/repetitive high-probability outputs and diversity-preserving decoding.

6. [Probing the Creativity of Large Language Models: Can Models Produce Divergent Semantic Association?](2310.11158_chen2023_llm_divergent_association.pdf)
   - Authors: Honghua Chen, Nai Ding
   - Year: 2023
   - arXiv: 2310.11158
   - Why relevant: uses the Divergent Association Task (DAT) to probe creativity-like semantic association in LLMs.

7. [Putting GPT-3's Creativity to the (Alternative Uses) Test](2206.08932_stevenson2022_gpt3_alternative_uses.pdf)
   - Authors: Claire Stevenson, Iris Smal, Matthijs Baas, Raoul Grasman, Han van der Maas
   - Year: 2022
   - arXiv: 2206.08932
   - Why relevant: compares LLM outputs to human responses on a classic divergent thinking task.

8. [Has the Creativity of Large-Language Models peaked? An analysis of inter- and intra-LLM variability](2504.12320_haase2025_llm_creativity_peaked.pdf)
   - Authors: Jennifer Haase, Paul H. P. Hanel, Sebastian Pokutta
   - Year: 2025
   - arXiv: 2504.12320
   - Why relevant: recent benchmark-oriented analysis of AUT and DAT across many current models, including output variability.

9. [Large Language Models are Fixated by Red Herrings: Exploring Creative Problem Solving and Einstellung Effect using the Only Connect Wall Dataset](2306.11167_naeini2023_only_connect_wall.pdf)
   - Authors: Saeid Naeini, Raeid Saqur, Mozhgan Saeidi, John Giorgi, Babak Taati
   - Year: 2023
   - arXiv: 2306.11167
   - Why relevant: provides an associative creative problem-solving benchmark with distractors and human baselines.

## Detailed Reading Artifacts

- Chunked PDFs for deeper review are in `papers/pages/`.
- First-five-page text extractions used for note-taking are in `artifacts/`.
