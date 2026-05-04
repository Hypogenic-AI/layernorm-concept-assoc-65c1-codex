# Cloned Repositories

## Repo 1: Divergent Association Task
- URL: https://github.com/jayolson/divergent-association-task
- Commit: `9978dd8103670a90c59bc35a7210acc60995dcdb`
- Purpose: compute DAT creativity scores from semantically distant word lists
- Location: `code/divergent-association-task/`
- Key files: `dat.py`, `examples.py`, `words.txt`
- Notes: useful for scoring semantic-distance creativity outputs; requires an external embedding file such as GloVe.

## Repo 2: NoveltyBench
- URL: https://github.com/novelty-bench/novelty-bench
- Commit: `984876e92a5eefe6945f273b8c8543a4e6cff070`
- Purpose: benchmark for response diversity and quality in open-ended generation
- Location: `code/novelty-bench/`
- Key files: `src/inference.py`, `src/partition.py`, `src/score.py`, `src/summarize.py`, `data/*.jsonl`
- Notes: most directly reusable benchmark for testing collapse onto high-probability outputs; packaged data were copied into `datasets/novelty_bench/`.

## Repo 3: LayerNorm Expressivity Role
- URL: https://github.com/tech-srl/layer_norm_expressivity_role
- Commit: `0a6e74f3d18d66dded55cc06ddcc83c3d0652978`
- Purpose: reproduce theory/experiments from the ACL 2023 paper on LayerNorm and attention expressivity
- Location: `code/layer_norm_expressivity_role/`
- Key files: `majority/`, `unselectable/`, `requirements.txt`
- Notes: valuable for mechanistic insight; some experiments require Gurobi and WandB.

## Repo 4: Only Connect Wall (OCW)
- URL: https://github.com/TaatiTeam/OCW
- Commit: `4b125e451d8bd6ca3afb820202de184e490c1203`
- Purpose: dataset and baselines for creative problem solving with distractor-rich association tasks
- Location: `code/ocw/`
- Key files: `src/ocw/evaluate_only_connect.py`, `download_OCW.sh`, `scripts/randomize_test_set.py`
- Notes: includes instructions for the base dataset and two ablation datasets; benchmark data were downloaded into `datasets/ocw/`.

## Repo 5: Transformers Don't Need LayerNorm at Inference Time
- URL: https://github.com/submarat/removing-layer-norm
- Commit: `583a9e63ff639f17ee514c79a857f8d5a16728ff`
- Purpose: fine-tune GPT-2-family models into LN-free variants and evaluate loss impact
- Location: `code/removing-layer-norm/`
- Key files: `train.py`, `config.py`, `eval_pile.py`, `eval_all.sh`
- Notes: high-value baseline for direct LN-removal ablations; compute requirements are substantial (80GB+ GPU recommended by authors).

## Recommended Reuse Order

1. `code/novelty-bench/` for diversity-aware generation evaluation.
2. `code/divergent-association-task/` for DAT-style semantic-distance scoring.
3. `code/removing-layer-norm/` for LN-free model construction or adaptation ideas.
4. `code/ocw/` for associative creative problem-solving evaluation with red herrings.
5. `code/layer_norm_expressivity_role/` for mechanistic interpretation and hypothesis framing.
