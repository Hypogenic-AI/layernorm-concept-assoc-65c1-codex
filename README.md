# LayerNorm and Concept Association in Creative Tasks

This workspace runs a direct comparison between vanilla GPT-2 and an LN-free GPT-2 checkpoint to test whether LayerNorm contributes to creative-output collapse and more homogenized concept-association geometry. The study uses downloaded local models, the NoveltyBench creativity subset, and the Only Connect Wall validation split.

## Key Findings
- LN removal did **not** improve lexical diversity on 20 NoveltyBench creativity prompts: `distinct_1` fell from `0.463` to `0.451`, and `distinct_2` fell from `0.899` to `0.884`.
- LN removal **did** increase semantic spread across generations: embedding dispersion rose from `0.803` to `0.831` with FDR-corrected `p=0.024`.
- LN removal strongly changed hidden-state geometry on OCW clues: anisotropy dropped from `0.992` to `0.964` and effective rank rose from `1.35` to `2.56`, both significant after FDR correction.
- Association separability improved only slightly and not significantly: OCW AUC moved from `0.527` to `0.529`.
- The practical conclusion is mixed: LayerNorm appears to compress representation geometry, but removing it alone does not reliably unlock more varied lexical creativity in this GPT-2 setting.

## Reproduce
Use the isolated workspace environment that already exists in `.venv/`.

```bash
source .venv/bin/activate
python src/run_layernorm_study.py \
  --max-prompts 20 \
  --num-generations 12 \
  --max-walls 62 \
  --batch-size 12 \
  --results-dir results/layernorm_study \
  --figures-dir figures/layernorm_study
```

## File Structure
- `planning.md`: planning, novelty assessment, and analysis plan
- `src/run_layernorm_study.py`: end-to-end experiment runner
- `results/layernorm_study/`: raw generations, CSV metrics, statistical tests, metadata
- `figures/layernorm_study/`: summary plots for generation and geometry metrics
- `REPORT.md`: full research report

## Full Report
See [REPORT.md](REPORT.md) for methodology, tables, statistical tests, limitations, and next steps.
