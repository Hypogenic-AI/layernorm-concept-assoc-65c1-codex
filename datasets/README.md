# Downloaded Datasets

This directory contains local benchmark data for experiments on output diversity, concept association, and creative problem solving. Large data files are intentionally ignored by git through `datasets/.gitignore`.

## Dataset 1: NoveltyBench

### Overview
- Source: `novelty-bench/novelty-bench` repository data files
- Size: 2,300 prompt/annotation records across packaged splits
- Format: JSONL
- Task: diversity-aware open-ended generation
- Splits: `train` (1000), `val` (100), `curated` (100), `wildchat-1k` (1000), `humans` (100)
- License: repository is MIT licensed

### Local Location
- `datasets/novelty_bench/`

### Download Instructions

Already materialized locally by copying the benchmark data bundled in the cloned repo:

```bash
mkdir -p datasets/novelty_bench
cp code/novelty-bench/data/*.jsonl datasets/novelty_bench/
cp code/novelty-bench/data/readme.txt datasets/novelty_bench/
```

### Loading the Dataset

```python
import json
from pathlib import Path

rows = []
with Path("datasets/novelty_bench/curated.jsonl").open() as f:
    for line in f:
        rows.append(json.loads(line))
```

### Sample Data

Small samples are stored in:
- `datasets/novelty_bench/samples/curated_sample.json`
- `datasets/novelty_bench/samples/wildchat-1k_sample.json`
- `datasets/novelty_bench/samples/train_sample.json`

### Notes
- Best fit for directly testing the hypothesis about collapse toward safe, human-intuitive outputs.
- Metrics from the paper include `distinct_k` and `utility_k`, both worth reusing in experiments.
- The cloned repo in `code/novelty-bench/` contains scoring and partitioning code.

## Dataset 2: Only Connect Wall (OCW)

### Overview
- Source: `https://www.cs.toronto.edu/~taati/OCW/OCW.tar.gz`
- Size: 618 walls total
- Format: JSON
- Task: creative concept grouping and connection naming with deliberate red herrings
- Splits: `train` (62), `validation` (62), `test` (494), plus combined `OCW` (618)
- License: see repository and dataset docs in `code/ocw/` and `datasets/ocw/dataset/README.md`

### Local Location
- `datasets/ocw/dataset/`

### Download Instructions

```bash
mkdir -p datasets/ocw
cd datasets/ocw
curl -L https://www.cs.toronto.edu/~taati/OCW/OCW.tar.gz -o OCW.tar.gz
tar -xf OCW.tar.gz
rm -f OCW.tar.gz
```

### Loading the Dataset

```python
import json
from pathlib import Path

data = json.loads(Path("datasets/ocw/dataset/test.json").read_text())
rows = data["dataset"]
```

### Sample Data

Small samples are stored in:
- `datasets/ocw/samples/train_sample.json`
- `datasets/ocw/samples/validation_sample.json`
- `datasets/ocw/samples/test_sample.json`

### Notes
- Strong fit for evaluating associative creativity under distractors and fixation-style failure modes.
- The companion repo in `code/ocw/` includes evaluation scripts and additional ablation datasets (`OCW-Randomized`, `OCW-WordNet`).
- The stale Hugging Face loader was not used because the current `datasets` package rejects its legacy dataset script.

## Additional Benchmark Assets

### Divergent Association Task (DAT)
- Code and task assets are in `code/divergent-association-task/`.
- `words.txt` and `dat.py` are sufficient to score generated word sets once a compatible embedding model is added.
- The README points to open participant data at `https://osf.io/kbeq6/`, but that dataset was not pulled automatically here.

## Quick Validation

- `datasets/novelty_bench/` size: about 2.7 MB
- `datasets/ocw/` size: about 632 KB
- Sample files were saved for both datasets to support quick inspection without loading full corpora
