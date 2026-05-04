#!/usr/bin/env python
"""Run a local LayerNorm-vs-LN-free creativity and association study."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from transformers import AutoModelForCausalLM, AutoTokenizer


WORD_NUMBER_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


@dataclass
class ModelConfig:
    label: str
    hf_name: str


MODELS = [
    ModelConfig(label="gpt2", hf_name="gpt2"),
    ModelConfig(label="gpt2_lnfree", hf_name="schaeff/gpt2-small_LNFree300"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-generations", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-prompts", type=int, default=20)
    parser.add_argument("--ocw-split", choices=["validation", "test"], default="validation")
    parser.add_argument("--max-walls", type=int, default=62)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--results-dir", default="results/layernorm_study")
    parser.add_argument("--figures-dir", default="figures/layernorm_study")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def detect_hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": os.sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        info["gpus"] = []
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            info["gpus"].append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                }
            )
    return info


def load_creativity_prompts(path: Path, max_prompts: int) -> list[dict[str, Any]]:
    prompts = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["category"] == "Creativity":
                prompts.append(record)
    return prompts[:max_prompts]


def load_ocw_walls(path: Path, max_walls: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    return payload["dataset"][:max_walls]


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1)]


def distinct_n(responses: list[str], n: int) -> float:
    all_ngrams = []
    for response in responses:
        all_ngrams.extend(ngrams(words(response), n))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def pairwise_bigram_jaccard(responses: list[str]) -> float:
    scores = []
    bigram_sets = [set(ngrams(words(text), 2)) for text in responses]
    for i in range(len(bigram_sets)):
        for j in range(i + 1, len(bigram_sets)):
            union = bigram_sets[i] | bigram_sets[j]
            if not union:
                scores.append(0.0)
                continue
            inter = bigram_sets[i] & bigram_sets[j]
            scores.append(len(inter) / len(union))
    return float(np.mean(scores)) if scores else 0.0


def repetition_rate(text: str, n: int = 3) -> float:
    text_ngrams = ngrams(words(text), n)
    if not text_ngrams:
        return 0.0
    return 1.0 - (len(set(text_ngrams)) / len(text_ngrams))


def sentence_count(text: str) -> int:
    parts = [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]
    return len(parts)


def expected_sentence_count(prompt: str) -> int | None:
    match = re.search(r"\b(\d+)\s+sentences?\b", prompt.lower())
    if match:
        return int(match.group(1))
    for word, number in WORD_NUMBER_MAP.items():
        if re.search(rf"\b{word}\s+sentences?\b", prompt.lower()):
            return number
    return None


def adherence_score(prompt: str, responses: list[str]) -> float:
    expected = expected_sentence_count(prompt)
    if expected is None:
        return float("nan")
    counts = [sentence_count(response) for response in responses]
    return float(np.mean([count == expected for count in counts]))


def mean_pairwise_cosine_distance(vectors: np.ndarray) -> float:
    if len(vectors) < 2:
        return 0.0
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-8)
    sims = vectors @ vectors.T
    tri = sims[np.triu_indices_from(sims, k=1)]
    return float(np.mean(1.0 - tri))


def anisotropy(vectors: np.ndarray) -> float:
    if len(vectors) < 2:
        return 0.0
    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-8)
    sims = normalized @ normalized.T
    tri = sims[np.triu_indices_from(sims, k=1)]
    return float(np.mean(tri))


def effective_rank(vectors: np.ndarray) -> float:
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    if centered.shape[0] < 2:
        return 1.0
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-12]
    if len(eigvals) == 0:
        return 1.0
    return float((eigvals.sum() ** 2) / np.square(eigvals).sum())


def bootstrap_ci(values: list[float], seed: int, n_boot: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    samples = []
    for _ in range(n_boot):
        draw = rng.choice(arr, size=len(arr), replace=True)
        samples.append(draw.mean())
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def paired_effect_size(diff: np.ndarray) -> float:
    std = diff.std(ddof=1)
    if std < 1e-12:
        return 0.0
    return float(diff.mean() / std)


def wilcoxon_effect_size(diff: np.ndarray) -> float:
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    pos = ranks[nonzero > 0].sum()
    neg = ranks[nonzero < 0].sum()
    return float((pos - neg) / (pos + neg))


def paired_test(df: pd.DataFrame, metric: str) -> dict[str, Any]:
    pivot = df.pivot(index="unit_id", columns="model", values=metric).dropna()
    baseline = pivot["gpt2"].to_numpy(dtype=float)
    variant = pivot["gpt2_lnfree"].to_numpy(dtype=float)
    diff = variant - baseline
    if np.allclose(diff, 0):
        return {
            "metric": metric,
            "n": int(len(diff)),
            "test": "constant",
            "statistic": 0.0,
            "p_value": 1.0,
            "effect_size": 0.0,
            "baseline_mean": float(baseline.mean()),
            "variant_mean": float(variant.mean()),
            "delta_mean": 0.0,
        }
    shapiro_p = stats.shapiro(diff).pvalue if 3 <= len(diff) <= 5000 else 0.0
    normal = bool(shapiro_p > 0.05)
    if normal:
        result = stats.ttest_rel(variant, baseline)
        test_name = "paired_t"
        effect = paired_effect_size(diff)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    else:
        result = stats.wilcoxon(variant, baseline, zero_method="wilcox", alternative="two-sided")
        test_name = "wilcoxon"
        effect = wilcoxon_effect_size(diff)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    return {
        "metric": metric,
        "n": int(len(diff)),
        "test": test_name,
        "shapiro_p": float(shapiro_p),
        "statistic": statistic,
        "p_value": p_value,
        "effect_size": float(effect),
        "baseline_mean": float(baseline.mean()),
        "variant_mean": float(variant.mean()),
        "delta_mean": float(diff.mean()),
    }


def build_generation_metrics(
    prompt_record: dict[str, Any],
    responses: list[str],
    embedder: SentenceTransformer,
) -> dict[str, Any]:
    response_embeddings = embedder.encode(
        responses,
        convert_to_numpy=True,
        normalize_embeddings=False,
        batch_size=min(16, len(responses)),
        show_progress_bar=False,
    )
    return {
        "unit_id": prompt_record["id"],
        "category": prompt_record["category"],
        "prompt": prompt_record["prompt"],
        "distinct_1": distinct_n(responses, 1),
        "distinct_2": distinct_n(responses, 2),
        "embedding_dispersion": mean_pairwise_cosine_distance(response_embeddings),
        "pairwise_bigram_jaccard": pairwise_bigram_jaccard(responses),
        "mean_repetition_rate": float(np.mean([repetition_rate(text) for text in responses])),
        "mean_response_words": float(np.mean([len(words(text)) for text in responses])),
        "sentence_adherence": adherence_score(prompt_record["prompt"], responses),
    }


def encode_texts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    embeddings = []
    device = model.device
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states[-1]
        mask = encoded["attention_mask"].unsqueeze(-1)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        pooled = summed / denom
        embeddings.append(pooled.detach().float().cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def build_ocw_metrics(wall: dict[str, Any], clue_embeddings: np.ndarray) -> dict[str, Any]:
    normalized = clue_embeddings / np.linalg.norm(clue_embeddings, axis=1, keepdims=True).clip(min=1e-8)
    similarity = normalized @ normalized.T

    group_lookup: dict[str, int] = {}
    for group_index, group in enumerate(wall["groups"].values()):
        for clue in group["gt_words"]:
            group_lookup[clue.lower()] = group_index

    labels = []
    sims = []
    clues = wall["words"]
    for i in range(len(clues)):
        for j in range(i + 1, len(clues)):
            labels.append(int(group_lookup[clues[i].lower()] == group_lookup[clues[j].lower()]))
            sims.append(float(similarity[i, j]))

    labels_arr = np.asarray(labels)
    sims_arr = np.asarray(sims)
    within = sims_arr[labels_arr == 1]
    between = sims_arr[labels_arr == 0]
    auc = roc_auc_score(labels_arr, sims_arr)
    return {
        "unit_id": wall["wall_id"],
        "within_group_similarity": float(within.mean()),
        "between_group_similarity": float(between.mean()),
        "similarity_gap": float(within.mean() - between.mean()),
        "association_auc": float(auc),
        "anisotropy": anisotropy(clue_embeddings),
        "effective_rank": effective_rank(clue_embeddings),
    }


def load_model_and_tokenizer(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    probe = tokenizer("tokenizer health check")
    if getattr(tokenizer, "vocab_size", 0) == 0 or len(probe.get("input_ids", [])) == 0:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = encoded["input_ids"].repeat(num_generations, 1)
    attention_mask = encoded["attention_mask"].repeat(num_generations, 1)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[:, input_ids.shape[1] :]
    texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [text.strip() for text in texts]


def run_generation_experiment(
    model_cfg: ModelConfig,
    prompts: list[dict[str, Any]],
    args: argparse.Namespace,
    embedder: SentenceTransformer,
    raw_dir: Path,
) -> pd.DataFrame:
    model, tokenizer = load_model_and_tokenizer(model_cfg.hf_name)
    records = []
    output_path = raw_dir / f"{model_cfg.label}_generations.jsonl"
    with output_path.open("w") as handle:
        for prompt_record in prompts:
            responses = generate_responses(
                model,
                tokenizer,
                prompt_record["prompt"],
                args.num_generations,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
            )
            handle.write(
                json.dumps(
                    {
                        "model": model_cfg.label,
                        "prompt_id": prompt_record["id"],
                        "prompt": prompt_record["prompt"],
                        "responses": responses,
                    }
                )
                + "\n"
            )
            metrics = build_generation_metrics(prompt_record, responses, embedder)
            metrics["model"] = model_cfg.label
            records.append(metrics)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(records)


def run_ocw_experiment(
    model_cfg: ModelConfig,
    walls: list[dict[str, Any]],
    args: argparse.Namespace,
    raw_dir: Path,
) -> pd.DataFrame:
    model, tokenizer = load_model_and_tokenizer(model_cfg.hf_name)
    records = []
    output_path = raw_dir / f"{model_cfg.label}_ocw_embeddings.jsonl"
    with output_path.open("w") as handle:
        for wall in walls:
            clues = wall["words"]
            clue_embeddings = encode_texts(model, tokenizer, clues, args.batch_size)
            handle.write(
                json.dumps(
                    {
                        "model": model_cfg.label,
                        "wall_id": wall["wall_id"],
                        "clues": clues,
                        "embedding_shape": list(clue_embeddings.shape),
                    }
                )
                + "\n"
            )
            metrics = build_ocw_metrics(wall, clue_embeddings)
            metrics["model"] = model_cfg.label
            records.append(metrics)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(records)


def summarize_metrics(df: pd.DataFrame, metrics: list[str], seed: int) -> list[dict[str, Any]]:
    summary = []
    for metric in metrics:
        for model_name, group in df.groupby("model"):
            values = group[metric].dropna().astype(float).tolist()
            ci_low, ci_high = bootstrap_ci(values, seed)
            summary.append(
                {
                    "metric": metric,
                    "model": model_name,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n": len(values),
                }
            )
    return summary


def compare_metric_family(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    results = [paired_test(df, metric) for metric in metrics]
    p_values = [result["p_value"] for result in results]
    reject, corrected, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
    for result, corr_p, rej in zip(results, corrected, reject):
        result["p_value_fdr"] = float(corr_p)
        result["reject_fdr_0_05"] = bool(rej)
    return pd.DataFrame(results)


def plot_metric_family(
    df: pd.DataFrame,
    metrics: list[str],
    title: str,
    output_path: Path,
) -> None:
    plot_df = df.melt(id_vars=["model", "unit_id"], value_vars=metrics, var_name="metric", value_name="value")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4.5), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]
    for axis, metric in zip(axes, metrics):
        subset = plot_df[plot_df["metric"] == metric]
        sns.boxplot(data=subset, x="model", y="value", ax=axis)
        sns.stripplot(data=subset, x="model", y="value", ax=axis, color="black", size=3, alpha=0.5)
        axis.set_title(metric)
        axis.set_xlabel("")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    start_time = time.time()
    results_dir = ensure_dir(args.results_dir)
    figures_dir = ensure_dir(args.figures_dir)
    raw_dir = ensure_dir(results_dir / "raw")

    prompts = load_creativity_prompts(Path("datasets/novelty_bench/curated.jsonl"), args.max_prompts)
    walls = load_ocw_walls(Path(f"datasets/ocw/dataset/{args.ocw_split}.json"), args.max_walls)

    embedder = SentenceTransformer(
        args.embedder,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    generation_frames = []
    ocw_frames = []
    for model_cfg in MODELS:
        generation_frames.append(run_generation_experiment(model_cfg, prompts, args, embedder, raw_dir))
        ocw_frames.append(run_ocw_experiment(model_cfg, walls, args, raw_dir))

    generation_df = pd.concat(generation_frames, ignore_index=True)
    ocw_df = pd.concat(ocw_frames, ignore_index=True)

    generation_metrics = [
        "distinct_1",
        "distinct_2",
        "embedding_dispersion",
        "pairwise_bigram_jaccard",
        "mean_repetition_rate",
        "sentence_adherence",
    ]
    ocw_metrics = [
        "association_auc",
        "similarity_gap",
        "anisotropy",
        "effective_rank",
    ]

    generation_summary = summarize_metrics(generation_df, generation_metrics, args.seed)
    ocw_summary = summarize_metrics(ocw_df, ocw_metrics, args.seed)
    generation_tests = compare_metric_family(generation_df, generation_metrics)
    ocw_tests = compare_metric_family(ocw_df, ocw_metrics)

    generation_df.to_csv(results_dir / "generation_metrics.csv", index=False)
    ocw_df.to_csv(results_dir / "ocw_metrics.csv", index=False)
    generation_tests.to_csv(results_dir / "generation_stats.csv", index=False)
    ocw_tests.to_csv(results_dir / "ocw_stats.csv", index=False)
    save_json(results_dir / "generation_summary.json", generation_summary)
    save_json(results_dir / "ocw_summary.json", ocw_summary)

    plot_metric_family(generation_df, generation_metrics, "Creative Generation Metrics", figures_dir / "generation_metrics.png")
    plot_metric_family(ocw_df, ocw_metrics, "OCW Association Geometry Metrics", figures_dir / "ocw_metrics.png")

    metadata = {
        "args": vars(args),
        "models": [asdict(model_cfg) for model_cfg in MODELS],
        "hardware": detect_hardware(),
        "prompt_count": len(prompts),
        "wall_count": len(walls),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    save_json(results_dir / "metadata.json", metadata)

    print(json.dumps(metadata, indent=2))
    print("Saved:", results_dir)
    print("Figures:", figures_dir)


if __name__ == "__main__":
    main()
