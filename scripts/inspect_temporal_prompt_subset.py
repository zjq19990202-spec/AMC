#!/usr/bin/env python3
"""List the actual prompts used by a selected temporal zT comparison."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import build_atomic_text_dataset


def _load_plot_helpers():
    path = Path(__file__).with_name("plot_zt_scale_temporal_tsne.py")
    spec = importlib.util.spec_from_file_location("zt_plot_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", action="append", required=True)
    parser.add_argument("--norm-assets-dir", default="/mnt/cunchu/zjq/target")
    parser.add_argument("--norm-asset-id", default="openpi_norm_compact_accepted_v3")
    parser.add_argument("--scan-samples", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    helpers = _load_plot_helpers()
    config = AtomicPi05Config(max_token_len=250, fast_action_ce_loss_weight=0.0)
    dataset = build_atomic_text_dataset(
        tuple(args.dataset_root),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        include_fast=False,
    )
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(
        len(dataset), size=min(args.scan_samples, len(dataset)), replace=False
    )
    by_pair = defaultdict(list)
    for index in indices:
        row = dataset[int(index)]
        labels = helpers._labels(row)
        if len(labels) != 2:
            continue
        temporal = helpers._temporal_class(row, labels)
        if temporal == "unclear":
            continue
        by_pair[labels].append(
            {
                "index": int(index),
                "row": row,
                "prompt": str(row["atomic_prompt"]),
                "labels": labels,
                "temporal": temporal,
                "amplitude": helpers._amplitude(row, labels),
            }
        )

    candidates = []
    for labels, rows in by_pair.items():
        neighborhood = helpers._best_prompt_neighborhood(
            rows,
            threshold=0.30,
            minimum_rows=10,
            require_temporal_diversity=True,
        )
        if neighborhood is None:
            continue
        reference, selected, mean_similarity = neighborhood
        counts = Counter(row["temporal"] for row in selected)
        valid = {name: count for name, count in counts.items() if count >= 3}
        if len(valid) < 2:
            continue
        candidates.append(
            (sum(valid.values()), labels, reference, selected, mean_similarity)
        )
    candidates.sort(reverse=True, key=lambda item: item[0])

    output = []
    for _, labels, reference, selected, mean_similarity in candidates[:3]:
        unique_prompts = sorted({row["prompt"] for row in selected})
        matrix = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1
        ).fit_transform([reference, *unique_prompts])
        similarities = np.asarray((matrix[0] @ matrix[1:].T).toarray()).reshape(-1)
        similarity_by_prompt = dict(zip(unique_prompts, similarities, strict=True))
        category_payload = {}
        for category in ("simultaneous", "A_then_B", "B_then_A"):
            prompt_counts = Counter(
                row["prompt"] for row in selected if row["temporal"] == category
            )
            if not prompt_counts:
                continue
            category_payload[category] = [
                {
                    "prompt": prompt,
                    "sample_count": count,
                    "tfidf_similarity_to_reference": round(
                        float(similarity_by_prompt[prompt]), 4
                    ),
                }
                for prompt, count in prompt_counts.most_common()
            ]
        output.append(
            {
                "labels": list(labels),
                "reference_prompt": reference,
                "mean_selected_prompt_similarity": mean_similarity,
                "categories": category_payload,
            }
        )
    report = {"scan_samples": len(indices), "groups": output}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
