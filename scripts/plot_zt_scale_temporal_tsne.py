#!/usr/bin/env python3
"""Visualize zT atomic clusters, scale variation, temporal organization, and drop rows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from openpi.models import model as _model

from atomic_latent_vla.atomic import ATOMIC_NAMES
from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import (
    atomic_text_collate,
    build_atomic_text_dataset,
    text_batch_to_observation,
)


@nnx.jit
def _encode(model, observation):
    query_hidden, _, _ = model._prefix_forward(observation)  # noqa: SLF001
    state = model._controlled_state(observation.state)  # noqa: SLF001
    raw_zt = model.queries.text_latent(query_hidden, state)
    direction = model.queries.direction(raw_zt)
    direction = direction / jnp.maximum(jnp.linalg.norm(direction, axis=-1, keepdims=True), 1e-8)
    codes = model.codebook.value
    codes = codes / jnp.maximum(jnp.linalg.norm(codes, axis=-1, keepdims=True), 1e-8)
    return direction, direction @ codes.T


def _labels(row: dict[str, Any]) -> tuple[int, ...]:
    if not bool(row["atomic_supervision_mask"]):
        return ()
    return tuple(int(value) for value in np.flatnonzero(np.asarray(row["atomic_weights"]) > 0))


def _motion_profiles(row: dict[str, Any], labels: tuple[int, ...]) -> tuple[np.ndarray, list[float]]:
    delta = np.asarray(row["tcp_twist_delta"], dtype=np.float64)
    increments = np.diff(
        np.concatenate([np.zeros((1, delta.shape[-1])), delta], axis=0), axis=0
    )
    profiles, physical = [], []
    for label in labels:
        axis = label // 2
        sign = 1.0 if label % 2 == 0 else -1.0
        profile = np.maximum(sign * increments[:, axis], 0.0)
        physical.append(float(profile.sum()))
        profiles.append(profile)
    return np.asarray(profiles), physical


def _amplitude(row: dict[str, Any], labels: tuple[int, ...]) -> float:
    _, masses = _motion_profiles(row, labels)
    scaled = [
        value / (0.02 if label // 2 < 3 else 0.15)
        for label, value in zip(labels, masses, strict=True)
    ]
    return float(np.linalg.norm(scaled))


def _temporal_class(row: dict[str, Any], labels: tuple[int, ...]) -> str:
    if len(labels) != 2:
        return "not_dual"
    profiles, _ = _motion_profiles(row, labels)
    masses = profiles.sum(axis=1)
    if np.any(masses <= 1e-8):
        return "unclear"
    profiles = profiles / masses[:, None]
    grid = np.linspace(0.0, 1.0, profiles.shape[1])
    centers = profiles @ grid
    overlap = float(np.minimum(profiles[0], profiles[1]).sum())
    gap = float(centers[1] - centers[0])
    if abs(gap) <= 0.10 and overlap >= 0.25:
        return "simultaneous"
    if gap >= 0.18:
        return "A_then_B"
    if gap <= -0.18:
        return "B_then_A"
    return "unclear"


def _tangent_vectors(vectors: np.ndarray) -> np.ndarray:
    """Riemannian log map from the unit sphere to the sample mean tangent plane."""
    center = vectors.mean(axis=0)
    center /= max(float(np.linalg.norm(center)), 1e-8)
    cosine = np.clip(vectors @ center, -1.0, 1.0)
    theta = np.arccos(cosine)
    residual = vectors - cosine[:, None] * center[None]
    scale = np.where(theta > 1e-7, theta / np.maximum(np.sin(theta), 1e-8), 1.0)
    return residual * scale[:, None]


def _run_tsne(vectors: np.ndarray, seed: int) -> np.ndarray:
    if len(vectors) < 4:
        raise ValueError("t-SNE needs at least four rows")
    if vectors.shape[1] > 50 and len(vectors) > 55:
        vectors = PCA(n_components=min(50, len(vectors) - 1), random_state=seed).fit_transform(vectors)
    perplexity = min(40.0, max(3.0, (len(vectors) - 1) / 4.0))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        max_iter=1500,
        random_state=seed,
    ).fit_transform(vectors)


def _best_prompt_neighborhood(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    minimum_rows: int,
    require_temporal_diversity: bool = False,
) -> tuple[str, list[dict[str, Any]], float] | None:
    """Find a dense lexical-semantic prompt neighborhood without using zT itself."""

    if len(rows) < minimum_rows:
        return None
    prompts = [row["prompt"] for row in rows]
    matrix = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1
    ).fit_transform(prompts)
    best = None
    for index in range(len(rows)):
        similarity = np.asarray((matrix[index] @ matrix.T).toarray()).reshape(-1)
        hit = similarity >= threshold
        neighborhood = [row for row, keep in zip(rows, hit, strict=True) if keep]
        if len(neighborhood) < minimum_rows:
            continue
        if require_temporal_diversity:
            counts = {
                name: sum(row.get("temporal") == name for row in neighborhood)
                for name in ("simultaneous", "A_then_B", "B_then_A")
            }
            if sum(value >= 3 for value in counts.values()) < 2:
                continue
            score = min(sorted((value for value in counts.values() if value >= 3), reverse=True)[:2])
        else:
            amplitudes = np.asarray([row["amplitude"] for row in neighborhood])
            active = amplitudes >= 0.20
            if active.sum() < minimum_rows:
                continue
            q10, q90 = np.quantile(amplitudes[active], [0.10, 0.90])
            if q90 / max(q10, 1e-8) < 2.0:
                continue
            score = active.sum() * np.log(q90 / max(q10, 1e-8))
        mean_similarity = float(similarity[hit].mean())
        candidate = (float(score), prompts[index], neighborhood, mean_similarity)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return None
    return best[1], best[2], best[3]


def _plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#fbfbfc",
            "axes.facecolor": "#fbfbfc",
            "axes.edgecolor": "#9ca3af",
            "axes.labelcolor": "#374151",
            "xtick.color": "#6b7280",
            "ytick.color": "#6b7280",
            "text.color": "#111827",
            "font.size": 9,
            "axes.titleweight": "semibold",
        }
    )


def _clean_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(color="#e5e7eb", linewidth=0.6, alpha=0.7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", action="append", required=True)
    parser.add_argument("--norm-assets-dir", default="/mnt/cunchu/zjq/target")
    parser.add_argument("--norm-asset-id", default="openpi_norm_compact_accepted_v3")
    parser.add_argument("--scan-samples", type=int, default=40000)
    parser.add_argument("--per-atom", type=int, default=120)
    parser.add_argument("--drop-samples", type=int, default=700)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    scan_indices = rng.choice(
        len(dataset), size=min(args.scan_samples, len(dataset)), replace=False
    )

    records: list[dict[str, Any]] = []
    scale_groups: dict[tuple[str, tuple[int, ...]], list[dict[str, Any]]] = defaultdict(list)
    rows_by_labels: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    temporal_groups: dict[tuple[str, tuple[int, int]], dict[str, list[dict[str, Any]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    singles: dict[int, list[dict[str, Any]]] = defaultdict(list)
    duals: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    drops: list[dict[str, Any]] = []
    for index in scan_indices:
        row = dataset[int(index)]
        labels = _labels(row)
        record = {
            "index": int(index),
            "row": row,
            "prompt": str(row["atomic_prompt"]),
            "labels": labels,
            "amplitude": _amplitude(row, labels) if labels else 0.0,
        }
        records.append(record)
        if len(labels) == 1:
            singles[labels[0]].append(record)
        elif len(labels) == 2:
            duals[labels].append(record)
        elif not labels:
            drops.append(record)
        if labels:
            scale_groups[(record["prompt"], labels)].append(record)
            rows_by_labels[labels].append(record)
        if len(labels) == 2:
            temporal = _temporal_class(row, labels)
            record["temporal"] = temporal
            if temporal != "unclear":
                temporal_groups[(record["prompt"], labels)][temporal].append(record)

    # Scale panels: same labels and a dense similar-prompt neighborhood.
    scale_candidates = []
    for labels, rows in rows_by_labels.items():
        neighborhood = _best_prompt_neighborhood(
            rows, threshold=0.40, minimum_rows=15
        )
        if neighborhood is None:
            continue
        representative_prompt, active, mean_similarity = neighborhood
        values = np.asarray([row["amplitude"] for row in active if row["amplitude"] >= 0.20])
        q10, q90 = np.quantile(values, [0.10, 0.90])
        score = len(active) * np.log(q90 / max(q10, 1e-8))
        scale_candidates.append(
            (score, (representative_prompt, labels), active, mean_similarity)
        )
    scale_candidates.sort(reverse=True, key=lambda item: item[0])
    chosen_scale = []
    used_label_sets = set()
    # Prefer three singles and two duals, all with different atomic identities.
    for desired_count in (1, 2):
        for item in scale_candidates:
            labels = item[1][1]
            if len(labels) != desired_count or labels in used_label_sets:
                continue
            chosen_scale.append(item)
            used_label_sets.add(labels)
            if sum(len(value[1][1]) == desired_count for value in chosen_scale) >= (
                3 if desired_count == 1 else 2
            ):
                break

    # Temporal panels: same pair and a similar-prompt neighborhood with >=2 organizations.
    temporal_candidates = []
    dual_rows_by_labels: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if len(record["labels"]) == 2 and record.get("temporal") not in {None, "unclear"}:
            dual_rows_by_labels[record["labels"]].append(record)
    for labels, rows in dual_rows_by_labels.items():
        neighborhood = _best_prompt_neighborhood(
            rows, threshold=0.30, minimum_rows=10, require_temporal_diversity=True
        )
        if neighborhood is None:
            continue
        representative_prompt, selected_rows, mean_similarity = neighborhood
        valid_categories = defaultdict(list)
        for row in selected_rows:
            valid_categories[row["temporal"]].append(row)
        valid_categories = {
            name: values for name, values in valid_categories.items() if len(values) >= 3
        }
        score = sum(len(values) for values in valid_categories.values())
        temporal_candidates.append(
            (score, (representative_prompt, labels), valid_categories, mean_similarity)
        )
    temporal_candidates.sort(reverse=True, key=lambda item: item[0])
    chosen_temporal = temporal_candidates[:3]

    # Global panel: balanced single-atom rows and a larger drop background.
    global_records = []
    global_labels = []
    global_pairs: list[tuple[int, int] | None] = []
    for label in range(12):
        rows = singles.get(label, [])
        if len(rows) > args.per_atom:
            rows = list(rng.choice(rows, size=args.per_atom, replace=False))
        global_records.extend(rows)
        global_labels.extend([label] * len(rows))
        global_pairs.extend([None] * len(rows))
    dual_records = []
    for pair, rows in sorted(duals.items()):
        if len(rows) > 20:
            rows = list(rng.choice(rows, size=20, replace=False))
        dual_records.extend(rows)
    if len(dual_records) > 600:
        dual_records = list(rng.choice(dual_records, size=600, replace=False))
    global_records.extend(dual_records)
    global_labels.extend([-2] * len(dual_records))
    global_pairs.extend([tuple(row["labels"]) for row in dual_records])
    if len(drops) > args.drop_samples:
        drops = list(rng.choice(drops, size=args.drop_samples, replace=False))
    global_records.extend(drops)
    global_labels.extend([-1] * len(drops))
    global_pairs.extend([None] * len(drops))

    selected = {row["index"]: row for row in global_records}
    for _, _, rows, _ in chosen_scale:
        # At most 100 ordered-by-amplitude rows per local panel.
        rows = sorted(rows, key=lambda row: row["amplitude"])
        if len(rows) > 100:
            positions = np.linspace(0, len(rows) - 1, 100).round().astype(int)
            rows = [rows[position] for position in positions]
        for row in rows:
            selected[row["index"]] = row
    for _, _, category_rows, _ in chosen_temporal:
        for rows in category_rows.values():
            for row in rows[:80]:
                selected[row["index"]] = row

    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    vector_by_index: dict[int, np.ndarray] = {}
    similarity_by_index: dict[int, np.ndarray] = {}
    selected_rows = list(selected.values())
    for start in range(0, len(selected_rows), 256):
        rows = selected_rows[start : start + 256]
        batch = atomic_text_collate([row["row"] for row in rows])
        observation = jax.tree.map(jnp.asarray, text_batch_to_observation(batch))
        vectors, similarities = map(np.asarray, jax.device_get(_encode(model, observation)))
        for row, vector, similarity in zip(rows, vectors, similarities, strict=True):
            vector_by_index[row["index"]] = vector.astype(np.float32)
            similarity_by_index[row["index"]] = similarity.astype(np.float32)

    _plot_style()
    palette = list(plt.get_cmap("tab20").colors[:12])
    manifest: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "scan_samples": len(scan_indices),
        "selection": {
            "single_counts": {ATOMIC_NAMES[i]: len(singles.get(i, [])) for i in range(12)},
            "drop_count": len([record for record in records if not record["labels"]]),
        },
        "scale_panels": [],
        "temporal_panels": [],
    }

    # Figure 1: local scale, paired PCA and t-SNE.
    if chosen_scale:
        figure, axes = plt.subplots(
            len(chosen_scale), 3, figsize=(18.5, 3.8 * len(chosen_scale)), constrained_layout=True
        )
        axes = np.atleast_2d(axes)
        for row_axis, (_, (prompt, labels), rows, mean_prompt_similarity) in enumerate(chosen_scale):
            rows = sorted(rows, key=lambda row: row["amplitude"])
            if len(rows) > 100:
                positions = np.linspace(0, len(rows) - 1, 100).round().astype(int)
                rows = [rows[position] for position in positions]
            vectors = np.stack([vector_by_index[row["index"]] for row in rows])
            tangent = _tangent_vectors(vectors)
            pca_model = PCA(n_components=2, random_state=args.seed).fit(tangent)
            pca_xy = pca_model.transform(tangent)
            amplitude = np.asarray([row["amplitude"] for row in rows])
            color_value = np.log1p(amplitude)
            rho, pvalue = spearmanr(pca_xy[:, 0], amplitude)
            if rho < 0:
                pca_xy[:, 0] *= -1
                rho = -rho
            tsne_xy = _run_tsne(tangent, args.seed + row_axis)
            title = " + ".join(ATOMIC_NAMES[label] for label in labels)
            low_cut = np.quantile(amplitude, 0.25)
            reference = vectors[amplitude <= low_cut].mean(axis=0)
            reference /= max(float(np.linalg.norm(reference)), 1e-8)
            angle_to_low = np.degrees(
                np.arccos(np.clip(vectors @ reference, -1.0, 1.0))
            )
            scale_rho, scale_pvalue = spearmanr(amplitude, angle_to_low)
            axis = axes[row_axis, 0]
            marks = axis.scatter(
                pca_xy[:, 0], pca_xy[:, 1], c=color_value, cmap="viridis", s=27,
                alpha=0.82, edgecolors="white", linewidths=0.25,
            )
            axis.set_title(
                f"{title} — Tangent PCA\n"
                f"n={len(rows)}, same matched subset",
                fontsize=11,
            )
            axis.set_xlabel("PC1")
            axis.set_ylabel("PC2")
            _clean_axis(axis)

            axis = axes[row_axis, 1]
            axis.scatter(
                amplitude, angle_to_low, c=color_value, cmap="viridis", s=27,
                alpha=0.82, edgecolors="white", linewidths=0.25,
            )
            order = np.argsort(amplitude)
            bins = np.array_split(order, min(6, max(2, len(rows) // 8)))
            bin_x = [float(np.median(amplitude[index])) for index in bins if len(index)]
            bin_y = [float(np.median(angle_to_low[index])) for index in bins if len(index)]
            axis.plot(bin_x, bin_y, color="#d97706", linewidth=1.6, marker="o",
                      markersize=3.5, label="binned median")
            axis.set_title(
                f"{title} — FK scale vs zT change\n"
                f"n={len(rows)}, similar-instruction subset",
                fontsize=11,
            )
            axis.set_xlabel("Scaled FK motion amplitude")
            axis.set_ylabel("zT angle from low-scale centroid (degrees)")
            axis.legend(frameon=False, fontsize=8)
            _clean_axis(axis)

            axis = axes[row_axis, 2]
            marks = axis.scatter(
                tsne_xy[:, 0], tsne_xy[:, 1], c=color_value, cmap="viridis", s=25,
                alpha=0.82, edgecolors="white", linewidths=0.25,
            )
            axis.set_title(
                f"{title} — Local t-SNE\n"
                f"n={len(rows)}, similar-instruction subset",
                fontsize=11,
            )
            axis.set_xlabel("t-SNE 1")
            axis.set_ylabel("t-SNE 2")
            _clean_axis(axis)
            colorbar = figure.colorbar(marks, ax=axis, shrink=0.82)
            colorbar.set_label("log(1 + scaled FK amplitude)")
            manifest["scale_panels"].append(
                {
                    "prompt": prompt,
                    "labels": [ATOMIC_NAMES[label] for label in labels],
                    "count": len(rows),
                    "amplitude_min": float(amplitude.min()),
                    "amplitude_max": float(amplitude.max()),
                    "pca_explained_variance": pca_model.explained_variance_ratio_.tolist(),
                    "pc1_amplitude_spearman": float(rho),
                    "pc1_amplitude_pvalue": float(pvalue),
                    "amplitude_vs_angle_from_low_scale_spearman": float(scale_rho),
                    "amplitude_vs_angle_from_low_scale_pvalue": float(scale_pvalue),
                    "mean_tfidf_prompt_similarity_to_reference": mean_prompt_similarity,
                }
            )
        figure.suptitle(
            "zT scale: tangent PCA, direct high-dimensional relation, and local t-SNE",
            fontsize=15, fontweight="bold",
        )
        scale_path = args.output_dir / "zt_scale_direct_tsne.png"
        figure.savefig(scale_path, dpi=190, bbox_inches="tight")
        plt.close(figure)
        manifest["scale_figure"] = str(scale_path)

    # Figure 2: local temporal organization, paired PCA and t-SNE.
    if chosen_temporal:
        temporal_colors = {
            "simultaneous": "#2563eb", "A_then_B": "#d97706", "B_then_A": "#be185d"
        }
        temporal_markers = {"simultaneous": "o", "A_then_B": "^", "B_then_A": "s"}
        figure, axes = plt.subplots(
            len(chosen_temporal), 3,
            figsize=(18.5, 4.0 * len(chosen_temporal)), constrained_layout=True,
        )
        axes = np.atleast_2d(axes)
        for row_axis, (_, (prompt, labels), category_rows, mean_prompt_similarity) in enumerate(
            chosen_temporal
        ):
            rows = []
            categories = []
            for category, values in category_rows.items():
                rows.extend(values[:80])
                categories.extend([category] * min(len(values), 80))
            vectors = np.stack([vector_by_index[row["index"]] for row in rows])
            tangent = _tangent_vectors(vectors)
            tsne_xy = _run_tsne(tangent, args.seed + 100 + row_axis)
            title = f"{ATOMIC_NAMES[labels[0]]} + {ATOMIC_NAMES[labels[1]]}"
            category_centers = {}
            for category in set(categories):
                hit = np.asarray([value == category for value in categories])
                center = vectors[hit].mean(axis=0)
                center /= max(float(np.linalg.norm(center)), 1e-8)
                category_centers[category] = center
            angles = {}
            names = sorted(category_centers)
            for i, first in enumerate(names):
                for second in names[i + 1 :]:
                    cosine = float(np.clip(category_centers[first] @ category_centers[second], -1, 1))
                    angles[f"{first}_vs_{second}"] = float(np.degrees(np.arccos(cosine)))

            center_order = [
                name for name in ("simultaneous", "A_then_B", "B_then_A")
                if name in category_centers
            ]
            angle_matrix = np.zeros((len(center_order), len(center_order)), dtype=np.float32)
            for first_index, first in enumerate(center_order):
                for second_index, second in enumerate(center_order):
                    cosine = float(
                        np.clip(category_centers[first] @ category_centers[second], -1, 1)
                    )
                    angle_matrix[first_index, second_index] = np.degrees(np.arccos(cosine))
            # Direct high-dimensional separation diagnostic.  The own-class
            # center is leave-one-out so the evaluated sample cannot make its
            # own angle artificially small.  "Other" is the nearest competing
            # category center, making this a conservative separation test.
            category_array = np.asarray(categories)
            own_angles_by_category = {}
            other_angles_by_category = {}
            margins_by_category = {}
            correct_by_category = {}
            for category in center_order:
                member_indices = np.flatnonzero(category_array == category)
                own_angles = []
                other_angles = []
                competitors = [
                    category_centers[name] for name in center_order if name != category
                ]
                for sample_index in member_indices:
                    leave_one_out = vectors[member_indices].sum(axis=0) - vectors[sample_index]
                    leave_one_out /= max(float(np.linalg.norm(leave_one_out)), 1e-8)
                    own_cosine = float(
                        np.clip(vectors[sample_index] @ leave_one_out, -1.0, 1.0)
                    )
                    own_angle = float(np.degrees(np.arccos(own_cosine)))
                    competitor_angles = [
                        float(
                            np.degrees(
                                np.arccos(
                                    np.clip(vectors[sample_index] @ center, -1.0, 1.0)
                                )
                            )
                        )
                        for center in competitors
                    ]
                    own_angles.append(own_angle)
                    other_angles.append(min(competitor_angles))
                own_angles = np.asarray(own_angles)
                other_angles = np.asarray(other_angles)
                own_angles_by_category[category] = own_angles
                other_angles_by_category[category] = other_angles
                margins_by_category[category] = other_angles - own_angles
                correct_by_category[category] = float(np.mean(own_angles < other_angles))

            axis = axes[row_axis, 0]
            rng = np.random.default_rng(args.seed + 700 + row_axis)
            positions = np.arange(len(center_order), dtype=np.float32)
            own_position = positions - 0.18
            other_position = positions + 0.18
            own_values = [own_angles_by_category[name] for name in center_order]
            other_values = [other_angles_by_category[name] for name in center_order]
            own_boxes = axis.boxplot(
                own_values, positions=own_position, widths=0.28, patch_artist=True,
                showfliers=False, manage_ticks=False,
            )
            other_boxes = axis.boxplot(
                other_values, positions=other_position, widths=0.28, patch_artist=True,
                showfliers=False, manage_ticks=False,
            )
            for box in own_boxes["boxes"]:
                box.set(facecolor="#93c5fd", edgecolor="#1d4ed8", linewidth=1.0)
            for box in other_boxes["boxes"]:
                box.set(facecolor="#fed7aa", edgecolor="#c2410c", linewidth=1.0)
            for collection, color in ((own_boxes, "#1d4ed8"), (other_boxes, "#c2410c")):
                for key in ("medians", "whiskers", "caps"):
                    for artist in collection[key]:
                        artist.set(color=color, linewidth=1.0)
            for category_index, category in enumerate(center_order):
                own = own_angles_by_category[category]
                other = other_angles_by_category[category]
                jitter = rng.uniform(-0.035, 0.035, size=len(own))
                axis.scatter(
                    own_position[category_index] + jitter, own, s=14,
                    color="#2563eb", alpha=0.35, edgecolors="none", zorder=3,
                )
                axis.scatter(
                    other_position[category_index] + jitter, other, s=14,
                    color="#ea580c", alpha=0.35, edgecolors="none", zorder=3,
                )
                axis.text(
                    positions[category_index], 0.98,
                    f"own closer {correct_by_category[category]:.0%}\n"
                    f"median margin {np.median(margins_by_category[category]):+.1f}°",
                    transform=axis.get_xaxis_transform(), ha="center", va="top",
                    fontsize=7.3, color="#374151",
                )
            axis.set_xticks(
                positions, [name.replace("_", " ") for name in center_order]
            )
            axis.set_ylabel("Angle to centroid (degrees)")
            axis.set_title(
                f"{title} — sample-to-centroid angles\n"
                "own = leave-one-out; other = nearest competing centroid",
                fontsize=11,
            )
            _clean_axis(axis)
            axis.legend(
                handles=[
                    Patch(facecolor="#93c5fd", edgecolor="#1d4ed8", label="own class"),
                    Patch(
                        facecolor="#fed7aa", edgecolor="#c2410c",
                        label="nearest other class",
                    ),
                ],
                frameon=False, fontsize=8, loc="lower right",
            )

            axis = axes[row_axis, 1]
            image = axis.imshow(
                angle_matrix, cmap="Blues", vmin=0,
                vmax=max(30.0, float(angle_matrix.max())), aspect="auto",
            )
            display_names = [name.replace("_", " ") for name in center_order]
            axis.set_xticks(range(len(center_order)), display_names, rotation=20, ha="right")
            axis.set_yticks(range(len(center_order)), display_names)
            for first_index in range(len(center_order)):
                for second_index in range(len(center_order)):
                    value = angle_matrix[first_index, second_index]
                    axis.text(
                        second_index, first_index, f"{value:.1f}°",
                        ha="center", va="center",
                        color="white" if value > 17 else "#111827", fontweight="bold",
                    )
            axis.set_title(
                f"{title} — zT centroid angles\n"
                f"same atom pair, similar instructions, state unrestricted",
                fontsize=11,
            )
            figure.colorbar(image, ax=axis, shrink=0.78, label="Angle (degrees)")

            axis = axes[row_axis, 2]
            for category in ("simultaneous", "A_then_B", "B_then_A"):
                hit = np.asarray([value == category for value in categories])
                if not np.any(hit):
                    continue
                axis.scatter(
                    tsne_xy[hit, 0], tsne_xy[hit, 1], s=34, alpha=0.82,
                    c=temporal_colors[category], marker=temporal_markers[category],
                    edgecolors="white", linewidths=0.35, label=category.replace("_", " "),
                )
            axis.set_title(
                f"{title} — Local t-SNE\n"
                f"n={len(rows)}, similar-instruction subset",
                fontsize=11,
            )
            axis.set_xlabel("t-SNE 1")
            axis.set_ylabel("t-SNE 2")
            _clean_axis(axis)
            axis.legend(frameon=False, fontsize=8)
            manifest["temporal_panels"].append(
                {
                    "prompt": prompt,
                    "labels": [ATOMIC_NAMES[label] for label in labels],
                    "counts": {name: categories.count(name) for name in set(categories)},
                    "centroid_angles_deg": angles,
                    "sample_to_centroid": {
                        name: {
                            "own_leave_one_out_median_deg": float(
                                np.median(own_angles_by_category[name])
                            ),
                            "nearest_other_median_deg": float(
                                np.median(other_angles_by_category[name])
                            ),
                            "margin_median_deg": float(
                                np.median(margins_by_category[name])
                            ),
                            "own_closer_fraction": correct_by_category[name],
                        }
                        for name in center_order
                    },
                    "mean_tfidf_prompt_similarity_to_reference": mean_prompt_similarity,
                }
            )
        figure.suptitle(
            "Dual-atom temporal organization: sample separation, centroid angles, and local t-SNE",
            fontsize=15, fontweight="bold",
        )
        temporal_path = args.output_dir / "zt_temporal_angles_tsne.png"
        figure.savefig(temporal_path, dpi=190, bbox_inches="tight")
        plt.close(figure)
        manifest["temporal_figure"] = str(temporal_path)

    # Figure 3: global 12 single-atom clusters plus drop.
    global_vectors = np.stack([vector_by_index[row["index"]] for row in global_records])
    global_labels_np = np.asarray(global_labels)
    global_xy = _run_tsne(global_vectors, args.seed + 999)
    figure, axis = plt.subplots(figsize=(14, 10), constrained_layout=True)
    drop_hit = global_labels_np == -1
    dual_hit = global_labels_np == -2
    axis.scatter(
        global_xy[drop_hit, 0], global_xy[drop_hit, 1], s=19, c="#9ca3af",
        marker="x", linewidths=0.65, alpha=0.34, label=f"drop (n={drop_hit.sum()})",
        zorder=1,
    )
    legend_handles = [
        Line2D([0], [0], marker="x", linestyle="", color="#9ca3af",
               label=f"drop (n={drop_hit.sum()})", markersize=6)
    ]
    pair_array = np.asarray(
        [pair if pair is not None else (-1, -1) for pair in global_pairs],
        dtype=np.int32,
    )
    for pair in sorted(set(pair for pair in global_pairs if pair is not None)):
        hit = dual_hit & (pair_array[:, 0] == pair[0]) & (pair_array[:, 1] == pair[1])
        axis.scatter(
            global_xy[hit, 0], global_xy[hit, 1], s=34,
            facecolors=[palette[pair[0]]], edgecolors=[palette[pair[1]]],
            marker="D", linewidths=1.05, alpha=0.72, zorder=2,
        )
    legend_handles.append(
        Line2D(
            [0], [0], marker="D", linestyle="", markerfacecolor="#60a5fa",
            markeredgecolor="#be185d", markeredgewidth=1.1, color="none",
            label=f"dual: face=atom A, edge=atom B (n={dual_hit.sum()})", markersize=6,
        )
    )
    for label in range(12):
        hit = global_labels_np == label
        if not np.any(hit):
            continue
        axis.scatter(
            global_xy[hit, 0], global_xy[hit, 1], s=25, c=[palette[label]],
            alpha=0.82, edgecolors="white", linewidths=0.25, zorder=3,
        )
        legend_handles.append(
            Line2D([0], [0], marker="o", linestyle="", color=palette[label],
                   label=f"{ATOMIC_NAMES[label]} (n={hit.sum()})", markersize=6)
        )
    axis.set_title(
        "Global zT t-SNE: 12 single atomic classes, dual combinations, and drop horizons",
        fontsize=15,
    )
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    _clean_axis(axis)
    axis.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                frameon=False, fontsize=8)
    global_path = args.output_dir / "zt_global_12atoms_drop_tsne.png"
    figure.savefig(global_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    single_hit = global_labels_np >= 0
    single_vectors = global_vectors[single_hit]
    single_labels = global_labels_np[single_hit]
    centroids = []
    for label in range(12):
        rows = single_vectors[single_labels == label]
        center = rows.mean(axis=0)
        center /= max(float(np.linalg.norm(center)), 1e-8)
        centroids.append(center)
    centroids = np.stack(centroids)
    predicted = np.argmax(single_vectors @ centroids.T, axis=1)
    own_cosine = np.sum(single_vectors * centroids[single_labels], axis=1)
    drop_vectors = global_vectors[drop_hit]
    drop_nearest = np.max(drop_vectors @ centroids.T, axis=1)
    code_predictions = np.asarray(
        [np.argmax(similarity_by_index[row["index"]]) for row in global_records]
    )[single_hit]
    manifest["global"] = {
        "figure": str(global_path),
        "single_count": int(single_hit.sum()),
        "dual_count": int(dual_hit.sum()),
        "dual_pair_count": len(set(pair for pair in global_pairs if pair is not None)),
        "drop_count": int(drop_hit.sum()),
        "per_class_plotted": {
            ATOMIC_NAMES[label]: int(np.sum(single_labels == label)) for label in range(12)
        },
        "high_dimensional_metrics": {
            "single_class_cosine_silhouette": float(
                silhouette_score(single_vectors, single_labels, metric="cosine")
            ),
            "nearest_class_centroid_accuracy": float(np.mean(predicted == single_labels)),
            "nearest_codebook_top1_accuracy": float(np.mean(code_predictions == single_labels)),
            "single_own_centroid_cosine_mean": float(own_cosine.mean()),
            "single_own_centroid_cosine_median": float(np.median(own_cosine)),
            "drop_nearest_atomic_centroid_cosine_mean": float(drop_nearest.mean()),
            "drop_nearest_atomic_centroid_cosine_median": float(np.median(drop_nearest)),
        },
    }
    manifest_path = args.output_dir / "zt_visualization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
