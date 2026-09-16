#!/usr/bin/env python3
"""Plot Atomic π0.5 training losses and the learned 12-code geometry."""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from atomic_latent_vla.atomic import ATOMIC_NAMES


_STEP_RE = re.compile(r"\bstep=(\d+)\s+(.*)")
_METRIC_RE = re.compile(r"([a-z_]+)=(-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")


def parse_log(path: Path, max_step: int | None = None) -> dict[str, np.ndarray]:
    rows: list[dict[str, float]] = []
    for line in path.read_text(errors="replace").splitlines():
        match = _STEP_RE.search(line)
        if not match:
            continue
        step = int(match.group(1))
        if max_step is not None and step > max_step:
            continue
        row = {"step": float(step)}
        row.update({key: float(value) for key, value in _METRIC_RE.findall(match.group(2))})
        rows.append(row)
    if not rows:
        raise ValueError(f"no step metrics in {path}")
    keys = set().union(*(row.keys() for row in rows))
    return {key: np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64) for key in keys}


def load_codebook(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)
    with path.open("rb") as handle:
        params = pickle.load(handle)
    codebook = np.asarray(params["codebook"], dtype=np.float32)
    if codebook.shape != (12, 512):
        raise ValueError(f"expected [12,512] codebook, got {codebook.shape}")
    return codebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="adapter params.pkl or a previously extracted [12,512] .npy codebook",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-step", type=int, help="ignore log entries after this checkpoint step")
    parser.add_argument(
        "--embedding-output",
        type=Path,
        help="optional 3-D codebook embedding plot; PCA is the geometry-faithful default",
    )
    parser.add_argument("--embedding-method", choices=("pca", "tsne"), default="pca")
    args = parser.parse_args()

    metrics = parse_log(args.log, max_step=args.max_step)
    codes = load_codebook(args.checkpoint)
    unit_codes = codes / np.maximum(np.linalg.norm(codes, axis=1, keepdims=True), 1e-8)
    similarity = unit_codes @ unit_codes.T
    off_diagonal = similarity[~np.eye(len(similarity), dtype=bool)]

    figure = plt.figure(figsize=(20, 16), constrained_layout=True)
    grid = figure.add_gridspec(3, 3, width_ratios=(1.15, 1.15, 1.5))

    axis = figure.add_subplot(grid[0, :2])
    for key, color in (
        ("loss", "black"),
        ("flow_loss", "#1f77b4"),
        ("coefficient_loss", "#ff7f0e"),
        ("coefficient_velocity_loss", "#17becf"),
        ("coefficient_wall_loss", "#bcbd22"),
        ("fast_action_ce_loss", "#e377c2"),
        ("fast_action_ce_weighted_loss", "#7f7f7f"),
    ):
        if key in metrics:
            axis.plot(metrics["step"], metrics[key], label=key, color=color, linewidth=1.5)
    axis.set_title("Training reconstruction losses")
    axis.set_xlabel("step")
    axis.set_ylabel("loss")
    axis.grid(alpha=0.25)
    axis.legend()

    axis = figure.add_subplot(grid[1, :2])
    # New runs expose every term. The old aggregate names remain supported so
    # existing logs can still be inspected.
    atomic_series = (
        ("full_samplewise_two_way_ranking_loss", "#1f77b4"),
        ("full_ratio_kl_loss", "#17becf"),
        ("text_samplewise_two_way_ranking_loss", "#d62728"),
        ("text_ratio_kl_loss", "#9467bd"),
        ("text_weighted_ratio_kl_loss", "#8c564b"),
        ("full_info_nce_loss", "#1f77b4"),
        ("text_info_nce_loss", "#d62728"),
        ("text_codebook_loss", "#ff7f0e"),
        ("text_commitment_loss", "#9467bd"),
        ("text_weighted_commitment_loss", "#8c564b"),
        ("text_atomic_total_loss", "black"),
        ("atomic_full_loss", "#2ca02c"),
        ("atomic_text_loss", "#d62728"),
    )
    for key, color in atomic_series:
        if key in metrics:
            axis.plot(metrics["step"], metrics[key], label=key, color=color, linewidth=1.5)
    axis.set_title("Atomic supervision: Two-Way ranking / ratio KL / codebook")
    axis.set_xlabel("step")
    axis.set_ylabel("raw loss")
    axis.grid(alpha=0.25)
    axis.legend()

    axis = figure.add_subplot(grid[2, :2])
    if "perplexity" in metrics:
        axis.plot(metrics["step"], metrics["perplexity"], label="perplexity", color="#9467bd", linewidth=1.5)
    axis.axhline(12.0, linestyle="--", linewidth=1, color="gray", label="12-code maximum")
    axis.set_title("Codebook utilization")
    axis.set_xlabel("step")
    axis.set_ylabel("perplexity")
    axis.grid(alpha=0.25)
    axis.legend()

    axis = figure.add_subplot(grid[:, 2])
    image = axis.imshow(similarity, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    short_names = [name.replace("move_", "m_").replace("rotate_", "r_") for name in ATOMIC_NAMES]
    axis.set_xticks(range(12), short_names, rotation=45, ha="right", fontsize=9)
    axis.set_yticks(range(12), short_names, fontsize=9)
    axis.set_title(
        "Learned codebook cosine similarity\n"
        f"off-diagonal mean={off_diagonal.mean():.3f}, max={off_diagonal.max():.3f}"
    )
    for row in range(12):
        for column in range(12):
            color = "white" if abs(similarity[row, column]) > 0.55 else "black"
            axis.text(column, row, f"{similarity[row, column]:.2f}", ha="center", va="center", fontsize=7, color=color)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="cosine similarity")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    np.save(args.output.with_suffix(".npy"), similarity)
    print(f"wrote {args.output}")
    print(f"offdiag_mean={off_diagonal.mean():.6f} offdiag_min={off_diagonal.min():.6f} offdiag_max={off_diagonal.max():.6f}")

    if args.embedding_output is not None:
        if args.embedding_method == "pca":
            embedding = PCA(n_components=3).fit_transform(unit_codes)
            title = "12 atomic codes: 3-D PCA (global geometry)"
        else:
            # With only 12 codes, this is exploratory only: t-SNE distorts global distances.
            embedding = TSNE(
                n_components=3,
                perplexity=3,
                init="pca",
                learning_rate="auto",
                random_state=0,
            ).fit_transform(unit_codes)
            title = "12 atomic codes: 3-D t-SNE (exploratory)"

        figure = plt.figure(figsize=(10, 8), constrained_layout=True)
        axis = figure.add_subplot(111, projection="3d")
        colors = ["#1f77b4"] * 6 + ["#d62728"] * 6
        for index, (name, color) in enumerate(zip(ATOMIC_NAMES, colors, strict=True)):
            x, y, z = embedding[index]
            axis.scatter(x, y, z, s=75, color=color, depthshade=True)
            axis.text(x, y, z, f"  {index}: {name}", fontsize=9)
        axis.set_xlabel("PC1" if args.embedding_method == "pca" else "t-SNE 1")
        axis.set_ylabel("PC2" if args.embedding_method == "pca" else "t-SNE 2")
        axis.set_zlabel("PC3" if args.embedding_method == "pca" else "t-SNE 3")
        axis.set_title(title + "\nblue: translations; red: rotations")
        args.embedding_output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.embedding_output, dpi=200, bbox_inches="tight")
        np.save(args.embedding_output.with_suffix(".npy"), embedding)
        print(f"wrote {args.embedding_output}")


if __name__ == "__main__":
    main()
