#!/usr/bin/env python3
"""Rebuild shared full-state K-means and export all selected-cluster members."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch


GRIPPER_INDICES = np.asarray([7, 15])
FULL_JOINT_INDICES = np.asarray([*range(7), *range(8, 15)])


def _kmeans(
    values: np.ndarray,
    cluster_count: int,
    *,
    seed: int,
    iterations: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.as_tensor(values, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    indices = torch.randperm(data.shape[0], generator=generator, device=device)[:cluster_count]
    centers = data[indices].clone()
    labels = torch.zeros(data.shape[0], dtype=torch.long, device=device)
    chunk_size = 32768
    for iteration in range(iterations):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(cluster_count, dtype=torch.float32, device=device)
        center_norm = torch.sum(centers * centers, dim=1)
        for start in range(0, data.shape[0], chunk_size):
            chunk = data[start : start + chunk_size]
            distances = (
                torch.sum(chunk * chunk, dim=1, keepdim=True)
                + center_norm[None, :]
                - 2.0 * chunk @ centers.T
            )
            chunk_labels = torch.argmin(distances, dim=1)
            labels[start : start + len(chunk)] = chunk_labels
            sums.index_add_(0, chunk_labels, chunk)
            counts += torch.bincount(chunk_labels, minlength=cluster_count)
        nonempty = counts > 0
        new_centers = centers.clone()
        new_centers[nonempty] = sums[nonempty] / counts[nonempty, None]
        maximum_change = torch.max(torch.abs(new_centers - centers)).item()
        print(f"kmeans iteration={iteration + 1} max_change={maximum_change:.7f}", flush=True)
        centers = new_centers
        if maximum_change < 1e-4:
            break
    return centers.cpu().numpy(), labels.cpu().numpy()


def _read_population(dataset_root: Path, frame_stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {dataset_root / 'data'}")
    states, episodes, frames = [], [], []
    for path in files:
        table = pq.read_table(
            path,
            columns=["observation.state", "episode_index", "frame_index"],
        )
        frame = table["frame_index"].to_numpy()
        keep = frame % frame_stride == 0
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        states.append(state[keep])
        episodes.append(table["episode_index"].to_numpy()[keep])
        frames.append(frame[keep])
        print(f"read {path.name}: kept {int(keep.sum())}/{len(keep)}", flush=True)
    return np.concatenate(states), np.concatenate(episodes), np.concatenate(frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not manifest["full_state_clustering"]:
        raise ValueError("manifest is not full-state clustering")
    states, episodes, frames = _read_population(args.dataset_root, manifest["frame_stride"])
    mean = states.mean(axis=0, dtype=np.float64)
    scale = np.maximum(states.std(axis=0, dtype=np.float64), 1e-4)
    feature_weights = np.ones(16, dtype=np.float64)
    feature_weights[GRIPPER_INDICES] = np.sqrt(manifest["gripper_distance_weight"])
    standardized = (states.astype(np.float64) - mean) / scale * feature_weights
    centers, labels = _kmeans(
        standardized,
        min(manifest["state_clusters_per_arm"], len(states)),
        seed=args.seed,
    )

    selected_ids = np.asarray(
        sorted({int(row["cluster_key"].split("_c", 1)[1]) for row in manifest["clusters"]}),
        dtype=np.int16,
    )
    selected_mask = np.isin(labels, selected_ids)
    raw_centers = np.empty_like(centers, dtype=np.float64)
    nonzero = feature_weights != 0
    raw_centers[:, nonzero] = centers[:, nonzero] / feature_weights[nonzero] * scale[nonzero] + mean[nonzero]
    raw_centers[:, ~nonzero] = mean[None, ~nonzero]

    expected: dict[int, dict[str, float]] = {}
    for row in manifest["clusters"]:
        cluster_id = int(row["cluster_key"].split("_c", 1)[1])
        expected.setdefault(
            cluster_id,
            {"frame_count": row["frame_count_3hz"], "state_rms_p90_rad": row["state_rms_p90_rad"]},
        )
    validation = []
    for cluster_id in selected_ids:
        members = states[labels == cluster_id]
        raw_center = members.mean(axis=0)
        joint_delta = members[:, FULL_JOINT_INDICES] - raw_center[FULL_JOINT_INDICES]
        raw_rms = np.sqrt(np.mean(np.square(joint_delta), axis=1))
        actual_count = int(len(members))
        actual_p90 = float(np.quantile(raw_rms, 0.9))
        reference = expected[int(cluster_id)]
        validation.append(
            {
                "cluster_id": int(cluster_id),
                "actual_frame_count": actual_count,
                "expected_frame_count": int(reference["frame_count"]),
                "actual_state_rms_p90_rad": actual_p90,
                "expected_state_rms_p90_rad": float(reference["state_rms_p90_rad"]),
                "frame_count_match": actual_count == int(reference["frame_count"]),
                "p90_absolute_error": abs(actual_p90 - float(reference["state_rms_p90_rad"])),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.validation_only:
        np.savez_compressed(
            args.output,
            states=states[selected_mask],
            cluster_id=labels[selected_mask].astype(np.int16),
            episode=episodes[selected_mask].astype(np.int32),
            frame=frames[selected_mask].astype(np.int32),
            selected_cluster_ids=selected_ids,
            standardized_centers=centers,
            raw_centers=raw_centers,
            population_mean=mean,
            population_scale=scale,
            feature_weights=feature_weights,
        )
    report = {
        "dataset_root": str(args.dataset_root),
        "manifest": str(args.manifest),
        "seed": args.seed,
        "population_count": len(states),
        "selected_geometric_cluster_count": len(selected_ids),
        "exported_member_count": int(selected_mask.sum()),
        "all_frame_counts_match": all(row["frame_count_match"] for row in validation),
        "maximum_p90_absolute_error": max(row["p90_absolute_error"] for row in validation),
        "clusters": validation,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "clusters"}, indent=2))


if __name__ == "__main__":
    main()
