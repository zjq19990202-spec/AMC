#!/usr/bin/env python3
"""Evaluate atomic steering only when the opposite arm is stay or unlabeled.

For every selected anchor, the image, robot state, and flow noise are fixed.
Only the text changes among the native subtask, the reviewed atomic prompt,
and its semantic sign reversal.  This makes prompt-pair separation the primary
metric and prevents the scene's nominal motion bias from being counted as
steering.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from atomic_latent_vla.atomic import ATOMIC_NAMES
from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import build_atomic_dataset

import evaluate_many_cluster_single_dual_steering as base
from evaluate_zt_empty_vs_atomic import reverse_atomic_prompt


ARM_INDEX = {"right": 0, "left": 1}
ARM_STATE_SLICE = {"right": slice(8, 15), "left": slice(0, 7)}
FULL_JOINT_INDICES = np.asarray([*range(0, 7), *range(8, 15)], dtype=np.int64)
GRIPPER_INDICES = np.asarray([7, 15], dtype=np.int64)


def _as_trained_atomic_prompt(prompt: str, arm: str, other_status: str) -> str:
    """Recover the single-arm text format used by the existing checkpoints.

    New data code explicitly prefixes every labeled arm.  The evaluated 60K
    and 70K checkpoints predate that change: a sole labeled arm had no prefix,
    while a moving arm paired with a stay arm used both explicit prefixes.
    """

    if other_status == "unlabeled":
        prefix = f"{arm.capitalize()} arm: "
        if prompt.startswith(prefix):
            return prompt[len(prefix) :]
    return prompt


def _collect_candidates(
    dataset,
    *,
    frame_stride: int,
    require_other_stay: bool = False,
) -> tuple[list[dict], dict[str, dict[str, list]]]:
    raw = dataset._raw  # noqa: SLF001
    raw._ensure_annotations()  # noqa: SLF001
    if not raw._uses_target_sidecars or raw._target_sidecars is None:  # noqa: SLF001
        raise ValueError("isolated-arm evaluation requires target annotation sidecars")
    sidecars = raw._target_sidecars  # noqa: SLF001
    candidates: list[dict] = []
    population = {
        arm: {"states": [], "atoms": []}
        for arm in ("right", "left")
    }
    visible = np.asarray(raw.base._visible_indices)  # noqa: SLF001
    frames = np.asarray(raw.base._frame_index[visible])  # noqa: SLF001
    for dataset_index in np.flatnonzero(frames % frame_stride == 0):
        data_index = int(visible[dataset_index])
        frame = int(frames[dataset_index])
        episode = int(raw.base._episode_index[data_index])  # noqa: SLF001
        annotations = [
            sidecars.atomic_horizon(episode, frame, arm=arm)
            for arm in ("right", "left")
        ]
        mask = np.asarray([annotation is not None for annotation in annotations], dtype=bool)
        weights = np.stack(
            [
                annotation.weights
                if annotation is not None
                else np.zeros(13, dtype=np.float32)
                for annotation in annotations
            ]
        )
        full_state_16d = np.asarray(raw.base._states[data_index]).astype(float)  # noqa: SLF001
        for arm, arm_index in ARM_INDEX.items():
            other_index = 1 - arm_index
            active = np.flatnonzero(weights[arm_index, :12] > 0.0)
            state_7d = full_state_16d[ARM_STATE_SLICE[arm]]
            population[arm]["states"].append(state_7d)
            population[arm].setdefault("full_states", []).append(full_state_16d)
            population[arm]["atoms"].append(
                tuple(ATOMIC_NAMES[index] for index in active) if mask[arm_index] else ()
            )
            if not mask[arm_index] or len(active) not in (1, 2) or weights[arm_index, 12] > 0.0:
                continue
            other_active = np.flatnonzero(weights[other_index] > 0.0)
            if not mask[other_index]:
                other_status = "unlabeled"
            elif np.array_equal(other_active, np.asarray([12])):
                other_status = "stay"
            else:
                continue
            if require_other_stay and other_status != "stay":
                continue
            atoms = tuple(ATOMIC_NAMES[index] for index in active)
            if not base._valid_mode(atoms):  # noqa: SLF001
                continue
            present = [
                (name, annotation.prompt)
                for name, annotation in zip(
                    ("Right arm", "Left arm"), annotations, strict=True
                )
                if annotation is not None
            ]
            prompt = (
                present[0][1]
                if len(present) == 1
                else " ".join(f"{name}: {text}" for name, text in present)
            )
            prompt = _as_trained_atomic_prompt(prompt, arm, other_status)
            try:
                reverse_prompt = reverse_atomic_prompt(prompt)
            except ValueError:
                continue
            candidates.append(
                {
                    "dataset_index": int(dataset_index),
                    "episode": episode,
                    "frame": frame,
                    "arm": arm,
                    "kind": "single" if len(atoms) == 1 else "dual",
                    "atoms": list(atoms),
                    "other_status": other_status,
                    "state_7d": state_7d.tolist(),
                    "full_state_16d": full_state_16d.tolist(),
                    "subtask_prompt": sidecars.subtask_prompt(
                        episode, frame, frame + 50
                    ) or sidecars.global_prompt(episode, ""),
                    "atomic_prompt": prompt,
                    "reverse_prompt": reverse_prompt,
                }
            )
    return candidates, population


def _kmeans(
    values: np.ndarray,
    cluster_count: int,
    *,
    seed: int,
    iterations: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Small 7-D K-means implementation; GPU when available, CPU otherwise."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.as_tensor(values, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    indices = torch.randperm(data.shape[0], generator=generator, device=device)[:cluster_count]
    centers = data[indices].clone()
    labels = torch.zeros(data.shape[0], dtype=torch.long, device=device)
    chunk_size = 32768
    for _ in range(iterations):
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
        if torch.max(torch.abs(new_centers - centers)).item() < 1e-4:
            centers = new_centers
            break
        centers = new_centers
    return centers.cpu().numpy(), labels.cpu().numpy()


def _assign_clusters(values: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = torch.as_tensor(values, dtype=torch.float32)
    codebook = torch.as_tensor(centers, dtype=torch.float32)
    distances = (
        torch.sum(data * data, dim=1, keepdim=True)
        + torch.sum(codebook * codebook, dim=1)[None, :]
        - 2.0 * data @ codebook.T
    )
    labels = torch.argmin(distances, dim=1)
    minimum = torch.sqrt(torch.clamp_min(distances.gather(1, labels[:, None])[:, 0], 0.0))
    return labels.numpy(), minimum.numpy()


def _select_diverse_anchors(
    candidates: list[dict],
    population: dict[str, dict[str, list]],
    *,
    state_clusters_per_arm: int,
    minimum_cluster_frames: int,
    minimum_cluster_atoms: int,
    target_clusters: int,
    anchors_per_cluster: int,
    seed: int,
    full_state_clustering: bool = False,
    gripper_distance_weight: float = 0.1,
    anchor_center_quantile: float = 0.5,
) -> tuple[list[dict], list[dict], dict[str, np.ndarray] | None]:
    if anchors_per_cluster < 2:
        raise ValueError("anchors_per_cluster must be at least two")
    if not 0.0 < anchor_center_quantile <= 1.0:
        raise ValueError("anchor_center_quantile must be in (0, 1]")
    pairable_clusters: dict[str, list[dict]] = {"right": [], "left": []}
    shared_centers = None
    shared_assignment = None
    shared_mean = None
    shared_scale = None
    feature_weights = None
    if full_state_clustering:
        shared_states = np.asarray(population["right"]["full_states"], dtype=np.float64)
        shared_mean = shared_states.mean(axis=0)
        shared_scale = np.maximum(shared_states.std(axis=0), 1e-4)
        # Distances are standardized per coordinate.  Grippers remain visible
        # to clustering but contribute only `gripper_distance_weight` to the
        # squared distance, because their units differ from joint radians.
        feature_weights = np.ones(shared_states.shape[-1], dtype=np.float64)
        feature_weights[GRIPPER_INDICES] = np.sqrt(gripper_distance_weight)
        shared_standardized = (shared_states - shared_mean) / shared_scale
        shared_standardized *= feature_weights
        cluster_count = min(state_clusters_per_arm, len(shared_states))
        shared_centers, shared_assignment = _kmeans(
            shared_standardized,
            cluster_count,
            seed=seed,
        )
    for arm in ("right", "left"):
        if full_state_clustering:
            states = np.asarray(population[arm]["full_states"], dtype=np.float64)
            mean, scale = shared_mean, shared_scale
            standardized = (states - mean) / scale * feature_weights
            centers, assignment = shared_centers, shared_assignment
            cluster_count = centers.shape[0]
        else:
            states = np.asarray(population[arm]["states"], dtype=np.float64)
            mean = states.mean(axis=0)
            scale = np.maximum(states.std(axis=0), 1e-4)
            standardized = (states - mean) / scale
            cluster_count = min(state_clusters_per_arm, len(states))
            centers, assignment = _kmeans(
                standardized,
                cluster_count,
                seed=seed + ARM_INDEX[arm],
            )
        atom_sets = [set() for _ in range(cluster_count)]
        mode_sets = [set() for _ in range(cluster_count)]
        for cluster, atoms in zip(assignment, population[arm]["atoms"], strict=True):
            atom_sets[int(cluster)].update(atoms)
            if atoms:
                mode_sets[int(cluster)].add(tuple(atoms))
        frame_counts = np.bincount(assignment, minlength=cluster_count)
        population_eligible = {
            cluster
            for cluster in range(cluster_count)
            if int(frame_counts[cluster]) >= minimum_cluster_frames
            and len(atom_sets[cluster]) >= minimum_cluster_atoms
        }
        arm_candidates = [row for row in candidates if row["arm"] == arm]
        if not arm_candidates:
            continue
        state_key = "full_state_16d" if full_state_clustering else "state_7d"
        candidate_states = np.asarray([row[state_key] for row in arm_candidates])
        candidate_standardized = (candidate_states - mean) / scale
        if full_state_clustering:
            candidate_standardized *= feature_weights
        candidate_clusters, candidate_distances = _assign_clusters(
            candidate_standardized, centers
        )
        by_cluster: dict[int, list[dict]] = defaultdict(list)
        for row, cluster, distance in zip(
            arm_candidates, candidate_clusters, candidate_distances, strict=True
        ):
            cluster = int(cluster)
            if cluster not in population_eligible:
                continue
            item = dict(row)
            item["cluster_key"] = f"{arm}_c{cluster:03d}"
            item["cluster_distance"] = float(distance)
            by_cluster[cluster].append(item)
        for cluster, rows in by_cluster.items():
            modes = {tuple(row["atoms"]) for row in rows}
            if len(modes) < 2:
                continue
            member_states = states[assignment == cluster]
            raw_center = member_states.mean(axis=0)
            if full_state_clustering:
                # First require both arms independently to be close to the full-state
                # cluster center. Random selection then avoids always choosing the
                # single nearest frame while retaining a fixed-seed manifest.
                member_left_rms = np.sqrt(
                    np.mean(np.square(member_states[:, :7] - raw_center[:7]), axis=1)
                )
                member_right_rms = np.sqrt(
                    np.mean(np.square(member_states[:, 8:15] - raw_center[8:15]), axis=1)
                )
                candidate_full_states = np.asarray(
                    [row["full_state_16d"] for row in rows], dtype=np.float64
                )
                candidate_left_rms = np.sqrt(
                    np.mean(np.square(candidate_full_states[:, :7] - raw_center[:7]), axis=1)
                )
                candidate_right_rms = np.sqrt(
                    np.mean(np.square(candidate_full_states[:, 8:15] - raw_center[8:15]), axis=1)
                )
                quantiles = np.unique(
                    np.append(np.arange(anchor_center_quantile, 1.0, 0.1), 1.0)
                )
                central_rows: list[dict] = []
                used_center_quantile = 1.0
                left_center_threshold = float(member_left_rms.max())
                right_center_threshold = float(member_right_rms.max())
                for quantile in quantiles:
                    left_threshold = float(np.quantile(member_left_rms, quantile))
                    right_threshold = float(np.quantile(member_right_rms, quantile))
                    central_rows = [
                        row
                        for row, left_rms, right_rms in zip(
                            rows, candidate_left_rms, candidate_right_rms, strict=True
                        )
                        if left_rms <= left_threshold and right_rms <= right_threshold
                    ]
                    if (
                        len(central_rows) >= anchors_per_cluster
                        and len({tuple(row["atoms"]) for row in central_rows}) >= 2
                    ):
                        used_center_quantile = float(quantile)
                        left_center_threshold = left_threshold
                        right_center_threshold = right_threshold
                        break
                if (
                    len(central_rows) < anchors_per_cluster
                    or len({tuple(row["atoms"]) for row in central_rows}) < 2
                ):
                    continue
                cluster_rng = np.random.default_rng(
                    seed + cluster + 1_000_003 * ARM_INDEX[arm]
                )
                first = central_rows[int(cluster_rng.integers(len(central_rows)))]
                different_mode = [
                    row
                    for row in central_rows
                    if tuple(row["atoms"]) != tuple(first["atoms"])
                ]
                different_episode = [
                    row for row in different_mode if row["episode"] != first["episode"]
                ]
                second_pool = different_episode or different_mode
                second = second_pool[int(cluster_rng.integers(len(second_pool)))]
                selected_rows = [first, second]
                remaining = [
                    row for row in central_rows if row is not first and row is not second
                ]
                while len(selected_rows) < anchors_per_cluster:
                    pick = int(cluster_rng.integers(len(remaining)))
                    selected_rows.append(remaining.pop(pick))
                for row in selected_rows:
                    full_state = np.asarray(row["full_state_16d"], dtype=np.float64)
                    row["left_center_rms_rad"] = float(
                        np.sqrt(np.mean(np.square(full_state[:7] - raw_center[:7])))
                    )
                    row["right_center_rms_rad"] = float(
                        np.sqrt(np.mean(np.square(full_state[8:15] - raw_center[8:15])))
                    )
                joint_delta = member_states[:, FULL_JOINT_INDICES] - raw_center[FULL_JOINT_INDICES]
                gripper_delta = member_states[:, GRIPPER_INDICES] - raw_center[GRIPPER_INDICES]
                raw_rms = np.sqrt(np.mean(np.square(joint_delta), axis=1))
                gripper_rms = np.sqrt(np.mean(np.square(gripper_delta), axis=1))
                selected_joint_delta = (
                    np.asarray(first["full_state_16d"])[FULL_JOINT_INDICES]
                    - np.asarray(second["full_state_16d"])[FULL_JOINT_INDICES]
                )
                selected_gripper_delta = (
                    np.asarray(first["full_state_16d"])[GRIPPER_INDICES]
                    - np.asarray(second["full_state_16d"])[GRIPPER_INDICES]
                )
            else:
                rows.sort(key=lambda row: row["cluster_distance"])
                first = rows[0]
                different_mode = [
                    row for row in rows if tuple(row["atoms"]) != tuple(first["atoms"])
                ]
                different_episode = [
                    row for row in different_mode if row["episode"] != first["episode"]
                ]
                second = (different_episode or different_mode)[0]
                selected_rows = [first, second]
                remaining = [row for row in rows if row is not first and row is not second]
                selected_rows.extend(remaining[: max(0, anchors_per_cluster - 2)])
                if len(selected_rows) < anchors_per_cluster:
                    continue
                used_center_quantile = None
                left_center_threshold = None
                right_center_threshold = None
                central_rows = rows
                raw_rms = np.sqrt(np.mean(np.square(member_states - raw_center), axis=1))
                gripper_rms = None
                selected_joint_delta = np.asarray(first["state_7d"]) - np.asarray(second["state_7d"])
                selected_gripper_delta = None
            selected_state_key = "full_state_16d" if full_state_clustering else "state_7d"
            selected_joint_states = np.asarray(
                [row[selected_state_key] for row in selected_rows], dtype=np.float64
            )
            if full_state_clustering:
                selected_joint_states = selected_joint_states[:, FULL_JOINT_INDICES]
            pair_i, pair_j = np.triu_indices(len(selected_joint_states), k=1)
            selected_pairwise_joint_rms = np.sqrt(
                np.mean(
                    np.square(selected_joint_states[pair_i] - selected_joint_states[pair_j]),
                    axis=1,
                )
            )
            pairable_clusters[arm].append(
                {
                    "cluster_key": f"{arm}_c{cluster:03d}",
                    "arm": arm,
                    "frame_count_3hz": int(frame_counts[cluster]),
                    "state_rms_p90_rad": float(np.quantile(raw_rms, 0.9)),
                    "full_state_clustering": full_state_clustering,
                    "selected_pair_joint_rms_rad": float(
                        np.sqrt(np.mean(np.square(selected_joint_delta)))
                    ),
                    "selected_anchor_pairwise_joint_rms_median_rad": float(
                        np.median(selected_pairwise_joint_rms)
                    ),
                    "selected_anchor_pairwise_joint_rms_max_rad": float(
                        np.max(selected_pairwise_joint_rms)
                    ),
                    "gripper_rms_p90": (
                        float(np.quantile(gripper_rms, 0.9)) if gripper_rms is not None else None
                    ),
                    "selected_pair_gripper_rms": (
                        float(np.sqrt(np.mean(np.square(selected_gripper_delta))))
                        if selected_gripper_delta is not None
                        else None
                    ),
                    "distinct_motion_atoms": sorted(atom_sets[cluster]),
                    "distinct_atom_count": len(atom_sets[cluster]),
                    "distinct_modes": [list(mode) for mode in sorted(mode_sets[cluster])],
                    "distinct_mode_count": len(mode_sets[cluster]),
                    "isolated_candidate_count": len(rows),
                    "isolated_modes": [list(mode) for mode in sorted(modes)],
                    "isolated_mode_count": len(modes),
                    "selected_count": anchors_per_cluster,
                    "anchor_selection": (
                        "seeded_random_bilateral_center_pool"
                        if full_state_clustering
                        else "nearest_center_diverse_mode"
                    ),
                    "requested_center_quantile": (
                        anchor_center_quantile if full_state_clustering else None
                    ),
                    "used_center_quantile": used_center_quantile,
                    "central_candidate_count": len(central_rows),
                    "left_center_threshold_rms_rad": left_center_threshold,
                    "right_center_threshold_rms_rad": right_center_threshold,
                    "selected": selected_rows,
                    "pair_max_cluster_distance": max(
                        row["cluster_distance"] for row in selected_rows
                    ),
                }
            )

    per_arm, remainder = divmod(target_clusters, 2)
    quotas = {"right": per_arm + remainder, "left": per_arm}
    chosen_clusters: list[dict] = []
    leftovers: list[dict] = []
    for arm in ("right", "left"):
        ranked = sorted(
            pairable_clusters[arm],
            key=lambda row: (
                row["used_center_quantile"] if row["used_center_quantile"] is not None else 1.0,
                row["state_rms_p90_rad"],
                row["pair_max_cluster_distance"],
                -row["distinct_atom_count"],
            ),
        )
        chosen_clusters.extend(ranked[: quotas[arm]])
        leftovers.extend(ranked[quotas[arm] :])
    if len(chosen_clusters) < target_clusters:
        leftovers.sort(
            key=lambda row: (
                row["used_center_quantile"] if row["used_center_quantile"] is not None else 1.0,
                row["state_rms_p90_rad"],
                row["pair_max_cluster_distance"],
            )
        )
        chosen_clusters.extend(leftovers[: target_clusters - len(chosen_clusters)])
    if len(chosen_clusters) < target_clusters:
        raise RuntimeError(
            f"only {len(chosen_clusters)} clusters have >={minimum_cluster_frames} frames, "
            f">={minimum_cluster_atoms} atoms, and two isolated modes; requested {target_clusters}"
        )
    selected: list[dict] = []
    cluster_report: list[dict] = []
    for cluster in chosen_clusters[:target_clusters]:
        pair_id = cluster["cluster_key"]
        for pair_index, row in enumerate(cluster["selected"]):
            item = dict(row)
            item["cluster_motion_atoms"] = list(cluster["distinct_motion_atoms"])
            item["cluster_seen_modes"] = list(cluster["distinct_modes"])
            item["cluster_isolated_modes"] = list(cluster["isolated_modes"])
            item["pair_index"] = pair_index
            item["anchor_key"] = (
                f"{pair_id}_p{pair_index}_ep{item['episode']:06d}_f{item['frame']:06d}_"
                + "+".join(item["atoms"])
            )
            selected.append(item)
        cluster_report.append(
            {key: value for key, value in cluster.items() if key != "selected"}
        )
    member_export = None
    if full_state_clustering:
        assert shared_states is not None
        assert shared_assignment is not None
        assert shared_centers is not None
        assert shared_mean is not None
        assert shared_scale is not None
        assert feature_weights is not None
        selected_geometric_ids = np.asarray(
            sorted(
                {
                    int(cluster["cluster_key"].split("_c", 1)[1])
                    for cluster in chosen_clusters[:target_clusters]
                }
            ),
            dtype=np.int16,
        )
        member_mask = np.isin(shared_assignment, selected_geometric_ids)
        member_export = {
            "states": shared_states[member_mask].astype(np.float32),
            "cluster_id": shared_assignment[member_mask].astype(np.int16),
            "selected_cluster_ids": selected_geometric_ids,
            "standardized_centers": shared_centers.astype(np.float32),
            "population_mean": shared_mean,
            "population_scale": shared_scale,
            "feature_weights": feature_weights,
        }
    return selected, cluster_report, member_export


def _threshold(component: int, translation_mm: float, rotation_deg: float) -> float:
    return translation_mm if component < 3 else rotation_deg


def _score(outputs: list[dict], *, translation_mm: float, rotation_deg: float) -> list[dict]:
    grouped: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for row in outputs:
        grouped[(row["anchor_key"], int(row["repeat"]))][row["variant"]] = row
    records = []
    for (anchor_key, repeat), variants in grouped.items():
        empty = variants["empty"]
        subtask = variants["subtask"]
        original = variants["atomic"]
        reverse = variants["reverse"]
        components = [base._component(atom) for atom in original["atoms"]]  # noqa: SLF001
        for step in (25, 50):
            e = np.asarray(empty["trajectory_twist"][str(step)], dtype=np.float64)
            b = np.asarray(subtask["trajectory_twist"][str(step)], dtype=np.float64)
            o = np.asarray(original["trajectory_twist"][str(step)], dtype=np.float64)
            r = np.asarray(reverse["trajectory_twist"][str(step)], dtype=np.float64)
            original_abs, reverse_abs = [], []
            original_vs_empty, reverse_vs_empty = [], []
            original_vs_subtask, reverse_vs_subtask, pair_separation = [], [], []
            margins = []
            for component, sign in components:
                threshold = _threshold(component, translation_mm, rotation_deg)
                original_abs.append(sign * o[component] > threshold)
                reverse_abs.append(-sign * r[component] > threshold)
                original_vs_empty.append(sign * (o[component] - e[component]) > threshold)
                reverse_vs_empty.append(-sign * (r[component] - e[component]) > threshold)
                original_vs_subtask.append(sign * (o[component] - b[component]) > threshold)
                reverse_vs_subtask.append(-sign * (r[component] - b[component]) > threshold)
                margin = sign * (o[component] - r[component])
                pair_separation.append(margin > threshold)
                margins.append(float(margin))
            records.append(
                {
                    "anchor_key": anchor_key,
                    "cluster_key": original["cluster_key"],
                    "episode": original["episode"],
                    "frame": original["frame"],
                    "arm": original["arm"],
                    "kind": original["kind"],
                    "other_status": original["other_status"],
                    "atoms": "+".join(original["atoms"]),
                    "repeat": repeat,
                    "step": step,
                    "pair_steer_success": bool(all(pair_separation)),
                    "original_absolute_success": bool(all(original_abs)),
                    "reverse_absolute_success": bool(all(reverse_abs)),
                    "original_vs_empty_success": bool(all(original_vs_empty)),
                    "reverse_vs_empty_success": bool(all(reverse_vs_empty)),
                    "original_vs_subtask_success": bool(all(original_vs_subtask)),
                    "reverse_vs_subtask_success": bool(all(reverse_vs_subtask)),
                    "minimum_pair_margin": min(margins),
                }
            )
    return records


def _summarize(records: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        for arm in (row["arm"], "both"):
            for kind in (row["kind"], "all"):
                for status in (row["other_status"], "all"):
                    grouped[(arm, kind, status, row["step"])].append(row)
    output = []
    metrics = (
        "pair_steer_success",
        "original_absolute_success",
        "reverse_absolute_success",
        "original_vs_empty_success",
        "reverse_vs_empty_success",
        "original_vs_subtask_success",
        "reverse_vs_subtask_success",
    )
    for (arm, kind, status, step), rows in sorted(grouped.items()):
        summary = {
            "arm": arm,
            "kind": kind,
            "other_status": status,
            "step": step,
            "paired_trials": len(rows),
            "anchors": len({row["anchor_key"] for row in rows}),
            "clusters": len({row["cluster_key"] for row in rows}),
        }
        summary.update({metric + "_rate": float(np.mean([row[metric] for row in rows])) for metric in metrics})
        summary["median_minimum_pair_margin"] = float(
            np.median([row["minimum_pair_margin"] for row in rows])
        )
        output.append(summary)
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", help="NAME=PATH")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=10, help="10 at 30 Hz = 3 Hz anchors")
    parser.add_argument("--state-clusters-per-arm", type=int, default=512)
    parser.add_argument(
        "--full-state-clustering",
        action="store_true",
        help="Cluster the complete [left7, left_gripper, right7, right_gripper] state.",
    )
    parser.add_argument(
        "--gripper-distance-weight",
        type=float,
        default=0.1,
        help="Weight of each gripper coordinate in squared standardized distance.",
    )
    parser.add_argument("--minimum-cluster-frames", type=int, default=101)
    parser.add_argument("--minimum-cluster-atoms", type=int, default=4)
    parser.add_argument(
        "--require-other-stay",
        action="store_true",
        help="Require the opposite arm to carry an explicit stay atom; exclude unlabeled arms.",
    )
    parser.add_argument("--target-clusters", type=int, default=250)
    parser.add_argument("--anchors-per-cluster", type=int, default=4)
    parser.add_argument(
        "--anchor-center-quantile",
        type=float,
        default=0.5,
        help=(
            "For full-state clustering, randomly sample different-mode anchors only after "
            "both arms fall within this per-arm RMS quantile around the cluster center. "
            "The pool expands by 0.1 only when two modes are unavailable."
        ),
    )
    parser.add_argument("--noise-repeats", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--translation-threshold-mm", type=float, default=5.0)
    parser.add_argument("--rotation-threshold-deg", type=float, default=1.0)
    parser.add_argument("--max-token-len", type=int, default=200)
    parser.add_argument(
        "--coefficient-target-kind",
        choices=("tcp_twist", "joint_delta"),
        default="joint_delta",
    )
    parser.add_argument("--coefficient-target-dim", type=int, default=14)
    parser.add_argument(
        "--enable-layerwise-atomic-flow",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--norm-assets-dir",
        type=Path,
        default=Path("/mnt/cunchu/zjq/target"),
    )
    parser.add_argument(
        "--norm-asset-id",
        default="openpi_norm_union2375_allframes_v1",
    )
    parser.add_argument(
        "--atomic-composition-sidecar",
        default="fk_horizon_3hz_gate_top5_stay_v2",
    )
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="Reuse an already reviewed selection instead of rebuilding state clusters.",
    )
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = AtomicPi05Config(
        max_token_len=args.max_token_len,
        fast_action_ce_loss_weight=0.0,
        coefficient_target_kind=args.coefficient_target_kind,
        coefficient_target_dim=args.coefficient_target_dim,
        enable_layerwise_atomic_flow=args.enable_layerwise_atomic_flow,
    )
    dataset = build_atomic_dataset(
        (args.dataset_root,),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        include_fast=False,
        atomic_composition_sidecar=args.atomic_composition_sidecar,
    )
    if args.selection_manifest is None:
        candidates, population = _collect_candidates(
            dataset,
            frame_stride=args.frame_stride,
            require_other_stay=args.require_other_stay,
        )
        anchors, clusters, cluster_members = _select_diverse_anchors(
            candidates,
            population,
            state_clusters_per_arm=args.state_clusters_per_arm,
            minimum_cluster_frames=args.minimum_cluster_frames,
            minimum_cluster_atoms=args.minimum_cluster_atoms,
            target_clusters=args.target_clusters,
            anchors_per_cluster=args.anchors_per_cluster,
            seed=args.seed,
            full_state_clustering=args.full_state_clustering,
            gripper_distance_weight=args.gripper_distance_weight,
            anchor_center_quantile=args.anchor_center_quantile,
        )
        candidate_count = len(candidates)
    else:
        reused = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        anchors = reused["anchors"]
        clusters = reused["clusters"]
        candidate_count = int(reused["candidate_count"])
        cluster_members = None
    if not anchors:
        raise RuntimeError("no isolated-arm atomic anchors found")

    if args.selection_manifest is not None:
        # A reused manifest is part of the paired-evaluation contract. Preserve it
        # byte-for-byte semantically instead of replacing its selection metadata
        # with the current invocation's CLI defaults.
        selection_payload = reused
    else:
        selection_payload = {
            "contract": "target arm has one/two motion atoms; opposite arm is stay or unlabeled",
            "frame_stride": args.frame_stride,
            "state_clusters_per_arm": args.state_clusters_per_arm,
            "full_state_clustering": args.full_state_clustering,
            "gripper_distance_weight": args.gripper_distance_weight,
            "minimum_cluster_frames": args.minimum_cluster_frames,
            "minimum_cluster_atoms": args.minimum_cluster_atoms,
            "require_other_stay": args.require_other_stay,
            "target_clusters": args.target_clusters,
            "anchors_per_cluster": args.anchors_per_cluster,
            "anchor_selection": (
                "seeded_random_bilateral_center_pool"
                if args.full_state_clustering
                else "nearest_center_diverse_mode"
            ),
            "anchor_center_quantile": args.anchor_center_quantile,
            "seed": args.seed,
            "candidate_count": candidate_count,
            "anchor_count": len(anchors),
            "clusters": clusters,
            "anchors": anchors,
            "existing_checkpoint_prompt_format": True,
        }
    (args.output_dir / "isolated_arm_selection_manifest.json").write_text(
        json.dumps(selection_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if cluster_members is not None:
        np.savez_compressed(args.output_dir / "selected_cluster_members.npz", **cluster_members)
        unique_geometric_clusters = len(cluster_members["selected_cluster_ids"])
        (args.output_dir / "selected_cluster_members.json").write_text(
            json.dumps(
                {
                    "contract": (
                        "All 3 Hz population states assigned during the exact same K-means "
                        "call that produced isolated_arm_selection_manifest.json"
                    ),
                    "selected_cluster_count_with_arm_prefix": len(clusters),
                    "unique_geometric_cluster_count": unique_geometric_clusters,
                    "member_state_count": len(cluster_members["states"]),
                    "state_layout": "[left7, left_gripper, right7, right_gripper]",
                    "gripper_distance_weight": args.gripper_distance_weight,
                    "seed": args.seed,
                    "anchor_selection": "seeded_random_bilateral_center_pool",
                    "anchor_center_quantile": args.anchor_center_quantile,
                    "kmeans_device_policy": "CUDA when available, otherwise CPU",
                    "strict_rerun_determinism": False,
                    "reproducibility_note": (
                        "Reuse selected_cluster_members.npz and this manifest for exact "
                        "downstream reproduction; CUDA floating-point reductions can alter "
                        "boundary assignments when K-means is rerun."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if args.selection_only:
        print(
            json.dumps(
                {
                    "candidate_count": candidate_count,
                    "cluster_count": len(clusters),
                    "anchor_count": len(anchors),
                },
                indent=2,
            )
        )
        return
    if not args.checkpoint:
        raise ValueError("provide at least one --checkpoint unless --selection-only is used")

    rng = np.random.default_rng(args.seed)
    row_specs, noises = [], []
    for anchor in anchors:
        for repeat in range(args.noise_repeats):
            noise = rng.standard_normal((config.action_horizon, config.action_dim), dtype=np.float32)
            for variant, prompt in (
                ("empty", ""),
                ("subtask", anchor["subtask_prompt"]),
                ("atomic", anchor["atomic_prompt"]),
                ("reverse", anchor["reverse_prompt"]),
            ):
                row_specs.append(
                    {
                        **{key: value for key, value in anchor.items() if key != "state_7d"},
                        "repeat": repeat,
                        "variant": variant,
                        "prompt": prompt,
                    }
                )
                noises.append(noise)
    noises_np = np.stack(noises)
    noise_sha256 = hashlib.sha256(
        np.ascontiguousarray(noises_np).view(np.uint8)
    ).hexdigest()

    checkpoints = dict(item.split("=", 1) for item in args.checkpoint)
    model_outputs, model_records, model_summaries = {}, {}, {}
    for name, checkpoint in checkpoints.items():
        outputs = base._evaluate_checkpoint(  # noqa: SLF001
            name,
            Path(checkpoint),
            config,
            dataset,
            args.dataset_root,
            row_specs,
            noises_np,
            batch_size=args.batch_size,
            stored_steps=(25, 50),
            norm_assets_dir=args.norm_assets_dir,
            norm_asset_id=args.norm_asset_id,
        )
        records = _score(
            outputs,
            translation_mm=args.translation_threshold_mm,
            rotation_deg=args.rotation_threshold_deg,
        )
        summary = _summarize(records)
        model_outputs[name] = outputs
        model_records[name] = records
        model_summaries[name] = summary
        _write_csv(args.output_dir / f"{name}_isolated_arm_steering_summary.csv", summary)

    payload = {
        "checkpoints": checkpoints,
        "selection": selection_payload,
        "metric": {
            "primary": "atomic minus reversed-atomic endpoint component under identical observation/state/noise",
            "pair_threshold_translation_mm": args.translation_threshold_mm,
            "pair_threshold_rotation_deg": args.rotation_threshold_deg,
            "secondary": "absolute sign from initial TCP and change relative to empty/native-subtask prompts",
        },
        "evaluation_contract": {
            "max_token_len": config.max_token_len,
            "coefficient_target_kind": config.coefficient_target_kind,
            "coefficient_target_dim": config.coefficient_target_dim,
            "enable_layerwise_atomic_flow": config.enable_layerwise_atomic_flow,
            "norm_assets_dir": str(args.norm_assets_dir),
            "norm_asset_id": args.norm_asset_id,
            "atomic_composition_sidecar": args.atomic_composition_sidecar,
            "tcp_offset_m": 0.20,
            "action_horizon": config.action_horizon,
            "noise_sha256": noise_sha256,
            "same_noise_variants": ["empty", "subtask", "atomic", "reverse"],
        },
        "summary": model_summaries,
        "records": model_records,
        "outputs": model_outputs,
    }
    (args.output_dir / "isolated_arm_atomic_steering.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    selection_path = args.output_dir / "isolated_arm_selection_manifest.json"
    selection_sha256 = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    command = " ".join(shlex.quote(value) for value in sys.argv)
    (args.output_dir / "run_contract.md").write_text(
        "\n".join(
            [
                "# Isolated-arm atomic steering evaluation",
                "",
                f"- Checkpoints: `{json.dumps(checkpoints, ensure_ascii=False)}`",
                f"- Dataset: `{args.dataset_root}`",
                f"- Norm assets: `{args.norm_assets_dir}`",
                f"- Norm ID: `{args.norm_asset_id}`",
                f"- Atomic sidecar: `{args.atomic_composition_sidecar}`",
                f"- Model config: layerwise={args.enable_layerwise_atomic_flow}, "
                f"{args.coefficient_target_kind} {args.coefficient_target_dim}D, "
                f"max_token_len={args.max_token_len}",
                "- State clustering: full bimanual joints, gripper distance weight 0",
                f"- Selection: `{selection_path}`",
                f"- Selection SHA-256: `{selection_sha256}`",
                f"- Seed: {args.seed}",
                f"- Noise repeats: {args.noise_repeats}",
                f"- Noise SHA-256: `{noise_sha256}`",
                "- Paired variants: empty, native subtask, native atomic, exact reversal",
                "- TCP offset: 0.20 m",
                f"- Command: `{command}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "candidate_count": candidate_count,
                "anchor_count": len(anchors),
                "summary": model_summaries,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
