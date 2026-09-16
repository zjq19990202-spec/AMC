#!/usr/bin/env python3
"""Find state clusters that contain multiple macro-consistent atomic modes."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from atomic_latent_vla.atomic import ATOMIC_NAMES
from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import build_atomic_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--clusters", type=int, default=128)
    parser.add_argument("--min-episodes", type=int, default=4)
    parser.add_argument("--translation-min-m", type=float, default=0.01)
    parser.add_argument("--rotation-min-deg", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = AtomicPi05Config(max_token_len=250, fast_action_ce_loss_weight=0.0)
    dataset = build_atomic_dataset(
        (args.dataset_root,),
        norm_assets_dir="/mnt/cunchu/zjq/target",
        norm_asset_id="openpi_norm_compact_accepted_v3",
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        include_fast=False,
    )
    raw = dataset._raw  # noqa: SLF001
    raw._ensure_annotations()  # noqa: SLF001
    rows: list[dict] = []
    rotation_min = np.deg2rad(args.rotation_min_deg)
    for dataset_index, data_index in enumerate(raw.base._visible_indices):  # noqa: SLF001
        if int(raw._type[data_index]) <= 0:  # noqa: SLF001
            continue
        metadata = raw.metadata(dataset_index)
        if not metadata["atomic_supervision_mask"]:
            continue
        weights = np.asarray(metadata["atomic_weights"])
        active = np.flatnonzero(weights > 0)
        macro_delta = np.asarray(metadata["tcp_twist_delta"])[-1]
        consistent = True
        for label_index in active:
            axis = int(label_index) // 2
            expected_positive = int(label_index) % 2 == 0
            threshold = args.translation_min_m if axis < 3 else rotation_min
            value = float(macro_delta[axis])
            consistent &= (
                (value >= 0.0) == expected_positive and abs(value) >= threshold
            )
        if not consistent:
            continue
        rows.append(
            {
                "dataset_index": dataset_index,
                "episode": int(raw.base._episode_index[data_index]),  # noqa: SLF001
                "labels": tuple(ATOMIC_NAMES[index] for index in active),
                "state": np.asarray(raw.base._states[data_index])[8:15],  # noqa: SLF001
            }
        )

    states = np.stack([row["state"] for row in rows])
    assignment = KMeans(
        n_clusters=args.clusters,
        n_init=20,
        random_state=20260802,
    ).fit_predict(states)
    candidates = []
    for cluster_index in range(args.clusters):
        indices = np.flatnonzero(assignment == cluster_index)
        by_mode: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for index in indices:
            by_mode[rows[index]["labels"]].append(int(index))
        modes = []
        for labels, mode_indices in by_mode.items():
            episodes = {rows[index]["episode"] for index in mode_indices}
            if len(episodes) < args.min_episodes:
                continue
            mode_states = states[mode_indices]
            modes.append(
                {
                    "labels": list(labels),
                    "rows": len(mode_indices),
                    "episodes": len(episodes),
                    "state_mean": mode_states.mean(axis=0).tolist(),
                }
            )
        if len(modes) < 2:
            continue
        cluster_states = states[indices]
        center = cluster_states.mean(axis=0)
        distances = np.sqrt(np.mean(np.square(cluster_states - center), axis=1))
        mode_centers = np.stack([np.asarray(mode["state_mean"]) for mode in modes])
        pairwise = np.sqrt(
            np.mean(
                np.square(mode_centers[:, None, :] - mode_centers[None, :, :]),
                axis=-1,
            )
        )
        candidates.append(
            {
                "cluster": cluster_index,
                "center": center.tolist(),
                "rows": len(indices),
                "state_rms_median": float(np.median(distances)),
                "state_rms_p90": float(np.quantile(distances, 0.9)),
                "max_mode_center_rms": float(pairwise.max()),
                "modes": sorted(
                    modes,
                    key=lambda item: (-item["episodes"], item["labels"]),
                ),
            }
        )
    candidates.sort(
        key=lambda item: (
            -len(item["modes"]),
            item["state_rms_p90"],
            -sum(mode["episodes"] for mode in item["modes"]),
        )
    )
    result = {
        "macro_consistent_rows": len(rows),
        "clusters": args.clusters,
        "min_episodes": args.min_episodes,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for candidate in candidates[:30]:
        modes = "; ".join(
            f"{'+'.join(mode['labels'])}={mode['episodes']}eps"
            for mode in candidate["modes"]
        )
        print(
            f"C{candidate['cluster']:03d} p90={candidate['state_rms_p90']:.4f} "
            f"between={candidate['max_mode_center_rms']:.4f}: {modes}"
        )


if __name__ == "__main__":
    main()
