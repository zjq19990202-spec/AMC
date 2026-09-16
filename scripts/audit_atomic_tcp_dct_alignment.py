#!/usr/bin/env python3
"""Audit atomic-label signs against state-FK and action-FK TCP trajectories."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from atomic_latent_vla.atomic import ATOMIC_NAMES
from atomic_latent_vla.annotation.motion import CartesianPose, base_frame_pose_delta
from atomic_latent_vla.pi05.training_data import build_atomic_text_dataset


def _pose(row: np.ndarray) -> CartesianPose:
    return CartesianPose(row[:3], row[3:].reshape(3, 3))


def _trajectory_delta(
    poses: np.ndarray,
    current_index: int,
    future_indices: np.ndarray,
    state_offset: int,
    future_offset: int,
) -> np.ndarray:
    start = _pose(poses[current_index, state_offset : state_offset + 12])
    return np.stack(
        [
            base_frame_pose_delta(
                start, _pose(poses[index, future_offset : future_offset + 12])
            )
            for index in future_indices
        ]
    )


def _record(bucket: dict, value: float) -> None:
    bucket["count"] += 1
    bucket["signed_values"].append(float(value))
    bucket["matches"] += int(value > 0)


def _summary(bucket: dict) -> dict:
    values = np.asarray(bucket["signed_values"], dtype=np.float64)
    return {
        "count": int(bucket["count"]),
        "sign_match_fraction": float(np.mean(values > 0)) if len(values) else None,
        "signed_mean_median": float(np.median(values)) if len(values) else None,
        "signed_mean_mean": float(np.mean(values)) if len(values) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--indices-file", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", default="/mnt/cunchu/zjq/target")
    parser.add_argument("--norm-asset-id", default="openpi_norm_compact_accepted_v3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = build_atomic_text_dataset(
        (args.dataset_root,),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        action_horizon=50,
        max_token_len=250,
        include_fast=False,
    )
    raw = dataset._raw  # noqa: SLF001
    poses = raw.tcp_pose
    if poses.shape[1] != 48:
        raise ValueError(f"audit requires bimanual sidecar [N,48], got {poses.shape}")
    indices = [int(value) for value in json.loads(args.indices_file.read_text())]

    def fresh():
        return {"count": 0, "matches": 0, "signed_values": []}

    totals = {
        "state_mean": fresh(),
        "action_mean": fresh(),
        "state_endpoint": fresh(),
        "action_endpoint": fresh(),
    }
    by_arm = {
        arm: {key: fresh() for key in totals} for arm in ("right", "left")
    }
    by_kind = {
        kind: {key: fresh() for key in totals} for kind in ("single", "dual")
    }
    by_label = defaultdict(lambda: {key: fresh() for key in totals})
    state_action_same_sign = []
    examples = []

    layouts = {
        "right": (24, 36, slice(0, 6), 0),
        "left": (0, 12, slice(6, 12), 1),
    }
    episodes = set()
    for sample_index in indices:
        metadata = raw.metadata(sample_index)
        data_index = int(raw.base._visible_indices[sample_index])
        episode = int(raw.base._episode_index[data_index])
        episodes.add(episode)
        query_indices, _ = raw.base._get_query_indices(data_index, episode)
        future_indices = np.asarray(query_indices["action"], dtype=np.int64)
        weights = np.asarray(metadata["atomic_weights"])
        action_target = np.asarray(metadata["tcp_twist_delta"])
        for arm, (state_offset, action_offset, target_slice, arm_index) in layouts.items():
            labels = np.flatnonzero(weights[arm_index, :12] > 0)
            if not len(labels):
                continue
            kind = "single" if len(labels) == 1 else "dual"
            state_delta = _trajectory_delta(
                poses, data_index, future_indices, state_offset, state_offset
            )
            sidecar_action_delta = _trajectory_delta(
                poses, data_index, future_indices, state_offset, action_offset
            )
            np.testing.assert_allclose(
                sidecar_action_delta,
                action_target[:, target_slice],
                rtol=1e-5,
                atol=1e-6,
            )
            for label in labels:
                axis = int(label) // 2
                sign = 1.0 if int(label) % 2 == 0 else -1.0
                values = {
                    "state_mean": sign * float(np.mean(state_delta[:, axis])),
                    "action_mean": sign * float(np.mean(sidecar_action_delta[:, axis])),
                    "state_endpoint": sign * float(state_delta[-1, axis]),
                    "action_endpoint": sign * float(sidecar_action_delta[-1, axis]),
                }
                for key, value in values.items():
                    _record(totals[key], value)
                    _record(by_arm[arm][key], value)
                    _record(by_kind[kind][key], value)
                    _record(by_label[ATOMIC_NAMES[int(label)]][key], value)
                state_action_same_sign.append(
                    np.sign(state_delta[:, axis].mean())
                    == np.sign(sidecar_action_delta[:, axis].mean())
                )
                if values["state_mean"] > 0 and values["action_mean"] < 0 and len(examples) < 20:
                    examples.append(
                        {
                            "dataset_index": sample_index,
                            "episode": episode,
                            "frame": int(raw.base._frame_index[data_index]),
                            "arm": arm,
                            "label": ATOMIC_NAMES[int(label)],
                            "prompt": metadata["atomic_prompt"],
                            "signed_state_mean": values["state_mean"],
                            "signed_action_mean": values["action_mean"],
                        }
                    )

    report = {
        "samples": len(indices),
        "unique_episodes": len(episodes),
        "contract": {
            "labels": "observation.state FK over five 1/3-s blocks",
            "dct": "future action FK relative to current observation.state TCP",
            "target_order": "right 6D then left 6D",
        },
        "overall": {key: _summary(value) for key, value in totals.items()},
        "by_arm": {
            arm: {key: _summary(value) for key, value in metrics.items()}
            for arm, metrics in by_arm.items()
        },
        "by_kind": {
            kind: {key: _summary(value) for key, value in metrics.items()}
            for kind, metrics in by_kind.items()
        },
        "by_label": {
            label: {key: _summary(value) for key, value in metrics.items()}
            for label, metrics in sorted(by_label.items())
        },
        "state_action_same_axis_sign_fraction": float(np.mean(state_action_same_sign)),
        "state_matches_but_action_opposes_examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
