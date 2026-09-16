#!/usr/bin/env python3
"""Counterfactual 12-atom sweep at frequent proprioceptive state clusters."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans2

from openpi.models import model as _model

from atomic_latent_vla.atomic import ATOMIC_NAMES
from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import (
    _paligemma_tokenizer,
    atomic_collate,
    batch_to_observation,
    build_atomic_dataset,
)
import evaluate_zm_fk_trajectory_ablation as _fk_eval
from evaluate_global_episode_chunks import _sample_global
from evaluate_zm_fk_trajectory_ablation import _output_transform, _world_rotation_vector_deg


ATOMS = tuple(ATOMIC_NAMES[:12])
PROMPT_MODES = (*ATOMS, "__neutral__")
PHRASES = {
    "move_x_pos": "Move forward along base-frame +x.",
    "move_x_neg": "Move backward along base-frame -x.",
    "move_y_pos": "Move leftward along base-frame +y.",
    "move_y_neg": "Move rightward along base-frame -y.",
    "move_z_pos": "Move upward along base-frame +z.",
    "move_z_neg": "Move downward along base-frame -z.",
    "rotate_x_pos": "Rotate positively about the base-frame x axis.",
    "rotate_x_neg": "Rotate negatively about the base-frame x axis.",
    "rotate_y_pos": "Rotate positively about the base-frame y axis.",
    "rotate_y_neg": "Rotate negatively about the base-frame y axis.",
    "rotate_z_pos": "Rotate positively about the base-frame z axis.",
    "rotate_z_neg": "Rotate negatively about the base-frame z axis.",
}


def _prompt(arm: str, atom: str) -> str:
    other = "left" if arm == "right" else "right"
    if atom == "__neutral__":
        return (
            f"{arm.capitalize()} arm: Keep the {arm} TCP stationary in its current "
            f"base-frame pose. {other.capitalize()} arm: Keep the {other} TCP "
            "stationary in its current base-frame pose."
        )
    moving = f"{arm.capitalize()} arm: {PHRASES[atom]}"
    stationary = (
        f"{other.capitalize()} arm: Keep the {other} TCP stationary in its "
        "current base-frame pose."
    )
    return f"{moving} {stationary}"


def _collect_states(dataset, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = dataset._raw  # noqa: SLF001
    dataset_indices = np.arange(0, len(dataset), stride, dtype=np.int64)
    states, weights = [], []
    for index in dataset_indices:
        metadata = raw.metadata(int(index))
        states.append(np.asarray(metadata["raw_state"], dtype=np.float32)[:16])
        weights.append(np.asarray(metadata["atomic_weights"], dtype=np.float32))
    return dataset_indices, np.stack(states), np.stack(weights)


def _cluster_representatives(
    dataset, *, stride: int, cluster_count: int, clusters_per_arm: int, seed: int
) -> list[dict]:
    indices, states, weights = _collect_states(dataset, stride)
    raw = dataset._raw  # noqa: SLF001
    selected = []
    for arm_index, (arm, q_slice) in enumerate(
        (("right", slice(8, 15)), ("left", slice(0, 7)))
    ):
        q = states[:, q_slice]
        mean = q.mean(axis=0)
        scale = np.maximum(q.std(axis=0), 0.05)
        standardized = (q - mean) / scale
        np.random.seed(seed + arm_index)
        centroids, assignment = kmeans2(
            standardized, cluster_count, iter=50, minit="++", seed=seed + arm_index
        )
        candidates = []
        for cluster in range(cluster_count):
            members = np.flatnonzero(assignment == cluster)
            if len(members) < 10:
                continue
            label_presence = np.any(weights[members, arm_index, :12] > 0, axis=0)
            diversity = int(label_presence.sum())
            distances = np.linalg.norm(standardized[members] - centroids[cluster], axis=1)
            representative_member = int(members[int(np.argmin(distances))])
            representative_index = int(indices[representative_member])
            data_index = int(raw.base._visible_indices[representative_index])  # noqa: SLF001
            episode = int(raw.base._episode_index[data_index])  # noqa: SLF001
            candidates.append(
                {
                    "arm": arm,
                    "cluster": cluster,
                    "sampled_count": int(len(members)),
                    "estimated_full_count": int(len(members) * stride),
                    "label_diversity": diversity,
                    "labels_seen": [ATOMS[i] for i in np.flatnonzero(label_presence)],
                    "dataset_index": representative_index,
                    "data_index": data_index,
                    "episode": episode,
                    "frame": int(raw.base._frame_index[data_index]),  # noqa: SLF001
                    "center_qpos": (centroids[cluster] * scale + mean).tolist(),
                    "representative_rms_standardized": float(
                        distances.min() / np.sqrt(q.shape[1])
                    ),
                    "score": float(len(members) * (1.0 + 0.35 * diversity)),
                }
            )
        candidates.sort(key=lambda row: (row["score"], row["label_diversity"]), reverse=True)
        selected.extend(candidates[:clusters_per_arm])
    return selected


def _representatives_from_report(
    path: Path, clusters_per_arm: int, arms: tuple[str, ...]
) -> list[dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    selected = []
    for arm in arms:
        for cluster in report["arms"][arm]["recommended"][:clusters_per_arm]:
            representative = min(
                cluster["representatives"], key=lambda row: row["rms_to_center"]
            )
            selected.append(
                {
                    "arm": arm,
                    "cluster": int(cluster["cluster"]),
                    "sampled_count": int(cluster["samples_3hz"]),
                    "estimated_full_count": int(cluster["samples_3hz"] * 10),
                    "label_diversity": int(cluster["supported_atom_count"]),
                    "labels_seen": list(cluster["atom_counts"]),
                    "dataset_index": int(representative["dataset_index"]),
                    "data_index": -1,
                    "episode": int(representative["episode"]),
                    "frame": int(representative["frame"]),
                    "center_qpos": cluster["center_raw_qpos_7d"],
                    "representative_rms_standardized": float(
                        representative["rms_to_center"]
                    ),
                    "score": float(cluster["recommendation_score"]),
                    "rms_radius_p90_rad": float(cluster["rms_radius_p90_rad"]),
                    "episode_count": int(cluster["episodes"]),
                }
            )
    return selected


def _pose_trajectory(fk, raw_state: np.ndarray, actions: np.ndarray, arm: str):
    q_slice = slice(8, 15) if arm == "right" else slice(0, 7)
    current_position, current_rotation = fk.pose(raw_state[q_slice])
    poses = [fk.pose(row[q_slice]) for row in actions]
    positions = np.stack([pose[0] for pose in poses])
    rotations = np.stack([pose[1] for pose in poses])
    rotation_vectors = np.stack(
        [_world_rotation_vector_deg(current_rotation, rotation) for rotation in rotations]
    )
    return (
        np.concatenate([np.zeros((1, 3)), (positions - current_position) * 1000.0]),
        np.concatenate([np.zeros((1, 3)), rotation_vectors]),
    )


def _evaluate_checkpoint(
    name: str,
    checkpoint: Path,
    config,
    observation,
    prompt_tokens,
    prompt_masks,
    batch,
    rows,
    dataset,
    dataset_root: Path,
    noise_rows,
    noise_repeats: int,
):
    params = _model.restore_params(checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    decode = _output_transform(dataset_root, config)
    _fk_eval.TCP_LOCAL_Z_OFFSET_M = 0.20
    fk = _fk_eval.SimpleCR1FK(
        Path("assets/cr1_right_arm_tcp_red/urdf/CR1ARMR_TCP_RED.urdf")
    )
    predictions = [[] for _ in rows]
    for repeat in range(noise_repeats):
        values = jax.device_get(
            _sample_global(
                model,
                observation,
                jnp.asarray(noise_rows[repeat]),
                prompt_tokens,
                prompt_masks,
                None,
            )
        )
        for index, value in enumerate(np.asarray(values)):
            raw_state = np.asarray(rows[index]["metadata"]["raw_state"])
            actions = decode(
                np.asarray(batch["state"][index]), raw_state, value
            )["actions"]
            predictions[index].append(_pose_trajectory(fk, raw_state, actions, rows[index]["arm"]))
        print(f"{name}: sampled repeat {repeat + 1}/{noise_repeats}", flush=True)
    output = []
    for row, repeats in zip(rows, predictions, strict=True):
        position = np.mean([item[0] for item in repeats], axis=0)
        rotation = np.mean([item[1] for item in repeats], axis=0)
        output.append(
            {
                "cluster_key": row["cluster_key"],
                "arm": row["arm"],
                "atom": row["atom"],
                "prompt": row["prompt"],
                "tcp_position_mm": position.tolist(),
                "tcp_rotation_vector_deg": rotation.tolist(),
                "endpoint_twist": np.concatenate([position[-1], rotation[-1]]).tolist(),
            }
        )
    del model, params
    gc.collect()
    jax.clear_caches()
    return output


def _target_component(atom: str, twist: np.ndarray) -> float:
    axis = "xyz".index(atom.split("_")[1])
    sign = 1.0 if atom.endswith("pos") else -1.0
    offset = 3 if atom.startswith("rotate") else 0
    return float(sign * twist[offset + axis])


def _metrics(outputs: list[dict]) -> dict:
    by_cluster = {}
    for row in outputs:
        by_cluster.setdefault(row["cluster_key"], {})[row["atom"]] = row
    pair_rows = []
    target_components = []
    for cluster_key, atoms in by_cluster.items():
        neutral = np.asarray(atoms["__neutral__"]["endpoint_twist"])
        for atom, row in atoms.items():
            if atom == "__neutral__":
                continue
            component = _target_component(atom, np.asarray(row["endpoint_twist"]))
            steering_component = _target_component(
                atom, np.asarray(row["endpoint_twist"]) - neutral
            )
            row["target_component"] = component
            row["steering_target_component_vs_neutral"] = steering_component
            target_components.append(component)
        for family in ("move", "rotate"):
            for axis in "xyz":
                positive = atoms[f"{family}_{axis}_pos"]["endpoint_twist"]
                negative = atoms[f"{family}_{axis}_neg"]["endpoint_twist"]
                component = axis if family == "move" else f"r{axis}"
                coordinate = "xyz".index(axis) + (3 if family == "rotate" else 0)
                separation = float(np.asarray(positive)[coordinate] - np.asarray(negative)[coordinate])
                pair_rows.append(
                    {
                        "cluster_key": cluster_key,
                        "component": component,
                        "positive_minus_negative": separation,
                        "ordered": bool(separation > 0),
                    }
                )
    return {
        "positive_target_component_rate": float(np.mean(np.asarray(target_components) > 0)),
        "mean_target_component": float(np.mean(target_components)),
        "positive_steering_vs_neutral_rate": float(
            np.mean(
                [
                    row["steering_target_component_vs_neutral"] > 0
                    for atoms in by_cluster.values()
                    for atom, row in atoms.items()
                    if atom != "__neutral__"
                ]
            )
        ),
        "opposite_pair_order_rate": float(np.mean([row["ordered"] for row in pair_rows])),
        "pairs": pair_rows,
    }


def _plot_cluster(cluster: dict, model_outputs: dict[str, list[dict]], output: Path):
    key = f"{cluster['arm']}_c{cluster['cluster']}"
    figure = plt.figure(figsize=(19, 11), constrained_layout=True)
    grid = figure.add_gridspec(2, len(model_outputs))
    colors = plt.get_cmap("tab10")
    for column, (model_name, outputs) in enumerate(model_outputs.items()):
        rows = {row["atom"]: row for row in outputs if row["cluster_key"] == key}
        position_axis = figure.add_subplot(grid[0, column], projection="3d")
        for index, atom in enumerate(ATOMS[:6]):
            xyz = np.asarray(rows[atom]["tcp_position_mm"])
            position_axis.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=colors(index), label=atom)
            position_axis.scatter(*xyz[-1], color=colors(index), s=22)
        position_axis.set_title(f"{model_name}: translation prompts")
        position_axis.set_xlabel("base x / mm")
        position_axis.set_ylabel("base y / mm")
        position_axis.set_zlabel("base z / mm")
        position_axis.legend(frameon=False, fontsize=8, ncol=2)

        rotation_axis = figure.add_subplot(grid[1, column], projection="3d")
        for index, atom in enumerate(ATOMS[6:]):
            rot = np.asarray(rows[atom]["tcp_rotation_vector_deg"])
            rotation_axis.plot(rot[:, 0], rot[:, 1], rot[:, 2], color=colors(index), label=atom)
            rotation_axis.scatter(*rot[-1], color=colors(index), s=22)
        rotation_axis.set_title(f"{model_name}: rotation prompts")
        rotation_axis.set_xlabel("world rx / deg")
        rotation_axis.set_ylabel("world ry / deg")
        rotation_axis.set_zlabel("world rz / deg")
        rotation_axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.suptitle(
        f"Frequent state cluster {key} · ep{cluster['episode']} f{cluster['frame']} · "
        f"estimated count={cluster['estimated_full_count']} · labels seen={cluster['label_diversity']}\n"
        "same image/state/noise for all 12 prompts and both checkpoints · TCP=0.20 m",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_summary(model_outputs: dict[str, list[dict]], output: Path):
    model_names = list(model_outputs)
    clusters = sorted({row["cluster_key"] for rows in model_outputs.values() for row in rows})
    figure, axes = plt.subplots(len(clusters), len(model_names), figsize=(16, 3.8 * len(clusters)), constrained_layout=True)
    axes = np.asarray(axes).reshape(len(clusters), len(model_names))
    for row_index, cluster in enumerate(clusters):
        for column, model_name in enumerate(model_names):
            rows = {row["atom"]: row for row in model_outputs[model_name] if row["cluster_key"] == cluster}
            matrix = np.stack([np.asarray(rows[atom]["endpoint_twist"]) for atom in ATOMS])
            normalized = matrix / np.asarray([50.0, 50.0, 50.0, 15.0, 15.0, 15.0])
            axis = axes[row_index, column]
            image = axis.imshow(normalized, cmap="coolwarm", vmin=-2.0, vmax=2.0, aspect="auto")
            axis.set_xticks(range(6), ("x mm", "y mm", "z mm", "rx°", "ry°", "rz°"))
            axis.set_yticks(range(12), ATOMS, fontsize=8)
            axis.set_title(f"{cluster} · {model_name}")
            for i in range(12):
                for j in range(6):
                    axis.text(j, i, f"{matrix[i,j]:.0f}", ha="center", va="center", fontsize=6,
                              color="white" if abs(normalized[i,j]) > 1.1 else "black")
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.7, label="normalized endpoint response")
    figure.suptitle("12-prompt endpoint TCP twist response at frequent state clusters", fontsize=16, fontweight="bold")
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH; repeat twice")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--state-stride", type=int, default=5)
    parser.add_argument("--cluster-count", type=int, default=48)
    parser.add_argument("--clusters-per-arm", type=int, default=2)
    parser.add_argument(
        "--arms", choices=("right", "left", "both"), default="right",
        help="Evaluate right, left, or both arm state clusters.",
    )
    parser.add_argument("--cluster-report", type=Path)
    parser.add_argument("--noise-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    checkpoints = dict(item.split("=", 1) for item in args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = AtomicPi05Config(max_token_len=250, fast_action_ce_loss_weight=0.0)
    dataset = build_atomic_dataset(
        (args.dataset_root,), norm_assets_dir="/mnt/cunchu/zjq/target",
        norm_asset_id="openpi_norm_compact_accepted_v3", action_horizon=config.action_horizon,
        max_token_len=config.max_token_len, include_fast=False,
    )
    arms = ("right", "left") if args.arms == "both" else (args.arms,)
    clusters = (
        _representatives_from_report(args.cluster_report, args.clusters_per_arm, arms)
        if args.cluster_report
        else _cluster_representatives(
            dataset, stride=args.state_stride, cluster_count=args.cluster_count,
            clusters_per_arm=args.clusters_per_arm, seed=args.seed,
        )
    )
    rows = []
    for cluster in clusters:
        for atom in PROMPT_MODES:
            sample = dataset[cluster["dataset_index"]]
            metadata = dataset._raw.metadata(cluster["dataset_index"])  # noqa: SLF001
            prompt = (
                str(metadata["global_prompt"])
                if atom == "__neutral__"
                else _prompt(cluster["arm"], atom)
            )
            rows.append({
                "cluster_key": f"{cluster['arm']}_c{cluster['cluster']}",
                "arm": cluster["arm"], "atom": atom,
                "prompt": prompt,
                "dataset_index": cluster["dataset_index"],
                "metadata": metadata,
                "sample": sample,
            })
    samples = [row["sample"] for row in rows]
    batch = atomic_collate(samples)
    observation_np, _ = batch_to_observation(batch)
    observation = jax.tree.map(jnp.asarray, observation_np)
    tokenizer = _paligemma_tokenizer(config.max_token_len)
    tokenized = [
        tokenizer.tokenize(row["prompt"], np.asarray(sample["state"]))
        for row, sample in zip(rows, samples, strict=True)
    ]
    tokens = jnp.asarray(np.stack([item[0] for item in tokenized]))
    masks = jnp.asarray(np.stack([item[1] for item in tokenized]))
    rng = np.random.default_rng(args.seed)
    noise_rows = []
    for _ in range(args.noise_repeats):
        cluster_noise = rng.standard_normal((len(clusters), 50, config.action_dim), dtype=np.float32)
        noise_rows.append(np.repeat(cluster_noise, len(PROMPT_MODES), axis=0))

    model_outputs = {}
    for name, checkpoint in checkpoints.items():
        model_outputs[name] = _evaluate_checkpoint(
            name, Path(checkpoint), config, observation, tokens, masks, batch, rows,
            dataset, args.dataset_root, noise_rows, args.noise_repeats,
        )
    report = {
        "checkpoints": checkpoints,
        "tcp_offset_m": 0.20,
        "state_clustering": {
            "stride": args.state_stride, "cluster_count_per_arm": args.cluster_count,
            "clusters_per_arm": args.clusters_per_arm, "clusters": clusters,
        },
        "metrics": {name: _metrics(outputs) for name, outputs in model_outputs.items()},
        "outputs": model_outputs,
    }
    (args.output_dir / "state_cluster_atomic_sweep.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _plot_summary(model_outputs, args.output_dir / "state_cluster_atomic_sweep_heatmap.png")
    for cluster in clusters:
        key = f"{cluster['arm']}_c{cluster['cluster']}"
        _plot_cluster(cluster, model_outputs, args.output_dir / f"tcp_sweep_{key}.png")
    print(json.dumps({"clusters": clusters, "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
