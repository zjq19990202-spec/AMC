#!/usr/bin/env python3
"""Evaluate single/dual atomic steering over many frequent bimanual state clusters."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from openpi.models import model as _model

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


PHRASES = {
    "move_x_pos": "move forward along base-frame +x",
    "move_x_neg": "move backward along base-frame -x",
    "move_y_pos": "move leftward along base-frame +y",
    "move_y_neg": "move rightward along base-frame -y",
    "move_z_pos": "move upward along base-frame +z",
    "move_z_neg": "move downward along base-frame -z",
    "rotate_x_pos": "rotate positively about the base-frame x axis",
    "rotate_x_neg": "rotate negatively about the base-frame x axis",
    "rotate_y_pos": "rotate positively about the base-frame y axis",
    "rotate_y_neg": "rotate negatively about the base-frame y axis",
    "rotate_z_pos": "rotate positively about the base-frame z axis",
    "rotate_z_neg": "rotate negatively about the base-frame z axis",
}


def _opposite(atom: str) -> str:
    if atom.endswith("_pos"):
        return atom[:-4] + "_neg"
    if atom.endswith("_neg"):
        return atom[:-4] + "_pos"
    raise ValueError(atom)


def _component(atom: str) -> tuple[int, float]:
    family, axis, sign_name = atom.split("_")
    index = "xyz".index(axis) + (3 if family == "rotate" else 0)
    return index, 1.0 if sign_name == "pos" else -1.0


def _mode_key(atoms: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(atoms))


def _pair_key(atoms: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    direct = _mode_key(atoms)
    reverse = _mode_key(tuple(_opposite(atom) for atom in atoms))
    return min(direct, reverse), max(direct, reverse)


def _valid_mode(atoms: tuple[str, ...]) -> bool:
    if len(atoms) not in (1, 2) or any(atom not in PHRASES for atom in atoms):
        return False
    coordinates = [_component(atom)[0] for atom in atoms]
    return len(set(coordinates)) == len(coordinates)


def _prompt(arm: str, atoms: tuple[str, ...]) -> str:
    other = "left" if arm == "right" else "right"
    actions = " and simultaneously ".join(PHRASES[atom] for atom in atoms)
    return (
        f"{arm.capitalize()} arm: {actions}. "
        f"{other.capitalize()} arm: keep the {other} TCP stationary in its current base-frame pose."
    )


def _select_tests(
    report: dict,
    *,
    minimum_mode_count: int,
    minimum_mode_episodes: int,
    minimum_modes_per_cluster: int,
    minimum_cluster_samples: int,
    minimum_cluster_episodes: int,
    clusters_per_arm: int,
    max_single_per_cluster: int,
    max_dual_per_cluster: int,
) -> tuple[list[dict], list[dict]]:
    selected_clusters = []
    tests = []
    for arm in ("right", "left"):
        accepted = 0
        cluster_source = report["arms"][arm].get(
            "clusters", report["arms"][arm]["top_candidates"]
        )
        for cluster in sorted(cluster_source, key=lambda row: int(row["cluster"])):
            if (
                int(cluster["samples_3hz"]) < minimum_cluster_samples
                or int(cluster["episodes"]) < minimum_cluster_episodes
            ):
                continue
            aggregated: dict[tuple[str, ...], int] = defaultdict(int)
            for row in cluster["mode_counts"]:
                atoms = _mode_key(tuple(row["labels"]))
                if (
                    int(row["samples"]) >= minimum_mode_count
                    and int(row.get("episodes", minimum_mode_episodes)) >= minimum_mode_episodes
                    and _valid_mode(atoms)
                ):
                    aggregated[atoms] += int(row["samples"])
            # A mode and its complete sign reversal form one causal steering test.
            deduplicated: dict[tuple, tuple[tuple[str, ...], int]] = {}
            for atoms, count in aggregated.items():
                key = _pair_key(atoms)
                current = deduplicated.get(key)
                if current is None or count > current[1]:
                    deduplicated[key] = (atoms, count)
            modes = sorted(deduplicated.values(), key=lambda item: item[1], reverse=True)
            if len(modes) < minimum_modes_per_cluster:
                continue
            singles = [item for item in modes if len(item[0]) == 1][:max_single_per_cluster]
            duals = [item for item in modes if len(item[0]) == 2][:max_dual_per_cluster]
            chosen = singles + duals
            if len(chosen) < minimum_modes_per_cluster:
                continue
            representative = cluster.get("center_representative")
            if representative is None:
                representative = min(cluster["representatives"], key=lambda row: row["rms_to_center"])
            cluster_key = f"{arm}_c{int(cluster['cluster']):03d}"
            selected_clusters.append(
                {
                    "cluster_key": cluster_key,
                    "arm": arm,
                    "cluster": int(cluster["cluster"]),
                    "samples_3hz": int(cluster["samples_3hz"]),
                    "episodes": int(cluster["episodes"]),
                    "supported_atom_count": int(cluster["supported_atom_count"]),
                    "available_supported_modes": len(modes),
                    "dataset_index": int(representative["dataset_index"]),
                    "episode": int(representative["episode"]),
                    "frame": int(representative["frame"]),
                    "rms_to_center": float(representative["rms_to_center"]),
                }
            )
            for atoms, support in chosen:
                test_key = cluster_key + "__" + "+".join(atoms)
                tests.append(
                    {
                        "test_key": test_key,
                        "cluster_key": cluster_key,
                        "arm": arm,
                        "kind": "single" if len(atoms) == 1 else "dual",
                        "atoms": list(atoms),
                        "reverse_atoms": [_opposite(atom) for atom in atoms],
                        "support": int(support),
                        "dataset_index": int(representative["dataset_index"]),
                    }
                )
            accepted += 1
            if clusters_per_arm > 0 and accepted >= clusters_per_arm:
                break
    return selected_clusters, tests


def _trajectory(fk, raw_state: np.ndarray, actions: np.ndarray, arm: str) -> np.ndarray:
    q_slice = slice(8, 15) if arm == "right" else slice(0, 7)
    p0, r0 = fk.pose(raw_state[q_slice])
    poses = [fk.pose(row[q_slice]) for row in actions]
    positions = np.stack([pose[0] for pose in poses])
    rotations = np.stack([pose[1] for pose in poses])
    rotation_vectors = np.stack([_world_rotation_vector_deg(r0, rotation) for rotation in rotations])
    return np.concatenate(
        [
            np.zeros((1, 6), dtype=np.float64),
            np.concatenate([(positions - p0) * 1000.0, rotation_vectors], axis=-1),
        ],
        axis=0,
    )


def _evaluate_checkpoint(
    name: str,
    checkpoint: Path,
    config: AtomicPi05Config,
    dataset,
    dataset_root: Path,
    row_specs: list[dict],
    noises: np.ndarray,
    *,
    batch_size: int,
    stored_steps: tuple[int, ...] | None = None,
    norm_assets_dir: Path = Path("/mnt/cunchu/zjq/target"),
    norm_asset_id: str = "openpi_norm_compact_accepted_v3",
) -> list[dict]:
    params = _model.restore_params(checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    decode = _output_transform(
        dataset_root,
        config,
        norm_assets_dir=norm_assets_dir,
        norm_asset_id=norm_asset_id,
    )
    _fk_eval.TCP_LOCAL_Z_OFFSET_M = 0.20
    fk = _fk_eval.SimpleCR1FK(Path("assets/cr1_right_arm_tcp_red/urdf/CR1ARMR_TCP_RED.urdf"))
    tokenizer = _paligemma_tokenizer(config.max_token_len)
    sample_cache = {index: dataset[index] for index in {row["dataset_index"] for row in row_specs}}
    metadata_cache = {
        index: dataset._raw.metadata(index)  # noqa: SLF001
        for index in sample_cache
    }
    outputs = []
    for start in range(0, len(row_specs), batch_size):
        rows = row_specs[start : start + batch_size]
        real_count = len(rows)
        if real_count < batch_size:
            rows = rows + [rows[-1]] * (batch_size - real_count)
        samples = [sample_cache[row["dataset_index"]] for row in rows]
        batch = atomic_collate(samples)
        observation_np, _ = batch_to_observation(batch)
        observation = jax.tree.map(jnp.asarray, observation_np)
        tokenized = [
            tokenizer.tokenize(row["prompt"], np.asarray(sample["state"]))
            for row, sample in zip(rows, samples, strict=True)
        ]
        tokens = jnp.asarray(np.stack([item[0] for item in tokenized]))
        masks = jnp.asarray(np.stack([item[1] for item in tokenized]))
        chunk_noises = noises[start : start + real_count]
        if real_count < batch_size:
            chunk_noises = np.concatenate(
                [chunk_noises, np.repeat(chunk_noises[-1:], batch_size - real_count, axis=0)], axis=0
            )
        predictions = np.asarray(
            jax.device_get(
                _sample_global(model, observation, jnp.asarray(chunk_noises), tokens, masks, None)
            )
        )
        for local_index, (row, prediction) in enumerate(zip(rows[:real_count], predictions[:real_count], strict=True)):
            metadata = metadata_cache[row["dataset_index"]]
            decoded = decode(
                np.asarray(batch["state"][local_index]),
                np.asarray(metadata["raw_state"]),
                prediction,
            )["actions"]
            trajectory = _trajectory(
                fk, np.asarray(metadata["raw_state"]), decoded, row["arm"]
            )
            outputs.append(
                {
                    **{key: value for key, value in row.items() if key != "prompt"},
                    "prompt": row["prompt"],
                    "trajectory_twist": (
                        trajectory.tolist()
                        if stored_steps is None
                        else {str(step): trajectory[step].tolist() for step in stored_steps}
                    ),
                }
            )
        print(f"{name}: {min(start + batch_size, len(row_specs))}/{len(row_specs)} rows", flush=True)
    del model, params
    gc.collect()
    jax.clear_caches()
    return outputs


def _summarize(outputs: list[dict]) -> tuple[list[dict], list[dict]]:
    paired = defaultdict(dict)
    for row in outputs:
        paired[(row["test_key"], row["repeat"])][row["variant"]] = row
    records = []
    for (test_key, repeat), variants in paired.items():
        original, reverse = variants["original"], variants["reverse"]
        original_trajectory = np.asarray(original["trajectory_twist"])
        reverse_trajectory = np.asarray(reverse["trajectory_twist"])
        components = [_component(atom) for atom in original["atoms"]]
        for step in (25, 50):
            original_hits = [sign * original_trajectory[step, index] > 0 for index, sign in components]
            reverse_hits = [-sign * reverse_trajectory[step, index] > 0 for index, sign in components]
            order_hits = [
                sign * (original_trajectory[step, index] - reverse_trajectory[step, index]) > 0
                for index, sign in components
            ]
            records.append(
                {
                    "test_key": test_key,
                    "cluster_key": original["cluster_key"],
                    "arm": original["arm"],
                    "kind": original["kind"],
                    "atoms": "+".join(original["atoms"]),
                    "support": original["support"],
                    "repeat": repeat,
                    "step": step,
                    "original_success": bool(all(original_hits)),
                    "reverse_success": bool(all(reverse_hits)),
                    "strict_direction_successes": int(all(original_hits)) + int(all(reverse_hits)),
                    "strict_direction_trials": 2,
                    "bidirectional_success": bool(all(original_hits) and all(reverse_hits)),
                    "pair_order_success": bool(all(order_hits)),
                    "component_order_successes": int(sum(order_hits)),
                    "component_order_trials": len(order_hits),
                }
            )
    summary = []
    grouping = defaultdict(list)
    for row in records:
        grouping[(row["arm"], row["kind"], row["step"])].append(row)
        grouping[("both", row["kind"], row["step"])].append(row)
        grouping[(row["arm"], "all", row["step"])].append(row)
        grouping[("both", "all", row["step"])].append(row)
    for (arm, kind, step), rows in sorted(grouping.items()):
        cluster_rates = []
        for cluster_key in sorted({row["cluster_key"] for row in rows}):
            cluster_rows = [row for row in rows if row["cluster_key"] == cluster_key]
            cluster_rates.append(
                sum(row["strict_direction_successes"] for row in cluster_rows)
                / sum(row["strict_direction_trials"] for row in cluster_rows)
            )
        summary.append(
            {
                "arm": arm,
                "kind": kind,
                "step": step,
                "clusters": len(cluster_rates),
                "paired_tests_with_repeats": len(rows),
                "strict_micro_rate": sum(row["strict_direction_successes"] for row in rows)
                / sum(row["strict_direction_trials"] for row in rows),
                "strict_macro_cluster_rate": float(np.mean(cluster_rates)),
                "bidirectional_pair_rate": float(np.mean([row["bidirectional_success"] for row in rows])),
                "causal_pair_order_rate": float(np.mean([row["pair_order_success"] for row in rows])),
                "causal_component_order_rate": sum(row["component_order_successes"] for row in rows)
                / sum(row["component_order_trials"] for row in rows),
            }
        )
    return records, summary


def _write_summary_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(summaries: dict[str, list[dict]], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    model_names = list(summaries)
    colors = {model_names[0]: "#2f6f9f", model_names[1]: "#d9793f"}
    for row_index, step in enumerate((25, 50)):
        for column, (metric, title) in enumerate(
            (("strict_micro_rate", "Strict direction from initial TCP"),
             ("causal_pair_order_rate", "Original vs reversed prompt ordering"))
        ):
            axis = axes[row_index, column]
            categories = [(arm, kind) for arm in ("right", "left", "both") for kind in ("single", "dual")]
            x = np.arange(len(categories))
            width = 0.36
            for model_index, model in enumerate(model_names):
                table = {(row["arm"], row["kind"], row["step"]): row for row in summaries[model]}
                values = [table[(arm, kind, step)][metric] for arm, kind in categories]
                axis.bar(x + (model_index - 0.5) * width, values, width, label=model, color=colors[model])
            axis.set_xticks(x, [f"{arm}\n{kind}" for arm, kind in categories])
            axis.set_ylim(0, 1.04)
            axis.set_ylabel("rate")
            axis.set_title(f"Step {step} · {title}")
            axis.grid(axis="y", alpha=0.2)
            axis.legend(frameon=False)
    figure.suptitle("Atomic prompt steering over many frequent state clusters", fontsize=16, fontweight="bold")
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cluster-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-mode-count", type=int, default=10)
    parser.add_argument("--minimum-mode-episodes", type=int, default=2)
    parser.add_argument("--minimum-modes-per-cluster", type=int, default=4)
    parser.add_argument("--minimum-cluster-samples", type=int, default=100)
    parser.add_argument("--minimum-cluster-episodes", type=int, default=10)
    parser.add_argument("--clusters-per-arm", type=int, default=12)
    parser.add_argument("--max-single-per-cluster", type=int, default=4)
    parser.add_argument("--max-dual-per-cluster", type=int, default=4)
    parser.add_argument("--noise-repeats", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = dict(item.split("=", 1) for item in args.checkpoint)
    cluster_report = json.loads(args.cluster_report.read_text(encoding="utf-8"))
    clusters, tests = _select_tests(
        cluster_report,
        minimum_mode_count=args.minimum_mode_count,
        minimum_mode_episodes=args.minimum_mode_episodes,
        minimum_modes_per_cluster=args.minimum_modes_per_cluster,
        minimum_cluster_samples=args.minimum_cluster_samples,
        minimum_cluster_episodes=args.minimum_cluster_episodes,
        clusters_per_arm=args.clusters_per_arm,
        max_single_per_cluster=args.max_single_per_cluster,
        max_dual_per_cluster=args.max_dual_per_cluster,
    )
    if not tests:
        raise RuntimeError("no eligible state clusters/modes")
    config = AtomicPi05Config(max_token_len=250, fast_action_ce_loss_weight=0.0)
    dataset = build_atomic_dataset(
        (args.dataset_root,),
        norm_assets_dir="/mnt/cunchu/zjq/target",
        norm_asset_id="openpi_norm_compact_accepted_v3",
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        include_fast=False,
    )
    row_specs = []
    rng = np.random.default_rng(args.seed)
    noise_by_key = {}
    for test in tests:
        for repeat in range(args.noise_repeats):
            noise_key = (test["test_key"], repeat)
            noise_by_key[noise_key] = rng.standard_normal(
                (config.action_horizon, config.action_dim), dtype=np.float32
            )
            for variant, atoms in (
                ("original", tuple(test["atoms"])),
                ("reverse", tuple(test["reverse_atoms"])),
            ):
                row_specs.append(
                    {
                        **test,
                        "repeat": repeat,
                        "variant": variant,
                        "prompt": _prompt(test["arm"], atoms),
                        "noise_key": noise_key,
                    }
                )
    noises = np.stack([noise_by_key[row["noise_key"]] for row in row_specs])
    # JSON cannot encode tuple keys; they are only needed to align shared noise.
    for row in row_specs:
        row.pop("noise_key")
    model_outputs = {}
    model_records = {}
    model_summaries = {}
    for name, checkpoint in checkpoints.items():
        outputs = _evaluate_checkpoint(
            name,
            Path(checkpoint),
            config,
            dataset,
            args.dataset_root,
            row_specs,
            noises,
            batch_size=args.batch_size,
            norm_assets_dir=args.norm_assets_dir,
            norm_asset_id=args.norm_asset_id,
        )
        records, summary = _summarize(outputs)
        model_outputs[name] = outputs
        model_records[name] = records
        model_summaries[name] = summary
        _write_summary_csv(summary, args.output_dir / f"{name}_steering_summary.csv")
    report = {
        "checkpoints": checkpoints,
        "selection": {
            "minimum_mode_count": args.minimum_mode_count,
            "minimum_mode_episodes": args.minimum_mode_episodes,
            "minimum_modes_per_cluster": args.minimum_modes_per_cluster,
            "minimum_cluster_samples": args.minimum_cluster_samples,
            "minimum_cluster_episodes": args.minimum_cluster_episodes,
            "clusters_per_arm_limit": args.clusters_per_arm,
            "noise_repeats": args.noise_repeats,
            "clusters": clusters,
            "tests": tests,
        },
        "metric_definition": {
            "strict_micro_rate": "fraction of original/reversed prompts whose every target component has the requested sign from the initial TCP",
            "strict_macro_cluster_rate": "strict rate averaged equally over clusters",
            "bidirectional_pair_rate": "both original and reversed prompt strictly succeed",
            "causal_pair_order_rate": "every target component orders original ahead of its complete sign reversal under identical observation/state/noise",
        },
        "summary": model_summaries,
        "records": model_records,
        "outputs": model_outputs,
    }
    (args.output_dir / "many_cluster_single_dual_steering.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _plot(model_summaries, args.output_dir / "many_cluster_single_dual_steering_rates.png")
    print(json.dumps({"clusters": clusters, "test_count": len(tests), "summary": model_summaries}, indent=2))


if __name__ == "__main__":
    main()
