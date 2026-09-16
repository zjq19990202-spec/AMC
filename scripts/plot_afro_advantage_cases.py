#!/usr/bin/env python3
"""Plot selected qualitative AFRO-vs-PI0.5 fruit steering cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from atomic_latent_vla.pi05.training_data import atomic_collate, batch_to_observation, build_atomic_dataset
from evaluate_fruit_target_switch import _find_row
from evaluate_global_episode_chunks import _endpoint_tcp_from_actions


CASES = ((124, 770, "green bitter melon"), (142, 0, "orange"), (107, 570, "yellow pear"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--plain-json", type=Path, required=True)
    ap.add_argument("--afro-npz", type=Path, required=True)
    ap.add_argument("--distance-json", type=Path, required=True)
    ap.add_argument("--target-json", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)

    plain = json.loads(args.plain_json.read_text())
    prows = {(r["episode"], r["frame"]): r for r in plain["rows"]}
    afro = np.load(args.afro_npz, allow_pickle=True)
    distance = json.loads(args.distance_json.read_text())
    drows = {(r["episode"], r["frame"], r["fruit"]): r for r in distance["rows"]}
    target_rows = json.loads(args.target_json.read_text())["rows"]
    targets = {(r["episode"], r["frame"], r["fruit"]): np.asarray(r["target_xyz_m"]) for r in target_rows}
    scene_targets = {}
    for item in target_rows:
        scene_targets.setdefault((item["episode"], item["frame"]), []).append(item)
    table = pq.read_table(args.dataset_root / "data/chunk-000/file-000.parquet", columns=["episode_index", "frame_index", "index"])
    eps, frames, indices = np.asarray(table["episode_index"]), np.asarray(table["frame_index"]), np.asarray(table["index"])
    tcp = np.load(args.dataset_root / "meta/tcp_pose_bimanual_base_tcp200.npy", mmap_mode="r")
    dataset = build_atomic_dataset((args.dataset_root,), norm_assets_dir="/mnt/cunchu/zjq/target", norm_asset_id="openpi_norm_union2375_allframes_v1", action_horizon=50, max_token_len=192, include_fast=False, atomic_composition_sidecar="fk_horizon_3hz_gate_top5_stay_v2", pad_subtask_horizon=True)

    fig = plt.figure(figsize=(15, 15), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, width_ratios=(1.0, 1.35))
    report = []
    for row_index, (ep, frame, fruit) in enumerate(CASES):
        info = drows[(ep, frame, fruit)]
        target = targets[(ep, frame, fruit)]
        dataset_index, *_ = _find_row(dataset, ep, frame)
        obs, _ = batch_to_observation(atomic_collate([dataset[dataset_index]]))
        image = np.clip((np.asarray(obs.images["base_0_rgb"])[0] + 1.0) * 127.5, 0, 255).astype(np.uint8)
        ax_img = fig.add_subplot(grid[row_index, 0]); ax_img.imshow(image); ax_img.axis("off")
        ax_img.set_title(f"episode {ep}, frame {frame}\nSUBtask: Move toward the {fruit} and grasp it", fontsize=11)

        pr = prows[(ep, frame)]; pi = next(i for i, text in enumerate(pr["prompts"]) if fruit in text.lower())
        plain_arm = info["plain_moving_arm"]; afro_arm = info["afro_moving_arm"]
        plain_traj = np.asarray(pr[f"{plain_arm}_tcp_trajectories_m"])[pi]
        prefix = f"episode_{ep:06d}_frame_{frame:06d}"; names = list(afro[prefix + "_prompt_names"]); ai = names.index(fruit)
        afro_traj = _endpoint_tcp_from_actions(None, 0, afro[prefix + "_prediction_actions"][ai], afro_arm)
        source_row = int(np.flatnonzero((eps == ep) & (frames == frame))[0])
        starts = {"left": np.asarray(tcp[int(indices[source_row]), 0:3]), "right": np.asarray(tcp[int(indices[source_row]), 24:27])}
        pfull = np.concatenate([starts[plain_arm][None], plain_traj]); afull = np.concatenate([starts[afro_arm][None], afro_traj])
        pd = np.linalg.norm(pfull - target, axis=1) * 1000; ad = np.linalg.norm(afull - target, axis=1) * 1000

        ax = fig.add_subplot(grid[row_index, 1], projection="3d")
        ax.plot(*pfull.T, color="#2563eb", lw=2.2, label=f"PI0.5 25K ({plain_arm})")
        ax.plot(*afull.T, color="#ef4444", lw=2.5, label=f"AFRO 50K + correct ZM ({afro_arm})")
        ax.scatter(*pfull[0], color="#2563eb", marker="o", s=55); ax.scatter(*afull[0], color="#ef4444", marker="o", s=55)
        ax.scatter(*pfull[int(pd.argmin())], color="#2563eb", marker="x", s=75)
        ax.scatter(*afull[int(ad.argmin())], color="#ef4444", marker="x", s=75)
        all_target_xyz = []
        for target_item in scene_targets[(ep, frame)]:
            xyz = np.asarray(target_item["target_xyz_m"])
            all_target_xyz.append(xyz)
            is_requested = target_item["fruit"] == fruit
            marker = "*" if is_requested else ("s" if target_item["target_kind"] == "placed_box_center" else "o")
            color = "#16a34a" if is_requested else ("#a855f7" if target_item["target_kind"] == "placed_box_center" else "#f59e0b")
            ax.scatter(*xyz, color=color, marker=marker, s=190 if is_requested else 48, zorder=8)
            label = target_item["fruit"] + (" (in box)" if target_item["target_kind"] == "placed_box_center" else "")
            ax.text(xyz[0], xyz[1], xyz[2] + 0.012, label, fontsize=7, color=color, zorder=9)
        ax.scatter([], [], [], color="#16a34a", marker="*", s=140, label="requested fruit")
        pts = np.concatenate([pfull, afull, np.asarray(all_target_xyz)], axis=0); lo, hi = pts.min(0), pts.max(0); center=(lo+hi)/2; radius=max((hi-lo).max()/2, .05)
        ax.set_xlim(center[0]-radius,center[0]+radius); ax.set_ylim(center[1]-radius,center[1]+radius); ax.set_zlim(center[2]-radius,center[2]+radius); ax.set_box_aspect((1,1,1))
        ax.set_xlabel("base x (m)"); ax.set_ylabel("base y (m)"); ax.set_zlabel("base z (m)")
        ax.set_title(f"PI0.5: {pd[0]:.0f}→{pd[-1]:.0f} mm (min {pd.min():.0f})\nAFRO: {ad[0]:.0f}→{ad[-1]:.0f} mm (min {ad.min():.0f})")
        ax.legend(loc="best", fontsize=8)
        report.append({"episode":ep,"frame":frame,"fruit":fruit,"plain_arm":plain_arm,"afro_arm":afro_arm,"plain_initial_mm":float(pd[0]),"plain_final_mm":float(pd[-1]),"plain_min_mm":float(pd.min()),"plain_closest_step":int(pd.argmin()),"afro_initial_mm":float(ad[0]),"afro_final_mm":float(ad[-1]),"afro_min_mm":float(ad.min()),"afro_closest_step":int(ad.argmin())})
    fig.suptitle("Qualitative AFRO advantages: same observation, fruit SUBtask and flow-noise seed", fontsize=16)
    fig.savefig(args.output_dir / "afro_advantage_three_cases.png", dpi=200, bbox_inches="tight")
    (args.output_dir / "cases.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__": main()
