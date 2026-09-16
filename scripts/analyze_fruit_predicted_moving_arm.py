#!/usr/bin/env python3
"""Score fruit steering using each model's own predicted moving arm.

Fruit prompts do not identify an arm.  For every model/prompt pair, the arm
whose horizon endpoint moves farther from its initial TCP is therefore used
for target-distance scoring.  Target locations are generated independently
from future grasp contacts or earlier placement endpoints.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_global_episode_chunks import _endpoint_tcp_from_actions


FRUITS = ("carrot", "orange", "green bitter melon", "green radish", "yellow pear", "banana", "red chili pepper")


def fruit_in(text: str) -> str | None:
    text = text.lower()
    aliases = {"yellow banana": "banana"}
    for alias, canonical in aliases.items():
        if alias in text:
            return canonical
    for fruit in sorted(FRUITS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(fruit)}\b", text):
            return fruit
    return None


def tcp(actions: np.ndarray, arm: str) -> np.ndarray:
    return np.asarray(_endpoint_tcp_from_actions(None, 0, np.asarray(actions), arm), dtype=float)


def model_metrics(start_actions: np.ndarray, predicted_actions: np.ndarray, target: np.ndarray, arm_rule: str) -> dict:
    starts = {arm: tcp(start_actions[None], arm)[0] for arm in ("left", "right")}
    trajectories = {arm: tcp(predicted_actions, arm) for arm in ("left", "right")}
    motion = {arm: float(np.linalg.norm(xyz[-1] - starts[arm]) * 1000) for arm, xyz in trajectories.items()}
    if arm_rule == "largest_motion":
        arm = max(motion, key=motion.get)
    elif arm_rule == "nearest_final":
        arm = min(trajectories, key=lambda name: np.linalg.norm(trajectories[name][-1] - target))
    else:
        raise ValueError(arm_rule)
    xyz = trajectories[arm]
    initial = float(np.linalg.norm(starts[arm] - target) * 1000)
    distances = np.linalg.norm(xyz - target, axis=1) * 1000
    displacement = xyz[-1] - starts[arm]
    target_direction = target - starts[arm]
    denom = np.linalg.norm(displacement) * np.linalg.norm(target_direction)
    cosine = float(np.dot(displacement, target_direction) / denom) if denom > 1e-12 else 0.0
    return {
        "moving_arm": arm,
        "left_motion_mm": motion["left"],
        "right_motion_mm": motion["right"],
        "initial_mm": initial,
        "final_mm": float(distances[-1]),
        "reduction_mm": initial - float(distances[-1]),
        "direction_cosine": cosine,
        "min_distance_mm": float(distances.min()),
        "closest_step": int(distances.argmin() + 1),
        "ever_closer": bool(distances.min() < initial),
    }


def aggregate(rows: list[dict], prefix: str) -> dict:
    return {
        "n": len(rows),
        "mean_final_mm": float(np.mean([r[f"{prefix}_final_mm"] for r in rows])),
        "median_final_mm": float(np.median([r[f"{prefix}_final_mm"] for r in rows])),
        "mean_reduction_mm": float(np.mean([r[f"{prefix}_reduction_mm"] for r in rows])),
        "median_reduction_mm": float(np.median([r[f"{prefix}_reduction_mm"] for r in rows])),
        "closer_than_start": int(sum(r[f"{prefix}_reduction_mm"] > 0 for r in rows)),
        "positive_direction": int(sum(r[f"{prefix}_direction_cosine"] > 0 for r in rows)),
        "mean_direction_cosine": float(np.mean([r[f"{prefix}_direction_cosine"] for r in rows])),
        "ever_closer": int(sum(r[f"{prefix}_ever_closer"] for r in rows)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain-json", type=Path, required=True)
    ap.add_argument("--afro-npz", type=Path, required=True)
    ap.add_argument("--targets-json", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bootstrap-seed", type=int, default=20260830)
    ap.add_argument("--bootstrap-repeats", type=int, default=10000)
    ap.add_argument("--arm-rule", choices=("largest_motion", "nearest_final"), default="largest_motion")
    args = ap.parse_args()

    plain = json.loads(args.plain_json.read_text())
    targets = json.loads(args.targets_json.read_text())["rows"]
    archive = np.load(args.afro_npz, allow_pickle=True)
    plain_rows = {(int(r["episode"]), int(r["frame"])): r for r in plain["rows"]}

    rows = []
    for target_row in targets:
        ep, frame, fruit = int(target_row["episode"]), int(target_row["frame"]), target_row["fruit"]
        key = f"episode_{ep:06d}_frame_{frame:06d}"
        names = [str(x) for x in archive[f"{key}_prompt_names"]]
        actions = archive[f"{key}_prediction_actions"]
        candidates = [i for i, name in enumerate(names) if fruit_in(name) == fruit]
        if not candidates:
            raise KeyError(f"AFRO prompt missing for {(ep, frame, fruit)}; names={names}")
        # Prefer the explicitly requested canonical target when a differently
        # spelled native prompt was appended as an eighth variant.
        exact = [i for i in candidates if names[i].lower() == fruit]
        ai = exact[0] if exact else candidates[0]
        raw_state = np.asarray(archive[f"{key}_raw_state"], dtype=float)

        pr = plain_rows[(ep, frame)]
        pcandidates = [i for i, prompt in enumerate(pr["prompts"]) if fruit_in(prompt) == fruit]
        if len(pcandidates) != 1:
            raise ValueError(f"plain prompt ambiguity for {(ep, frame, fruit)}: {pcandidates}")
        pi = pcandidates[0]
        plain_actions = np.asarray(pr["decoded_actions"][pi], dtype=float)
        target = np.asarray(target_row["target_xyz_m"], dtype=float)
        pm = model_metrics(raw_state, plain_actions, target, args.arm_rule)
        am = model_metrics(raw_state, np.asarray(actions[ai], dtype=float), target, args.arm_rule)
        row = {k: target_row[k] for k in ("episode", "frame", "fruit", "target_kind")}
        row.update({f"plain_{k}": v for k, v in pm.items()})
        row.update({f"afro_{k}": v for k, v in am.items()})
        rows.append(row)

    diff = np.asarray([r["plain_final_mm"] - r["afro_final_mm"] for r in rows])
    frames = sorted({(r["episode"], r["frame"]) for r in rows})
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["episode"], r["frame"])].append(r["plain_final_mm"] - r["afro_final_mm"])
    frame_means = np.asarray([np.mean(grouped[f]) for f in frames])
    rng = np.random.default_rng(args.bootstrap_seed)
    boot = np.mean(rng.choice(frame_means, size=(args.bootstrap_repeats, len(frame_means)), replace=True), axis=1)

    by_fruit = {}
    for fruit in FRUITS:
        selected = [r for r in rows if r["fruit"] == fruit]
        fd = np.asarray([r["plain_final_mm"] - r["afro_final_mm"] for r in selected])
        by_fruit[fruit] = {"n": len(selected), "plain_mean_final_mm": float(np.mean([r["plain_final_mm"] for r in selected])), "afro_mean_final_mm": float(np.mean([r["afro_final_mm"] for r in selected])), "afro_gain_mm": float(fd.mean()), "afro_wins": int(np.sum(fd > 0))}

    report = {
        "contract": {"source_observations": len(frames), "prompt_target_pairs": len(rows), "arm_rule": args.arm_rule, "horizon": 50, "bootstrap_unit": "source observation"},
        "summary": {
            "plain": aggregate(rows, "plain"),
            "afro": aggregate(rows, "afro"),
            "paired": {"afro_wins": int(np.sum(diff > 0)), "plain_wins": int(np.sum(diff < 0)), "ties": int(np.sum(diff == 0)), "mean_difference_plain_minus_afro_mm": float(diff.mean()), "pooled_relative_reduction_pct": float(100 * diff.mean() / np.mean([r["plain_final_mm"] for r in rows])), "median_difference_mm": float(np.median(diff)), "cluster_bootstrap_95_ci_mm": [float(x) for x in np.quantile(boot, [0.025, 0.975])]},
            "by_fruit": by_fruit,
        },
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
