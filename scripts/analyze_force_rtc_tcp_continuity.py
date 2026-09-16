#!/usr/bin/env python3
"""Measure TCP consistency across 10-step RTC force replans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_force_b2_episode50_ablation import _tcp


OFFSETS = (0, 10, 20, 30, 40)


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an < 1e-9 or bn < 1e-9:
        return None
    return float(np.dot(a, b) / (an * bn))


def _stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--moving-threshold-mm", type=float, default=5.0)
    args = parser.parse_args()

    data = np.load(args.input)
    predictions = {
        offset: np.asarray(data[f"prediction_offset{offset}"]) for offset in OFFSETS
    }
    gt = np.asarray(data["gt_decoded"])
    raw_state = np.asarray(data["raw_state"])
    frames = np.asarray(data["frame"])
    threshold_m = args.moving_threshold_mm / 1000.0

    rows = []
    for window in range(len(gt)):
        for arm in ("left", "right"):
            gt_tcp = _tcp(gt[window], arm)
            state_tcp = _tcp(raw_state[window][None], arm)[0]
            tcp = {offset: _tcp(predictions[offset][window], arm) for offset in OFFSETS}
            updated_block_vectors: dict[int, np.ndarray] = {}
            gt_vectors: dict[int, np.ndarray] = {}
            for offset in OFFSETS:
                end = min(offset + 10, 50) - 1
                gt_start = state_tcp if offset == 0 else gt_tcp[offset - 1]
                updated_block_vectors[offset] = tcp[offset][end] - gt_start
                gt_vectors[offset] = gt_tcp[end] - gt_start

            for offset in OFFSETS:
                end = min(offset + 10, 50) - 1
                current = state_tcp if offset == 0 else gt_tcp[offset - 1]
                gt_vector = gt_tcp[end] - current
                row = {
                    "window": int(window),
                    "frame": int(frames[window]),
                    "arm": arm,
                    "offset": int(offset),
                    "gt_block_displacement_mm": float(np.linalg.norm(gt_vector) * 1000.0),
                    "gt_active": bool(np.linalg.norm(gt_vector) >= threshold_m),
                    "updated_block_displacement_mm": float(
                        np.linalg.norm(updated_block_vectors[offset]) * 1000.0
                    ),
                }
                if offset > 0:
                    previous_offset = offset - 10
                    old_vector = tcp[previous_offset][end] - current
                    new_vector = tcp[offset][end] - current
                    replan_cosine = _cosine(old_vector, new_vector)
                    adjacent_cosine = _cosine(
                        updated_block_vectors[previous_offset],
                        updated_block_vectors[offset],
                    )
                    old_boundary_step = (
                        tcp[previous_offset][offset] - current
                    )
                    new_boundary_step = (
                        tcp[offset][offset] - current
                    )
                    row.update(
                        {
                            "replan_same_block_cosine": replan_cosine,
                            "replan_endpoint_shift_mm": float(
                                np.linalg.norm(tcp[offset][end] - tcp[previous_offset][end])
                                * 1000.0
                            ),
                            "updated_adjacent_block_cosine": adjacent_cosine,
                            "gt_adjacent_block_cosine": _cosine(
                                gt_vectors[previous_offset], gt_vectors[offset]
                            ),
                            "boundary_direction_cosine": _cosine(
                                old_boundary_step, new_boundary_step
                            ),
                            "old_planned_boundary_step_mm": float(
                                np.linalg.norm(old_boundary_step) * 1000.0
                            ),
                            "updated_boundary_step_mm": float(
                                np.linalg.norm(new_boundary_step) * 1000.0
                            ),
                            "boundary_replan_target_shift_mm": float(
                                np.linalg.norm(
                                    tcp[offset][offset] - tcp[previous_offset][offset]
                                )
                                * 1000.0
                            ),
                        }
                    )
                rows.append(row)

    comparisons = [row for row in rows if row["offset"] > 0]
    active = [row for row in comparisons if row["gt_active"]]

    def summarize(selected: list[dict]) -> dict:
        replans = [
            row["replan_same_block_cosine"]
            for row in selected
            if row.get("replan_same_block_cosine") is not None
        ]
        adjacent = [
            row["updated_adjacent_block_cosine"]
            for row in selected
            if row.get("updated_adjacent_block_cosine") is not None
        ]
        boundary_directions = [
            row["boundary_direction_cosine"]
            for row in selected
            if row.get("boundary_direction_cosine") is not None
        ]
        result = {
            "comparison_count": len(selected),
            "replan_same_block_cosine": _stats(replans),
            "replan_same_block_same_direction_fraction_cos_gt_0_5": (
                float(np.mean(np.asarray(replans) > 0.5)) if replans else None
            ),
            "replan_same_block_reverse_fraction_cos_lt_0": (
                float(np.mean(np.asarray(replans) < 0.0)) if replans else None
            ),
            "updated_adjacent_block_cosine": _stats(adjacent),
            "updated_adjacent_same_direction_fraction_cos_gt_0_5": (
                float(np.mean(np.asarray(adjacent) > 0.5)) if adjacent else None
            ),
            "updated_adjacent_reverse_fraction_cos_lt_0": (
                float(np.mean(np.asarray(adjacent) < 0.0)) if adjacent else None
            ),
            "gt_adjacent_block_cosine": _stats(
                [
                    row["gt_adjacent_block_cosine"]
                    for row in selected
                    if row.get("gt_adjacent_block_cosine") is not None
                ]
            ),
            "boundary_direction_cosine": _stats(boundary_directions),
            "boundary_direction_reverse_fraction_cos_lt_0": (
                float(np.mean(np.asarray(boundary_directions) < 0.0))
                if boundary_directions
                else None
            ),
            "replan_endpoint_shift_mm": _stats(
                [row["replan_endpoint_shift_mm"] for row in selected]
            ),
            "old_planned_boundary_step_mm": _stats(
                [row["old_planned_boundary_step_mm"] for row in selected]
            ),
            "updated_boundary_step_mm": _stats(
                [row["updated_boundary_step_mm"] for row in selected]
            ),
            "boundary_replan_target_shift_mm": _stats(
                [row["boundary_replan_target_shift_mm"] for row in selected]
            ),
        }
        return result

    summary = {
        "input": str(args.input.resolve()),
        "window_count": int(len(gt)),
        "arms": ["left", "right"],
        "moving_threshold_mm": args.moving_threshold_mm,
        "definitions": {
            "replan_same_block_cosine": (
                "cosine between the previous and updated plans for the same next 10-step "
                "TCP displacement, both measured from the GT current TCP"
            ),
            "updated_adjacent_block_cosine": (
                "cosine between consecutive offset-specific 10-step block displacements; "
                "each block starts from its clean teacher-forced GT current TCP"
            ),
            "boundary_direction_cosine": (
                "cosine between the previous plan's next TCP step and the updated plan's "
                "first TCP step, both starting at the prior block endpoint"
            ),
        },
        "all_arms": summarize(comparisons),
        "gt_active_arms": summarize(active),
        "by_offset_gt_active": {
            str(offset): summarize(
                [row for row in active if row["offset"] == offset]
            )
            for offset in OFFSETS[1:]
        },
        "by_arm_gt_active": {
            arm: summarize([row for row in active if row["arm"] == arm])
            for arm in ("left", "right")
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
