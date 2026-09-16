#!/usr/bin/env python3
"""Compare each 10-step base TCP increment with the increment caused by zF."""

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
    if an < 1.0e-9 or bn < 1.0e-9:
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


def _summary(rows: list[dict]) -> dict:
    alignments = [row["base_vs_zf_increment_cosine"] for row in rows]
    final_alignments = [row["base_vs_force_final_cosine"] for row in rows]
    full_gt_error = np.asarray(
        [row["full_tcp_gt_endpoint_error_mm"] for row in rows], dtype=np.float64
    )
    zero_gt_error = np.asarray(
        [row["zero_force_tcp_gt_endpoint_error_mm"] for row in rows], dtype=np.float64
    )
    return {
        "count": len(rows),
        "base_vs_zf_increment_cosine": _stats(alignments),
        "zf_reinforces_base_fraction_cos_gt_0_5": float(
            np.mean(np.asarray(alignments) > 0.5)
        ) if alignments else None,
        "zf_opposes_base_fraction_cos_lt_minus_0_5": float(
            np.mean(np.asarray(alignments) < -0.5)
        ) if alignments else None,
        "zf_lateral_fraction_abs_cos_le_0_5": float(
            np.mean(np.abs(np.asarray(alignments)) <= 0.5)
        ) if alignments else None,
        "base_vs_force_final_cosine": _stats(final_alignments),
        "base_increment_mm": _stats([row["base_increment_mm"] for row in rows]),
        "zf_increment_mm": _stats([row["zf_increment_mm"] for row in rows]),
        "zf_parallel_component_mm": _stats(
            [row["zf_parallel_component_mm"] for row in rows]
        ),
        "zf_lateral_component_mm": _stats(
            [row["zf_lateral_component_mm"] for row in rows]
        ),
        "zf_to_base_magnitude_ratio": _stats(
            [row["zf_to_base_magnitude_ratio"] for row in rows]
        ),
        "full_tcp_gt_endpoint_error_mm": _stats(full_gt_error.tolist()),
        "zero_force_tcp_gt_endpoint_error_mm": _stats(zero_gt_error.tolist()),
        "full_closer_to_gt_fraction": (
            float(np.mean(full_gt_error < zero_gt_error)) if len(rows) else None
        ),
        "full_gt_endpoint_rmse_mm": (
            float(np.sqrt(np.mean(np.square(full_gt_error)))) if len(rows) else None
        ),
        "zero_force_gt_endpoint_rmse_mm": (
            float(np.sqrt(np.mean(np.square(zero_gt_error)))) if len(rows) else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-moving-threshold-mm", type=float, default=5.0)
    parser.add_argument("--zf-effect-threshold-mm", type=float, default=1.0)
    args = parser.parse_args()

    data = np.load(args.input)
    full = {offset: np.asarray(data[f"prediction_offset{offset}"]) for offset in OFFSETS}
    zero = {
        offset: np.asarray(data[f"zero_force_prediction_offset{offset}"])
        for offset in OFFSETS
    }
    gt = np.asarray(data["gt_decoded"])
    raw_state = np.asarray(data["raw_state"])
    frames = np.asarray(data["frame"])

    rows = []
    for window in range(len(gt)):
        for arm in ("left", "right"):
            state_tcp = _tcp(raw_state[window][None], arm)[0]
            gt_tcp = _tcp(gt[window], arm)
            for offset in OFFSETS:
                end = min(offset + 10, 50) - 1
                current_tcp = state_tcp if offset == 0 else gt_tcp[offset - 1]
                full_end = _tcp(full[offset][window], arm)[end]
                zero_end = _tcp(zero[offset][window], arm)[end]
                gt_end = gt_tcp[end]
                gt_increment = gt_end - current_tcp
                base_increment = zero_end - current_tcp
                force_final_increment = full_end - current_tcp
                zf_increment = full_end - zero_end
                base_norm = float(np.linalg.norm(base_increment))
                zf_norm = float(np.linalg.norm(zf_increment))
                if base_norm < args.base_moving_threshold_mm / 1000.0:
                    continue
                if zf_norm < args.zf_effect_threshold_mm / 1000.0:
                    continue
                base_unit = base_increment / base_norm
                parallel = float(np.dot(zf_increment, base_unit))
                lateral = float(
                    np.linalg.norm(zf_increment - parallel * base_unit)
                )
                rows.append(
                    {
                        "window": int(window),
                        "frame": int(frames[window]),
                        "arm": arm,
                        "offset": int(offset),
                        "gt_active": bool(
                            np.linalg.norm(gt_increment)
                            >= args.base_moving_threshold_mm / 1000.0
                        ),
                        "base_increment_mm": base_norm * 1000.0,
                        "zf_increment_mm": zf_norm * 1000.0,
                        "zf_parallel_component_mm": parallel * 1000.0,
                        "zf_lateral_component_mm": lateral * 1000.0,
                        "zf_to_base_magnitude_ratio": zf_norm / base_norm,
                        "base_vs_zf_increment_cosine": _cosine(
                            base_increment, zf_increment
                        ),
                        "base_vs_force_final_cosine": _cosine(
                            base_increment, force_final_increment
                        ),
                        "full_tcp_gt_endpoint_error_mm": float(
                            np.linalg.norm(full_end - gt_end) * 1000.0
                        ),
                        "zero_force_tcp_gt_endpoint_error_mm": float(
                            np.linalg.norm(zero_end - gt_end) * 1000.0
                        ),
                    }
                )

    summary = {
        "input": str(args.input.resolve()),
        "window_count": int(len(gt)),
        "base_moving_threshold_mm": args.base_moving_threshold_mm,
        "zf_effect_threshold_mm": args.zf_effect_threshold_mm,
        "definition": (
            "base increment is the 10-step TCP endpoint displacement with force_update_scale=0; "
            "zF increment is full-force endpoint minus that paired zero-force endpoint; both use "
            "the same observation, prompt, GT current TCP, flow noise, RTC prefix, and offset"
        ),
        "all": _summary(rows),
        "gt_active": _summary([row for row in rows if row["gt_active"]]),
        "by_offset": {
            str(offset): _summary([row for row in rows if row["offset"] == offset])
            for offset in OFFSETS
        },
        "by_arm": {
            arm: _summary([row for row in rows if row["arm"] == arm])
            for arm in ("left", "right")
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
