#!/usr/bin/env python3
"""Merge Plug planner shards and report exact and task-graph-compatible success."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


LABELS = [
    "靠近并抓取充电插头",
    "举起充电插头并停留在胸前",
    "旋转调整充电插头使其充电端正对插排",
    "移动插头靠近插排并停在插排上方",
    "靠近并抓住插排使插排固定",
    "调整充电插头对准插排插孔并插入",
    "适当按压插头使其稳固且服帖",
    "任务完成后缩回",
]

# Frozen before the held-out sweep completed. These are optional task-graph
# predecessors, not arbitrary semantic aliases.
COMPATIBLE_OPTIONAL_TRANSITIONS = {
    (LABELS[4], LABELS[3]),  # a demonstration may skip the move-above phase
    (LABELS[7], LABELS[6]),  # a demonstration may skip the final press phase
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-episode", type=int, action="append", default=[])
    return parser.parse_args()


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float]:
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half = z * (
        rate * (1.0 - rate) / total + z * z / (4.0 * total * total)
    ) ** 0.5 / denominator
    return {"low": center - half, "high": center + half}


def mean_stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "min": ordered[0],
        "max": ordered[-1],
    }


def compatible(case: dict[str, object]) -> bool:
    if bool(case["exact_match"]):
        return True
    return (str(case["expected_subtask"]), str(case["predicted_subtask"])) in (
        COMPATIBLE_OPTIONAL_TRANSITIONS
    )


def memory_causality_violations(case: dict[str, object]) -> list[str]:
    parsed = case.get("parsed_response")
    if not isinstance(parsed, dict) or not isinstance(parsed.get("memory_to_store"), dict):
        return ["missing_memory"]
    memory = parsed["memory_to_store"]
    prediction = str(case.get("predicted_subtask"))
    violations: list[str] = []
    selected_completion_checks = {
        LABELS[0]: memory.get("plug_held") is True,
        LABELS[1]: memory.get("plug_lifted") is True,
        LABELS[2]: memory.get("plug_oriented") is True,
        LABELS[4]: memory.get("strip_stabilized") is True,
        LABELS[5]: memory.get("insertion_state") == "seated",
        LABELS[6]: memory.get("insertion_state") == "seated",
    }
    if selected_completion_checks.get(prediction, False):
        violations.append("selected_subtask_marked_completed")
    if memory.get("plug_lifted") is True and memory.get("plug_held") is not True:
        violations.append("lifted_without_held")
    if memory.get("plug_oriented") is True and (
        memory.get("plug_held") is not True or memory.get("plug_lifted") is not True
    ):
        violations.append("oriented_without_held_and_lifted")
    if memory.get("insertion_state") in {"alignment", "contact", "seated"} and (
        memory.get("plug_held") is not True
        or memory.get("plug_oriented") is not True
        or memory.get("strip_stabilized") is not True
    ):
        violations.append("insertion_state_without_prerequisites")
    return violations


def load_cases(input_dir: Path) -> list[dict[str, object]]:
    final = input_dir / "cases.json"
    partial = input_dir / "cases.partial.json"
    path = final if final.is_file() else partial
    if not path.is_file():
        raise FileNotFoundError(f"no cases file in {input_dir}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError(f"cases must be a list: {path}")
    return value


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged: dict[tuple[int, int, int], dict[str, object]] = {}
    for input_dir in args.input_dir:
        for case in load_cases(input_dir):
            key = (
                int(case["episode_id"]),
                int(case["boundary_index"]),
                int(case["recovery_step"]),
            )
            if key in merged:
                comparable = ("expected_subtask", "predicted_subtask", "exact_match")
                if any(merged[key].get(field) != case.get(field) for field in comparable):
                    raise ValueError(f"conflicting duplicate case: {key}")
                continue
            merged[key] = case
    cases = [merged[key] for key in sorted(merged)]
    by_boundary: dict[tuple[int, int], list[dict[str, object]]] = {}
    for case in cases:
        key = (int(case["episode_id"]), int(case["boundary_index"]))
        by_boundary.setdefault(key, []).append(case)
    boundaries: list[dict[str, object]] = []
    for (episode_id, boundary_index), attempts in sorted(by_boundary.items()):
        attempts.sort(key=lambda row: int(row["recovery_step"]))
        initial = attempts[0]
        exact_first = next((row for row in attempts if bool(row["exact_match"])), None)
        compatible_first = next((row for row in attempts if compatible(row)), None)
        boundaries.append(
            {
                "episode_id": episode_id,
                "boundary_index": boundary_index,
                "expected_subtask": initial["expected_subtask"],
                "initial_prediction": initial["predicted_subtask"],
                "initial_exact": bool(initial["exact_match"]),
                "initial_compatible": compatible(initial),
                "exact_recovered": exact_first is not None,
                "compatible_recovered": compatible_first is not None,
                "exact_recovery_step": None if exact_first is None else exact_first["recovery_step"],
                "compatible_recovery_step": (
                    None if compatible_first is None else compatible_first["recovery_step"]
                ),
                "exact_ready_after_boundary_s": (
                    None
                    if exact_first is None
                    else float(exact_first["offset_s"])
                    + float(exact_first["timing"]["subtask_ready_s"])
                ),
                "compatible_ready_after_boundary_s": (
                    None
                    if compatible_first is None
                    else float(compatible_first["offset_s"])
                    + float(compatible_first["timing"]["subtask_ready_s"])
                ),
            }
        )

    observed_episodes = sorted({int(row["episode_id"]) for row in boundaries})
    if args.expected_episode:
        expected = sorted(set(args.expected_episode))
        if observed_episodes != expected:
            missing = sorted(set(expected) - set(observed_episodes))
            extra = sorted(set(observed_episodes) - set(expected))
            raise ValueError(f"episode coverage mismatch: missing={missing} extra={extra}")

    initial_cases = [row for row in cases if int(row["recovery_step"]) == 0]
    boundary_count = len(boundaries)
    exact_initial = sum(bool(row["initial_exact"]) for row in boundaries)
    compatible_initial = sum(bool(row["initial_compatible"]) for row in boundaries)
    exact_eventual = sum(bool(row["exact_recovered"]) for row in boundaries)
    compatible_eventual = sum(bool(row["compatible_recovered"]) for row in boundaries)

    per_episode: dict[str, dict[str, object]] = {}
    for episode_id in observed_episodes:
        selected = [row for row in boundaries if int(row["episode_id"]) == episode_id]
        per_episode[str(episode_id)] = {
            "boundaries": len(selected),
            "all_initial_exact": all(bool(row["initial_exact"]) for row in selected),
            "all_initial_compatible": all(bool(row["initial_compatible"]) for row in selected),
            "all_eventual_exact": all(bool(row["exact_recovered"]) for row in selected),
            "all_eventual_compatible": all(bool(row["compatible_recovered"]) for row in selected),
        }

    per_label: dict[str, dict[str, object]] = {}
    confusion: dict[str, dict[str, int]] = {}
    for label in LABELS:
        selected = [row for row in initial_cases if row["expected_subtask"] == label]
        if not selected:
            continue
        exact = sum(bool(row["exact_match"]) for row in selected)
        compatible_count = sum(compatible(row) for row in selected)
        per_label[label] = {
            "total": len(selected),
            "exact": exact,
            "exact_rate": exact / len(selected),
            "compatible": compatible_count,
            "compatible_rate": compatible_count / len(selected),
        }
        counts: dict[str, int] = {}
        for row in selected:
            prediction = str(row["predicted_subtask"])
            counts[prediction] = counts.get(prediction, 0) + 1
        confusion[label] = counts

    exact_initial_errors = [row for row in boundaries if not bool(row["initial_exact"])]
    incompatible_initial_errors = [row for row in boundaries if not bool(row["initial_compatible"])]
    memory_violation_counts: dict[str, int] = {}
    strict_memory_valid = 0
    internally_consistent_memory = 0
    for row in initial_cases:
        violations = memory_causality_violations(row)
        if not violations:
            strict_memory_valid += 1
        if not any(
            violation
            in {
                "lifted_without_held",
                "oriented_without_held_and_lifted",
                "insertion_state_without_prerequisites",
            }
            for violation in violations
        ):
            internally_consistent_memory += 1
        for violation in violations:
            memory_violation_counts[violation] = memory_violation_counts.get(violation, 0) + 1
    summary = {
        "contract": {
            "input_dirs": [str(path.resolve()) for path in args.input_dir],
            "episode_count": len(observed_episodes),
            "episodes": observed_episodes,
            "scoring": "exact canonical label plus a separately reported frozen optional-transition compatibility score",
            "compatible_optional_transitions": [
                {"expected": expected, "accepted_optional_predecessor": predicted}
                for expected, predicted in sorted(COMPATIBLE_OPTIONAL_TRANSITIONS)
            ],
            "memory": "teacher-forced at each GT boundary; model memory fed forward only across retries within the same segment",
        },
        "case_count_including_retries": len(cases),
        "boundary_count": boundary_count,
        "initial_exact": exact_initial,
        "initial_exact_rate": exact_initial / boundary_count,
        "initial_exact_wilson95": wilson(exact_initial, boundary_count),
        "initial_compatible": compatible_initial,
        "initial_compatible_rate": compatible_initial / boundary_count,
        "initial_compatible_wilson95": wilson(compatible_initial, boundary_count),
        "eventual_exact_before_segment_end": exact_eventual,
        "eventual_exact_rate": exact_eventual / boundary_count,
        "eventual_exact_wilson95": wilson(exact_eventual, boundary_count),
        "eventual_compatible_before_segment_end": compatible_eventual,
        "eventual_compatible_rate": compatible_eventual / boundary_count,
        "eventual_compatible_wilson95": wilson(compatible_eventual, boundary_count),
        "initial_json_valid_rate": sum(bool(row["json_valid"]) for row in initial_cases) / boundary_count,
        "initial_memory_valid_rate": sum(bool(row["memory_valid"]) for row in initial_cases) / boundary_count,
        "initial_memory_internal_consistency_rate": internally_consistent_memory / boundary_count,
        "initial_memory_strict_causal_valid_rate": strict_memory_valid / boundary_count,
        "initial_memory_causality_violation_counts": memory_violation_counts,
        "initial_subtask_ready_latency_s": mean_stats(
            [float(row["timing"]["subtask_ready_s"]) for row in initial_cases]
        ),
        "initial_full_memory_json_latency_s": mean_stats(
            [float(row["timing"]["end_to_end_s"]) for row in initial_cases]
        ),
        "exact_initial_error_count": len(exact_initial_errors),
        "exact_initial_error_recovered_count": sum(
            bool(row["exact_recovered"]) for row in exact_initial_errors
        ),
        "exact_recovery_steps": mean_stats(
            [
                float(row["exact_recovery_step"])
                for row in exact_initial_errors
                if row["exact_recovery_step"] is not None
            ]
        ),
        "exact_recovery_ready_after_boundary_s": mean_stats(
            [
                float(row["exact_ready_after_boundary_s"])
                for row in exact_initial_errors
                if row["exact_ready_after_boundary_s"] is not None
            ]
        ),
        "incompatible_initial_error_count": len(incompatible_initial_errors),
        "incompatible_initial_error_recovered_count": sum(
            bool(row["compatible_recovered"]) for row in incompatible_initial_errors
        ),
        "episode_all_initial_exact": sum(
            bool(row["all_initial_exact"]) for row in per_episode.values()
        ),
        "episode_all_initial_exact_rate": sum(
            bool(row["all_initial_exact"]) for row in per_episode.values()
        ) / len(per_episode),
        "episode_all_initial_compatible": sum(
            bool(row["all_initial_compatible"]) for row in per_episode.values()
        ),
        "episode_all_initial_compatible_rate": sum(
            bool(row["all_initial_compatible"]) for row in per_episode.values()
        ) / len(per_episode),
        "episode_all_eventual_exact": sum(
            bool(row["all_eventual_exact"]) for row in per_episode.values()
        ),
        "episode_all_eventual_exact_rate": sum(
            bool(row["all_eventual_exact"]) for row in per_episode.values()
        ) / len(per_episode),
        "episode_all_eventual_compatible": sum(
            bool(row["all_eventual_compatible"]) for row in per_episode.values()
        ),
        "episode_all_eventual_compatible_rate": sum(
            bool(row["all_eventual_compatible"]) for row in per_episode.values()
        ) / len(per_episode),
        "per_label": per_label,
        "initial_confusion": confusion,
        "per_episode": per_episode,
    }

    (args.output_dir / "cases_merged.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "boundaries_merged.json").write_text(
        json.dumps(boundaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "per_episode.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["episode_id", *next(iter(per_episode.values())).keys()])
        writer.writeheader()
        for episode_id, row in per_episode.items():
            writer.writerow({"episode_id": episode_id, **row})
    table = [
        "| Metric | Success | Rate |",
        "|---|---:|---:|",
        f"| Boundary initial exact | {exact_initial}/{boundary_count} | {100 * exact_initial / boundary_count:.1f}% |",
        f"| Boundary initial task-graph compatible | {compatible_initial}/{boundary_count} | {100 * compatible_initial / boundary_count:.1f}% |",
        f"| Boundary eventual exact | {exact_eventual}/{boundary_count} | {100 * exact_eventual / boundary_count:.1f}% |",
        f"| Boundary eventual compatible | {compatible_eventual}/{boundary_count} | {100 * compatible_eventual / boundary_count:.1f}% |",
        f"| Episode all-initial exact | {summary['episode_all_initial_exact']}/{len(per_episode)} | {100 * summary['episode_all_initial_exact_rate']:.1f}% |",
        f"| Episode all-initial compatible | {summary['episode_all_initial_compatible']}/{len(per_episode)} | {100 * summary['episode_all_initial_compatible_rate']:.1f}% |",
        f"| Episode all-eventual exact | {summary['episode_all_eventual_exact']}/{len(per_episode)} | {100 * summary['episode_all_eventual_exact_rate']:.1f}% |",
        f"| Episode all-eventual compatible | {summary['episode_all_eventual_compatible']}/{len(per_episode)} | {100 * summary['episode_all_eventual_compatible_rate']:.1f}% |",
        f"| Memory JSON/schema valid | {sum(bool(row['memory_valid']) for row in initial_cases)}/{boundary_count} | {100 * summary['initial_memory_valid_rate']:.1f}% |",
        f"| Memory internally consistent | {internally_consistent_memory}/{boundary_count} | {100 * summary['initial_memory_internal_consistency_rate']:.1f}% |",
        f"| Memory strict causal valid | {strict_memory_valid}/{boundary_count} | {100 * summary['initial_memory_strict_causal_valid_rate']:.1f}% |",
    ]
    (args.output_dir / "paper_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
