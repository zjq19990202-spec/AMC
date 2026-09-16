#!/usr/bin/env python3
"""Measure multi-episode vase-subtask accuracy and recovery after a wrong decision."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from benchmark_qwen3vl8b_vase_memory_switches import (
    SYSTEM,
    extract_frame,
    infer,
    make_memory,
    parse_response,
    stats,
)
from test_qwen3vl8b_vase_subtask_zero_shot import LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--timeline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, action="append", required=True)
    parser.add_argument("--first-offset", type=float, default=0.33)
    parser.add_argument("--decision-period", type=float, default=0.8)
    parser.add_argument("--max-recovery-steps", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--only-boundary",
        action="append",
        default=[],
        metavar="EPISODE:INDEX",
        help="Restrict evaluation to one or more episode/boundary pairs.",
    )
    return parser.parse_args()


def prediction_from_response(response: str, parsed: dict[str, object] | None) -> str | None:
    if parsed is not None and isinstance(parsed.get("current_subtask"), str):
        return str(parsed["current_subtask"])
    match = re.search(r'"current_subtask"\s*:\s*"([^"]+)"', response)
    return match.group(1) if match else None


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def main() -> None:
    args = parse_args()
    only_boundaries = {
        tuple(int(part) for part in value.split(":")) for value in args.only_boundary
    }
    if any(len(item) != 2 for item in only_boundaries):
        raise ValueError("--only-boundary must use EPISODE:INDEX")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output_dir / "frames"
    episode_specs: list[dict[str, object]] = []
    for episode_id in args.episode:
        video = args.video_root / f"file-{episode_id:03d}.mp4"
        timeline = (
            args.timeline_root
            / f"episode_{episode_id:06d}"
            / "result.v11_final_memory.json"
        )
        if not video.is_file() or not timeline.is_file():
            raise FileNotFoundError(f"missing episode assets: video={video}, timeline={timeline}")
        payload = json.loads(timeline.read_text(encoding="utf-8"))
        subtasks = payload["parsed"]["subtasks"]
        episode_specs.append(
            {
                "episode_id": episode_id,
                "video": video,
                "timeline": timeline,
                "subtasks": subtasks,
            }
        )

    load_start = time.perf_counter()
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_s = time.perf_counter() - load_start

    warmup_spec = episode_specs[0]
    warmup_subtasks = warmup_spec["subtasks"]
    warmup_boundary = min(3, len(warmup_subtasks) - 1)
    warmup_s = float(warmup_subtasks[warmup_boundary]["start_s"]) + args.first_offset
    warmup_path = frame_dir / "warmup.png"
    extract_frame(Path(warmup_spec["video"]), warmup_s, warmup_path)
    warmup_payload = {
        "task": "wipe the vase",
        "allowed_subtasks": LABELS,
        "prior_memory": make_memory(warmup_subtasks, warmup_boundary),
        "observation_order": "one current frame; cross-frame state is summarized in prior_memory",
    }
    _, warmup_timing = infer(
        model, processor, [warmup_path], warmup_payload, args.max_new_tokens, False
    )

    cases: list[dict[str, object]] = []
    boundaries: list[dict[str, object]] = []
    for spec in episode_specs:
        episode_id = int(spec["episode_id"])
        video = Path(spec["video"])
        subtasks = spec["subtasks"]
        for boundary_index in range(1, len(subtasks)):
            if only_boundaries and (episode_id, boundary_index) not in only_boundaries:
                continue
            boundary_s = float(subtasks[boundary_index]["start_s"])
            segment_end_s = float(subtasks[boundary_index]["end_s"])
            expected = str(subtasks[boundary_index]["current_subtask"])
            prior_memory = make_memory(subtasks, boundary_index)
            attempts: list[dict[str, object]] = []
            for recovery_step in range(args.max_recovery_steps + 1):
                offset_s = args.first_offset + recovery_step * args.decision_period
                decision_s = boundary_s + offset_s
                if decision_s >= segment_end_s - 0.05:
                    break
                image_path = (
                    frame_dir
                    / f"ep{episode_id:03d}_b{boundary_index:02d}_r{recovery_step}_{decision_s:.3f}.png"
                )
                extract_frame(video, decision_s, image_path)
                user_payload = {
                    "task": "wipe the vase",
                    "allowed_subtasks": LABELS,
                    "prior_memory": prior_memory,
                    "observation_order": "one current frame; cross-frame state is summarized in prior_memory",
                    "decision_time_s": decision_s,
                }
                response, timing = infer(
                    model,
                    processor,
                    [image_path],
                    user_payload,
                    args.max_new_tokens,
                    False,
                )
                parsed = parse_response(response)
                prediction = prediction_from_response(response, parsed)
                exact_match = prediction == expected
                attempt = {
                    "episode_id": episode_id,
                    "boundary_index": boundary_index,
                    "recovery_step": recovery_step,
                    "boundary_s": boundary_s,
                    "segment_end_s": segment_end_s,
                    "offset_s": offset_s,
                    "decision_s": decision_s,
                    "previous_subtask": subtasks[boundary_index - 1]["current_subtask"],
                    "expected_subtask": expected,
                    "predicted_subtask": prediction,
                    "exact_match": exact_match,
                    "timing": timing,
                    "image": str(image_path.resolve()),
                    "prior_memory": user_payload["prior_memory"],
                    "raw_response": response,
                    "parsed_response": parsed,
                }
                attempts.append(attempt)
                cases.append(attempt)
                if exact_match:
                    break
                if parsed is not None and isinstance(parsed.get("memory_to_store"), dict):
                    prior_memory = parsed["memory_to_store"]
            first_attempt = attempts[0] if attempts else None
            first_correct = next((item for item in attempts if item["exact_match"]), None)
            boundaries.append(
                {
                    "episode_id": episode_id,
                    "boundary_index": boundary_index,
                    "expected_subtask": expected,
                    "initial_exact_match": bool(first_attempt and first_attempt["exact_match"]),
                    "attempt_count": len(attempts),
                    "recovered": first_correct is not None,
                    "recovery_step": None if first_correct is None else first_correct["recovery_step"],
                    "first_correct_offset_s": None if first_correct is None else first_correct["offset_s"],
                    "output_ready_after_boundary_s": (
                        None
                        if first_correct is None
                        else float(first_correct["offset_s"])
                        + float(first_correct["timing"]["subtask_ready_s"])
                    ),
                }
            )
            (args.output_dir / "cases.partial.json").write_text(
                json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            status = boundaries[-1]
            print(
                f"episode={episode_id:03d} boundary={boundary_index:02d} "
                f"initial={status['initial_exact_match']} recovered={status['recovered']} "
                f"step={status['recovery_step']}"
            )

    initial_correct = sum(bool(item["initial_exact_match"]) for item in boundaries)
    initial_errors = [item for item in boundaries if not item["initial_exact_match"]]
    recovered_errors = [item for item in initial_errors if item["recovered"]]
    initial_cases = [case for case in cases if int(case["recovery_step"]) == 0]
    ready_latencies = [float(case["timing"]["subtask_ready_s"]) for case in initial_cases]
    recovery_steps = [int(item["recovery_step"]) for item in recovered_errors]
    recovery_offsets = [float(item["first_correct_offset_s"]) for item in recovered_errors]
    ready_after_boundary = [
        float(item["output_ready_after_boundary_s"]) for item in recovered_errors
    ]
    per_episode: dict[str, object] = {}
    for episode_id in args.episode:
        selected = [item for item in boundaries if int(item["episode_id"]) == episode_id]
        correct = sum(bool(item["initial_exact_match"]) for item in selected)
        per_episode[str(episode_id)] = {
            "boundaries": len(selected),
            "initial_correct": correct,
            "initial_accuracy": correct / len(selected),
            "initial_errors": len(selected) - correct,
            "unrecovered": sum(
                (not bool(item["initial_exact_match"])) and (not bool(item["recovered"]))
                for item in selected
            ),
        }
    summary = {
        "evaluation_contract": {
            "model": str(args.model.resolve()),
            "episodes": args.episode,
            "episode_selection": "approximately equally spaced episodes with the legacy 15-stage memory contract, fixed before inference",
            "only_boundaries": sorted([list(item) for item in only_boundaries]),
            "teacher_forced_memory_at_boundary": True,
            "model_memory_fed_forward_after_error": True,
            "first_observation_offset_s": args.first_offset,
            "decision_period_s": args.decision_period,
            "maximum_recovery_steps": args.max_recovery_steps,
            "model_load_and_warmup_excluded_from_latency": True,
        },
        "system_message": SYSTEM,
        "model_load_s_excluded": model_load_s,
        "warmup_timing_excluded": warmup_timing,
        "episode_count": len(episode_specs),
        "boundary_count": len(boundaries),
        "initial_correct": initial_correct,
        "initial_accuracy": initial_correct / len(boundaries),
        "initial_error_count": len(initial_errors),
        "initial_error_recovered_count": len(recovered_errors),
        "initial_error_recovery_rate": (
            len(recovered_errors) / len(initial_errors) if initial_errors else None
        ),
        "unrecovered_error_count": len(initial_errors) - len(recovered_errors),
        "initial_subtask_ready_latency_s": stats(ready_latencies),
        "recovered_error_steps": {
            "mean": mean_or_none([float(value) for value in recovery_steps]),
            "median": statistics.median(recovery_steps) if recovery_steps else None,
            "max": max(recovery_steps) if recovery_steps else None,
        },
        "recovered_error_first_correct_offset_s": {
            "mean": mean_or_none(recovery_offsets),
            "median": statistics.median(recovery_offsets) if recovery_offsets else None,
            "max": max(recovery_offsets) if recovery_offsets else None,
        },
        "recovered_error_output_ready_after_boundary_s": {
            "mean": mean_or_none(ready_after_boundary),
            "median": statistics.median(ready_after_boundary) if ready_after_boundary else None,
            "max": max(ready_after_boundary) if ready_after_boundary else None,
        },
        "per_episode": per_episode,
    }
    (args.output_dir / "cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "boundaries.json").write_text(
        json.dumps(boundaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
