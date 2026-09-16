#!/usr/bin/env python3
"""Test Qwen3.5-4B boundaries on Plug episode 36, truncated before USB."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from benchmark_qwen3vl8b_vase_memory_switches import extract_frame, infer, parse_response


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/media/admin123/My Passport/models/Qwen3.5-4B-ModelScope")
VIDEO = Path("/media/admin123/T5 EVO/toread/lerobot_v3_4to1_plug_force/videos/observation.images.base_0_rgb/chunk-000/file-036.mp4")
TIMELINE = ROOT / "runtime_assets/eval_manifests/plug_ep36_manual_20260906/episode_subtasks.jsonl"
OUT = ROOT / "eval_outputs/qwen35_4b_ep36_plug_only_boundaries_strict_20260912"

LABELS = [
    "grasp the drawer handle and pull it open",
    "approach and grasp the power adapter",
    "lift the power adapter and reposition it with the other gripper",
    "grasp the socket and rotate it counterclockwise",
    "plug the power adapter into the socket",
]

SYSTEM = """You track a bimanual robot performing only the first power-adapter insertion part of an episode.
The later USB-plug task is outside the requested scope and must never be predicted. Select exactly one
item from allowed_subtasks, copying it verbatim. Trust completed facts in compact memory. Use the current
single base-camera frame to choose the earliest unfinished action. Do not skip prerequisites. Do not keep
an already completed subtask. If every allowed subtask is visibly complete, output TASK_COMPLETE.
Return one-line valid JSON only:
{"current_subtask":"EXACT_ALLOWED_LABEL_OR_TASK_COMPLETE","memory_to_store":{"progress_summary":"MAX 16 WORDS","completed_subtasks":["EXACT_COMPLETED_LABELS"],"next_required_phase":"MAX 8 WORDS"},"visual_evidence":"MAX 16 WORDS"}"""


def memory_for(boundary_index: int) -> dict[str, object]:
    completed = LABELS[:boundary_index]
    return {
        "progress_summary": (
            "Completed: " + "; ".join(completed)
            if completed else "No subtask has been completed."
        ),
        "completed_subtasks": completed,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = OUT / "frames"
    payload = json.loads(TIMELINE.read_text(encoding="utf-8").strip())
    segments = payload["semantic_segments"][:5]

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)

    # Warmup is excluded from latency statistics.
    warm = frames / "warmup.png"
    extract_frame(VIDEO, segments[2]["start_frame_30hz"] / 30 + 0.33, warm)
    infer(model, processor, [warm], {
        "allowed_subtasks": LABELS,
        "prior_memory": memory_for(2),
    }, 192, False, SYSTEM)

    cases: list[dict[str, object]] = []
    # Four internal boundaries plus a terminal probe immediately after plug completion.
    for boundary_index in range(1, 6):
        boundary_frame = (
            segments[boundary_index]["start_frame_30hz"]
            if boundary_index < 5 else segments[4]["end_frame_30hz_exclusive"]
        )
        expected = LABELS[boundary_index] if boundary_index < 5 else "TASK_COMPLETE"
        for retry, offset in enumerate((0.33, 1.13, 1.93)):
            decision_s = boundary_frame / 30 + offset
            frame = frames / f"b{boundary_index:02d}_r{retry}_{decision_s:.3f}.png"
            extract_frame(VIDEO, decision_s, frame)
            response, timing = infer(model, processor, [frame], {
                "task_scope": "Open drawer and insert the power adapter; stop before all USB actions.",
                "allowed_subtasks": LABELS,
                "prior_memory": memory_for(boundary_index),
            }, 192, False, SYSTEM)
            parsed = parse_response(response)
            predicted = parsed.get("current_subtask") if parsed else None
            cases.append({
                "boundary_index": boundary_index,
                "retry": retry,
                "offset_s": offset,
                "decision_s": decision_s,
                "expected": expected,
                "predicted": predicted,
                "exact": predicted == expected,
                "json_valid": parsed is not None,
                "timing": timing,
                "frame": str(frame.resolve()),
                "response": response,
            })
            print(boundary_index, retry, expected, "->", predicted, flush=True)
            if predicted == expected:
                break
        (OUT / "cases.partial.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")

    initial = [row for row in cases if row["retry"] == 0]
    internal_initial = [row for row in initial if row["boundary_index"] < 5]
    boundaries = []
    for boundary_index in range(1, 6):
        attempts = [row for row in cases if row["boundary_index"] == boundary_index]
        success = next((row for row in attempts if row["exact"]), None)
        boundaries.append({
            "boundary_index": boundary_index,
            "expected": attempts[0]["expected"],
            "initial_exact": attempts[0]["exact"],
            "eventual_exact": success is not None,
            "first_exact_offset_s": None if success is None else success["offset_s"],
        })
    latencies = [float(row["timing"]["subtask_ready_s"]) for row in initial]
    summary = {
        "scope": "episode 36, first five segments through power-adapter insertion; USB excluded",
        "internal_boundary_count": 4,
        "internal_initial_exact": sum(row["exact"] for row in internal_initial),
        "internal_initial_exact_rate": sum(row["exact"] for row in internal_initial) / 4,
        "internal_eventual_exact": sum(row["eventual_exact"] for row in boundaries[:4]),
        "terminal_plug_complete_initial_exact": boundaries[4]["initial_exact"],
        "terminal_plug_complete_eventual_exact": boundaries[4]["eventual_exact"],
        "initial_json_valid_rate": sum(row["json_valid"] for row in initial) / len(initial),
        "subtask_ready_latency_s": {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "max": max(latencies),
        },
        "boundaries": boundaries,
    }
    (OUT / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
