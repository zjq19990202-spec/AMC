#!/usr/bin/env python3
"""Benchmark memory-conditioned Qwen3-VL-8B decisions at every vase subtask boundary."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForMultimodalLM, AutoProcessor, PreTrainedModel, TextIteratorStreamer

from test_qwen3vl8b_vase_subtask_zero_shot import LABELS


SYSTEM = """You are the online task-state tracker for one vase-wiping robot task. Select exactly
one current subtask from `allowed_subtasks`, copied character-for-character. Use one current image
and the supplied compact semantic memory. Assume the memory is correct and all states described as
completed really are complete.

Follow this task-specific policy:
1. After the eraser is grasped but before it is at the vase, move the eraser in front of the vase.
2. After the eraser is in front but before the vase is stabilized, grasp and hold the upper rim.
3. Once the eraser is ready and the vase is stabilized, wipe the initial reachable writing.
4. The input contains exactly one current base-camera frame. The vase is white. The writing to erase
   is a blue line on the white vase surface. Inspect only the vase surface currently facing the
   camera. Any visible blue line on that front-facing white surface counts as writing, even if the
   line is thin, short, close to an edge, or partly occluded by the eraser. If it is reachable by
   the eraser, wipe it.
5. If the currently front-facing surface has no visible writing and the full circuit is unfinished,
   rotate the vase counterclockwise, then press it down to expose another surface.
6. Consecutive rotations are allowed. After one rotation completes, if the newly front-facing
   surface also has no visible writing, select rotation again. Do not force wipe/rotate alternation.
7. Writing visible only on a side that is not currently front-facing does not trigger wiping yet.
8. The black upper-rim marker determines whether the full circuit is complete. When it returns
   to its initial orientation and all encountered writing is handled, cleaning is complete.
9. After cleaning completes, first retract and release the eraser. Only after the eraser is released
   may the vase be moved farther away and the arm retracted.

Memory state constraints override an ambiguous image only for completed prior transitions. The
current single image decides whether writing is visible on the front-facing surface.

Return compact valid JSON only:
{"current_subtask":"EXACT_ALLOWED_LABEL","memory_to_store":{"progress_summary":"one compact
semantic task-state summary","marker_status":"not_returned or returned",
"full_circuit_complete":false,"front_writing_visible":false,
"next_required_phase":"short semantic phase"},
"visual_evidence":"at most 20 words"}
Update the semantic summary without producing counters or an event-history list. Do not mark the
selected current subtask completed."""

COMPACT_SYSTEM = SYSTEM.split("\nReturn compact valid JSON only:", 1)[0] + """

Return one-line JSON only:
{"current_subtask":"EXACT_ALLOWED_LABEL","memory_to_store":{"progress_summary":"MAX 8 WORDS",
"marker_status":"not_returned or returned","full_circuit_complete":false,
"front_writing_visible":false,"next_required_phase":"MAX 3 WORDS"},"visual_evidence":"MAX 6 WORDS"}
Memory is a semantic summary, never an event ledger. Do not mark the
selected current subtask completed."""

LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/home/admin123/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offset", type=float, action="append", default=[])
    parser.add_argument("--boundary", type=int, action="append", default=[])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--compact", action="store_true", help="Use the low-latency ID protocol")
    return parser.parse_args()


def extract_frame(video: Path, time_s: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-ss", f"{max(time_s, 0.0):.6f}",
            "-i", str(video), "-frames:v", "1", "-y", str(path),
        ],
        check=True,
    )


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_response(text: str) -> dict[str, object] | None:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json") :]
    if cleaned.endswith("```"):
        cleaned = cleaned[: -len("```")]
    try:
        value = json.loads(cleaned.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def make_memory(subtasks: list[dict[str, object]], boundary_index: int) -> dict[str, object]:
    if boundary_index == 1:
        summary = "The eraser is securely grasped bottom-side up; the vase has not yet been engaged."
        phase = "move the eraser in front of the vase"
        state = {"eraser_held": True, "eraser_at_vase": False, "vase_stabilized": False}
    elif boundary_index == 2:
        summary = "The eraser is securely grasped and positioned in front of the vase; the vase is not yet stabilized."
        phase = "stabilize the vase by grasping its upper rim"
        state = {"eraser_held": True, "eraser_at_vase": True, "vase_stabilized": False}
    elif boundary_index == 3:
        summary = "The eraser is ready in front and the vase is stabilized by its upper rim; no surface has yet been wiped."
        phase = "wipe the initial reachable writing"
        state = {"eraser_held": True, "eraser_at_vase": True, "vase_stabilized": True}
    elif boundary_index >= 14:
        summary = "The vase is clean around its full circuit and the eraser has been released; the vase may now be moved away."
        phase = "move the vase farther away and retract"
        state = {"eraser_held": False, "eraser_released": True, "vase_stabilized": True}
    elif boundary_index == 13:
        summary = "Cleaning is complete and the black marker returned to its initial orientation; the eraser is still held and must be released before moving the vase."
        phase = "retract and release the eraser"
        state = {"eraser_held": True, "eraser_released": False, "vase_stabilized": True}
    elif str(subtasks[boundary_index - 1]["current_subtask"]) == LABELS[3]:
        summary = "The eraser and vase remain held; writing on the current camera-facing surface has been removed, but the full circuit is unfinished."
        phase = "inspect the current front; rotate if it is clean, otherwise wipe"
        state = {
            "rotation_motion_complete": False,
            "current_front_requires_inspection": True,
            "consecutive_rotation_allowed": True,
        }
    else:
        summary = "The previous counterclockwise rotation is complete and exposed a new front-facing surface; the full circuit is unfinished."
        phase = "inspect the current front; wipe visible writing or rotate again if it is clean"
        state = {
            "rotation_motion_complete": True,
            "current_front_requires_inspection": True,
            "consecutive_rotation_allowed": True,
        }
    return {
        "progress_summary": summary,
        "marker_status": "returned" if boundary_index >= 13 else "not_returned",
        "full_circuit_complete": boundary_index >= 13,
        "next_required_phase": phase,
        **state,
    }


def infer(
    model: PreTrainedModel,
    processor: AutoProcessor,
    image_paths: list[Path],
    payload: dict[str, object],
    max_new_tokens: int,
    compact: bool = False,
    system_message: str | None = None,
) -> tuple[str, dict[str, float]]:
    active_system = (
        system_message
        if system_message is not None
        else (COMPACT_SYSTEM if compact else SYSTEM)
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": active_system}]},
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": str(path.resolve())} for path in image_paths],
                {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
            ],
        },
    ]
    start = time.perf_counter()
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to("cuda:0")
    synchronize()
    prepared = time.perf_counter()
    streamer = TextIteratorStreamer(
        processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=None
    )
    generation_error: list[BaseException] = []

    def generate() -> None:
        try:
            with torch.inference_mode():
                model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    streamer=streamer,
                )
        except BaseException as error:  # propagate worker failures to the caller
            generation_error.append(error)

    worker = threading.Thread(target=generate)
    worker.start()
    chunks: list[str] = []
    decision_ready_at: float | None = None
    for chunk in streamer:
        chunks.append(chunk)
        partial = "".join(chunks)
        if decision_ready_at is None and re.search(
            r'"current_subtask"\s*:\s*"[^"]+"', partial
        ):
            synchronize()
            decision_ready_at = time.perf_counter()
    worker.join()
    if generation_error:
        raise generation_error[0]
    synchronize()
    generated_at = time.perf_counter()
    response = "".join(chunks).strip()
    finished = time.perf_counter()
    return response, {
        "preprocess_s": prepared - start,
        "subtask_ready_s": (
            generated_at - prepared if decision_ready_at is None else decision_ready_at - prepared
        ),
        "generate_s": generated_at - prepared,
        "decode_s": finished - generated_at,
        "end_to_end_s": finished - start,
    }


def stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": ordered[p95_index],
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    offsets = args.offset or [0.33, 1.0]
    timeline_payload = json.loads(args.timeline.read_text(encoding="utf-8"))
    subtasks = timeline_payload["parsed"]["subtasks"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output_dir / "frames"
    model_load_start = time.perf_counter()
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    synchronize()
    model_load_s = time.perf_counter() - model_load_start

    # Explicit warmup excluded from reported latency.
    warmup_path = frame_dir / "warmup_76.330.png"
    extract_frame(args.video, 76.33, warmup_path)
    warmup_payload = {
        "task": "wipe the vase",
        "allowed_subtasks": LABELS,
        "prior_memory": make_memory(subtasks, 6),
        "observation_order": "one current frame",
    }
    _, warmup_timing = infer(
        model, processor, [warmup_path], warmup_payload, args.max_new_tokens, args.compact
    )

    cases: list[dict[str, object]] = []
    boundary_indices = args.boundary or list(range(1, len(subtasks)))
    for boundary_index in boundary_indices:
        boundary_s = float(subtasks[boundary_index]["start_s"])
        for offset_s in offsets:
            decision_s = boundary_s + offset_s
            current_path = frame_dir / f"b{boundary_index:02d}_o{offset_s:.2f}_{decision_s:.3f}.png"
            extract_frame(args.video, decision_s, current_path)
            payload = {
                "task": "wipe the vase",
                "allowed_subtasks": LABELS,
                "prior_memory": make_memory(subtasks, boundary_index),
                "observation_order": "one current frame; cross-frame state is summarized in prior_memory",
                "decision_time_s": decision_s,
            }
            response, timing = infer(
                model,
                processor,
                [current_path],
                payload,
                args.max_new_tokens,
                args.compact,
            )
            parsed = parse_response(response)
            if parsed is None:
                prediction = None
            elif args.compact:
                prediction = parsed.get("current_subtask")
            else:
                prediction = parsed.get("current_subtask")
            expected = subtasks[boundary_index]["current_subtask"]
            case = {
                "boundary_index": boundary_index,
                "boundary_s": boundary_s,
                "offset_s": offset_s,
                "decision_s": decision_s,
                "previous_subtask": subtasks[boundary_index - 1]["current_subtask"],
                "expected_subtask": expected,
                "predicted_subtask": prediction,
                "exact_match": prediction == expected,
                "timing": timing,
                "images": [str(current_path.resolve())],
                "prior_memory": payload["prior_memory"],
                "raw_response": response,
                "parsed_response": parsed,
            }
            cases.append(case)
            (args.output_dir / "cases.partial.json").write_text(
                json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"boundary={boundary_index:02d} offset={offset_s:.2f}s "
                f"match={case['exact_match']} e2e={timing['end_to_end_s']:.3f}s"
            )
    latency_values = [float(case["timing"]["end_to_end_s"]) for case in cases]  # type: ignore[index]
    generate_values = [float(case["timing"]["generate_s"]) for case in cases]  # type: ignore[index]
    subtask_ready_values = [float(case["timing"]["subtask_ready_s"]) for case in cases]  # type: ignore[index]
    per_offset: dict[str, object] = {}
    for offset_s in offsets:
        selected = [case for case in cases if case["offset_s"] == offset_s]
        per_offset[str(offset_s)] = {
            "correct": sum(bool(case["exact_match"]) for case in selected),
            "total": len(selected),
            "accuracy": sum(bool(case["exact_match"]) for case in selected) / len(selected),
        }
    first_correct: list[dict[str, object]] = []
    for boundary_index in boundary_indices:
        selected = [case for case in cases if case["boundary_index"] == boundary_index]
        correct = [case for case in selected if case["exact_match"]]
        first = min(correct, key=lambda case: float(case["offset_s"])) if correct else None
        first_correct.append(
            {
                "boundary_index": boundary_index,
                "first_correct_offset_s": None if first is None else first["offset_s"],
                "online_ready_after_s": (
                    None
                    if first is None
                    else float(first["offset_s"]) + float(first["timing"]["subtask_ready_s"])  # type: ignore[index]
                ),
                "memory_ready_after_s": (
                    None
                    if first is None
                    else float(first["offset_s"]) + float(first["timing"]["end_to_end_s"])  # type: ignore[index]
                ),
            }
        )
    summary = {
        "system_message": COMPACT_SYSTEM if args.compact else SYSTEM,
        "protocol": "compact_id_memory" if args.compact else "full_label_memory",
        "model": str(args.model.resolve()),
        "video": str(args.video.resolve()),
        "timeline": str(args.timeline.resolve()),
        "model_load_s_excluded": model_load_s,
        "warmup_timing_excluded": warmup_timing,
        "offsets_s": offsets,
        "case_count": len(cases),
        "latency_end_to_end_s": stats(latency_values),
        "latency_generate_s": stats(generate_values),
        "latency_subtask_ready_s": stats(subtask_ready_values),
        "per_offset": per_offset,
        "boundaries_correct_at_any_tested_offset": sum(
            item["first_correct_offset_s"] is not None for item in first_correct
        ),
        "total_boundaries": len(first_correct),
        "first_correct_by_boundary": first_correct,
    }
    (args.output_dir / "cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
