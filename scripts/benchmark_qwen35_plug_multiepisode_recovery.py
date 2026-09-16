#!/usr/bin/env python3
"""Evaluate Qwen3.5-4B plug-subtask planning and post-error recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from benchmark_qwen3vl8b_vase_memory_switches import (
    extract_frame,
    infer,
    parse_response,
    stats,
)


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

LABEL_ALIASES = {
    **{label: label for label in LABELS},
    "适当轻推插头使其牢固": LABELS[6],
    "适当按压插头使其稳固服帖插排": LABELS[6],
}

VIEW_TO_VIDEO = {
    "base": "observation.images.base_0_rgb",
    "left": "observation.images.left_wrist_0_rgb",
    "right": "observation.images.right_wrist_0_rgb",
}

SYSTEM = """你是双臂机器人“把充电插头插入插排”任务的在线状态跟踪器。每次只选择一个当前
可执行子任务，并从 allowed_subtasks 中逐字复制，禁止改写、合并或创造标签。输入包含当前
相机图像和一份紧凑语义记忆；记忆中明确完成的历史状态可信，但当前图像决定距离、朝向、
接触和是否完全插牢。

严格遵循以下策略：
1. 插头未抓住时，靠近并抓取插头。
2. 插头已抓住但尚未抬起到稳定操作高度时，将其举起并停在胸前。
3. 插头已抬起但充电端尚未朝向插排时，旋转调整插头朝向。
4. 朝向正确但插头仍明显远离插排时，将插头移动到插排上方；已经足够近时不要重复该步。
5. 插入前必须用另一只手抓住并稳定插排。插排已稳定时不要再次选择抓插排。
6. 插头方向正确、靠近插孔且插排已稳定后，选择对准并插入。对准过程中的轻微接触不等于
   插入完成；看不清是否到位时继续对准插入，不要仅凭动作意图宣布完成。
7. 插头已经进入插孔但仍有明显缝隙、倾斜或未贴合时，选择适当按压使其稳固服帖。
8. 只有图像确认插头已端正、完全贴合并保持稳定后，才选择任务完成后缩回。
9. 不要强制每个 episode 都出现“移动到上方”或“按压”步骤；当前几何状态已经满足时可以跳过。

返回紧凑有效 JSON，且只返回 JSON：
{"current_subtask":"EXACT_ALLOWED_LABEL","memory_to_store":{"progress_summary":"不超过20个中文词",
"plug_held":false,"plug_lifted":false,"plug_oriented":false,"strip_stabilized":false,
"insertion_state":"not_started/alignment/contact/seated","next_required_phase":"不超过8个中文词"},
"visual_evidence":"不超过20个中文词"}
不要把刚选择但尚未完成的子任务写成已完成。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--timeline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, action="append", required=True)
    parser.add_argument("--view", choices=sorted(VIEW_TO_VIDEO), action="append", default=[])
    parser.add_argument("--first-offset", type=float, default=0.33)
    parser.add_argument("--decision-period", type=float, default=0.8)
    parser.add_argument("--max-recovery-steps", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--episode-selection-note",
        default="fixed before inference",
        help="Human-readable selection rule stored in the evaluation contract.",
    )
    return parser.parse_args()


def canonical_label(label: str) -> str:
    try:
        return LABEL_ALIASES[label]
    except KeyError as error:
        raise ValueError(f"unmapped plug subtask label: {label!r}") from error


def make_memory(segments: list[dict[str, object]], boundary_index: int) -> dict[str, object]:
    completed = [canonical_label(str(item["current_subtask"])) for item in segments[:boundary_index]]
    completed_set = set(completed)
    insertion_attempted = LABELS[5] in completed_set
    seating_confirmed = LABELS[6] in completed_set
    if seating_confirmed:
        insertion_state = "seated"
        summary = "插头已压紧并确认端正贴合"
    elif insertion_attempted:
        insertion_state = "contact"
        summary = "已尝试插入，是否完全贴合需由当前图像确认"
    elif LABELS[4] in completed_set:
        insertion_state = "alignment"
        summary = "插排已固定，插头待对准插孔并插入"
    elif LABELS[2] in completed_set:
        insertion_state = "not_started"
        summary = "插头已抓取抬起并调整朝向，需检查与插排距离"
    elif LABELS[1] in completed_set:
        insertion_state = "not_started"
        summary = "插头已抓取并抬起，尚需调整充电端朝向"
    else:
        insertion_state = "not_started"
        summary = "插头已抓取，尚需抬起到稳定操作高度"
    return {
        "progress_summary": summary,
        "completed_subtasks": completed,
        "plug_held": LABELS[0] in completed_set,
        "plug_lifted": LABELS[1] in completed_set,
        "plug_oriented": LABELS[2] in completed_set,
        "plug_near_strip": LABELS[3] in completed_set or LABELS[4] in completed_set,
        "strip_stabilized": LABELS[4] in completed_set,
        "insertion_state": insertion_state,
    }


def prediction_from_response(response: str, parsed: dict[str, object] | None) -> str | None:
    prediction: object | None = None
    if parsed is not None:
        prediction = parsed.get("current_subtask")
    if not isinstance(prediction, str):
        match = re.search(r'"current_subtask"\s*:\s*"([^"]+)"', response)
        prediction = match.group(1) if match else None
    if not isinstance(prediction, str):
        return None
    return LABEL_ALIASES.get(prediction, prediction)


def valid_memory(parsed: dict[str, object] | None) -> bool:
    if parsed is None or not isinstance(parsed.get("memory_to_store"), dict):
        return False
    memory = parsed["memory_to_store"]
    required_strings = ("progress_summary", "insertion_state", "next_required_phase")
    required_bools = ("plug_held", "plug_lifted", "plug_oriented", "strip_stabilized")
    return (
        all(isinstance(memory.get(key), str) for key in required_strings)
        and all(isinstance(memory.get(key), bool) for key in required_bools)
        and memory.get("insertion_state")
        in {"not_started", "alignment", "contact", "seated"}
    )


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float]:
    if total <= 0:
        return {"low": float("nan"), "high": float("nan")}
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * ((rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) ** 0.5)
        / denominator
    )
    return {"low": center - half_width, "high": center + half_width}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    views = args.view or ["base"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output_dir / "frames"
    specs: list[dict[str, object]] = []
    for episode_id in args.episode:
        timeline = (
            args.timeline_root
            / f"episode_{episode_id:06d}"
            / "result.v11_final_memory.json"
        )
        payload = json.loads(timeline.read_text(encoding="utf-8"))
        segments = payload["semantic_segments"]
        canonical_sequence = [canonical_label(str(item["current_subtask"])) for item in segments]
        videos = {
            view: args.dataset_root / "videos" / VIEW_TO_VIDEO[view] / "chunk-000" / f"file-{episode_id:03d}.mp4"
            for view in views
        }
        missing = [str(path) for path in videos.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing videos: {missing}")
        specs.append(
            {
                "episode_id": episode_id,
                "timeline": timeline,
                "segments": segments,
                "canonical_sequence": canonical_sequence,
                "videos": videos,
            }
        )

    selection_manifest = {
        "selection_rule": args.episode_selection_note,
        "episodes": [
            {
                "episode_id": int(spec["episode_id"]),
                "timeline": str(Path(spec["timeline"]).resolve()),
                "timeline_sha256": sha256(Path(spec["timeline"])),
                "canonical_sequence": spec["canonical_sequence"],
                "views": {
                    view: str(Path(path).resolve())
                    for view, path in spec["videos"].items()
                },
            }
            for spec in specs
        ],
    }
    (args.output_dir / "selection_manifest.json").write_text(
        json.dumps(selection_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "run_contract.md").write_text(
        "# Qwen3.5-4B Plug planner evaluation contract\n\n"
        f"- Model: `{args.model.resolve()}`\n"
        f"- Dataset: `{args.dataset_root.resolve()}`\n"
        f"- Timeline root: `{args.timeline_root.resolve()}`\n"
        f"- Episode selection: {args.episode_selection_note}\n"
        f"- Episodes: {len(specs)}; views: {', '.join(views)}\n"
        f"- First decision: boundary + {args.first_offset:.2f} s\n"
        f"- Retry period: {args.decision_period:.2f} s; maximum retries: {args.max_recovery_steps}\n"
        "- Memory: teacher-forced compact semantic state at each GT boundary; after an initial error, "
        "the model-generated memory is fed into the next retry within that GT segment.\n"
        "- Score: exact canonical subtask label. Consecutive duplicate canonical labels are not new boundaries.\n",
        encoding="utf-8",
    )

    load_start = time.perf_counter()
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_s = time.perf_counter() - load_start

    warmup_spec = specs[0]
    warmup_segments = warmup_spec["segments"]
    warmup_boundary = min(2, len(warmup_segments) - 1)
    warmup_s = float(warmup_segments[warmup_boundary]["start_s"]) + args.first_offset
    warmup_images: list[Path] = []
    for view in views:
        path = frame_dir / f"warmup_{view}.png"
        extract_frame(Path(warmup_spec["videos"][view]), warmup_s, path)
        warmup_images.append(path)
    warmup_payload = {
        "task": "把充电插头插入插排",
        "allowed_subtasks": LABELS,
        "prior_memory": make_memory(warmup_segments, warmup_boundary),
        "observation_order": f"同一时刻的相机视图，顺序为 {views}",
    }
    _, warmup_timing = infer(
        model,
        processor,
        warmup_images,
        warmup_payload,
        args.max_new_tokens,
        False,
        SYSTEM,
    )

    cases: list[dict[str, object]] = []
    boundaries: list[dict[str, object]] = []
    skipped_duplicate_boundaries: list[dict[str, object]] = []
    for spec in specs:
        episode_id = int(spec["episode_id"])
        segments = spec["segments"]
        for boundary_index in range(1, len(segments)):
            boundary_s = float(segments[boundary_index]["start_s"])
            segment_end_s = float(segments[boundary_index]["end_s"])
            expected = canonical_label(str(segments[boundary_index]["current_subtask"]))
            previous = canonical_label(str(segments[boundary_index - 1]["current_subtask"]))
            if expected == previous:
                skipped_duplicate_boundaries.append(
                    {
                        "episode_id": episode_id,
                        "boundary_index": boundary_index,
                        "canonical_subtask": expected,
                        "reason": "consecutive duplicate canonical label",
                    }
                )
                continue
            prior_memory = make_memory(segments, boundary_index)
            attempts: list[dict[str, object]] = []
            for recovery_step in range(args.max_recovery_steps + 1):
                offset_s = args.first_offset + recovery_step * args.decision_period
                decision_s = boundary_s + offset_s
                if decision_s >= segment_end_s - 0.05:
                    break
                images: list[Path] = []
                for view in views:
                    image_path = (
                        frame_dir
                        / f"ep{episode_id:03d}_b{boundary_index:02d}_r{recovery_step}_{view}_{decision_s:.3f}.png"
                    )
                    extract_frame(Path(spec["videos"][view]), decision_s, image_path)
                    images.append(image_path)
                user_payload = {
                    "task": "把充电插头插入插排",
                    "allowed_subtasks": LABELS,
                    "prior_memory": prior_memory,
                    "observation_order": f"同一时刻的相机视图，顺序为 {views}",
                    "decision_time_s": decision_s,
                }
                response, timing = infer(
                    model,
                    processor,
                    images,
                    user_payload,
                    args.max_new_tokens,
                    False,
                    SYSTEM,
                )
                parsed = parse_response(response)
                prediction = prediction_from_response(response, parsed)
                exact_match = prediction == expected
                case = {
                    "episode_id": episode_id,
                    "boundary_index": boundary_index,
                    "recovery_step": recovery_step,
                    "boundary_s": boundary_s,
                    "segment_end_s": segment_end_s,
                    "offset_s": offset_s,
                    "decision_s": decision_s,
                    "views": views,
                    "previous_subtask": previous,
                    "expected_subtask": expected,
                    "predicted_subtask": prediction,
                    "exact_match": exact_match,
                    "json_valid": parsed is not None,
                    "allowed_label_valid": prediction in LABELS,
                    "memory_valid": valid_memory(parsed),
                    "timing": timing,
                    "images": [str(path.resolve()) for path in images],
                    "prior_memory": user_payload["prior_memory"],
                    "raw_response": response,
                    "parsed_response": parsed,
                }
                attempts.append(case)
                cases.append(case)
                if exact_match:
                    break
                if parsed is not None and isinstance(parsed.get("memory_to_store"), dict):
                    prior_memory = parsed["memory_to_store"]
            first = attempts[0] if attempts else None
            first_correct = next((item for item in attempts if item["exact_match"]), None)
            boundaries.append(
                {
                    "episode_id": episode_id,
                    "boundary_index": boundary_index,
                    "expected_subtask": expected,
                    "initial_exact_match": bool(first and first["exact_match"]),
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
            result = boundaries[-1]
            print(
                f"episode={episode_id:03d} boundary={boundary_index:02d} "
                f"initial={result['initial_exact_match']} recovered={result['recovered']} "
                f"step={result['recovery_step']}"
            )

    initial_cases = [case for case in cases if int(case["recovery_step"]) == 0]
    initial_correct = sum(bool(item["initial_exact_match"]) for item in boundaries)
    errors = [item for item in boundaries if not item["initial_exact_match"]]
    recovered_errors = [item for item in errors if item["recovered"]]
    ready_latencies = [float(case["timing"]["subtask_ready_s"]) for case in initial_cases]
    per_episode: dict[str, object] = {}
    for episode_id in args.episode:
        selected = [item for item in boundaries if int(item["episode_id"]) == episode_id]
        correct = sum(bool(item["initial_exact_match"]) for item in selected)
        per_episode[str(episode_id)] = {
            "boundaries": len(selected),
            "initial_correct": correct,
            "initial_accuracy": correct / len(selected),
            "all_initially_correct": correct == len(selected),
            "all_correct_before_segment_end": all(bool(item["recovered"]) for item in selected),
            "unrecovered": sum(
                (not bool(item["initial_exact_match"])) and (not bool(item["recovered"]))
                for item in selected
            ),
        }
    initial_json_valid = sum(bool(case["json_valid"]) for case in initial_cases)
    initial_label_valid = sum(bool(case["allowed_label_valid"]) for case in initial_cases)
    initial_memory_valid = sum(bool(case["memory_valid"]) for case in initial_cases)
    episode_initial_success = sum(
        bool(item["all_initially_correct"]) for item in per_episode.values()
    )
    episode_eventual_success = sum(
        bool(item["all_correct_before_segment_end"]) for item in per_episode.values()
    )
    confusion: dict[str, dict[str, int]] = {}
    per_label: dict[str, dict[str, object]] = {}
    for expected in LABELS:
        selected = [case for case in initial_cases if case["expected_subtask"] == expected]
        if not selected:
            continue
        counts: dict[str, int] = {}
        for case in selected:
            predicted = str(case["predicted_subtask"])
            counts[predicted] = counts.get(predicted, 0) + 1
        confusion[expected] = counts
        label_correct = sum(bool(case["exact_match"]) for case in selected)
        per_label[expected] = {
            "correct": label_correct,
            "total": len(selected),
            "accuracy": label_correct / len(selected),
            "wilson95": wilson_interval(label_correct, len(selected)),
        }
    recovery_steps = [int(item["recovery_step"]) for item in recovered_errors]
    ready_after = [float(item["output_ready_after_boundary_s"]) for item in recovered_errors]
    summary = {
        "evaluation_contract": {
            "model": str(args.model.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "timeline_root": str(args.timeline_root.resolve()),
            "episodes": args.episode,
            "episode_selection": args.episode_selection_note,
            "views": views,
            "canonical_labels": LABELS,
            "teacher_forced_memory_at_boundary": True,
            "model_memory_fed_forward_after_error": True,
            "first_observation_offset_s": args.first_offset,
            "decision_period_s": args.decision_period,
            "maximum_recovery_steps": args.max_recovery_steps,
            "consecutive_duplicate_labels_scored_as_boundaries": False,
            "model_load_and_warmup_excluded_from_latency": True,
        },
        "system_message": SYSTEM,
        "model_load_s_excluded": model_load_s,
        "warmup_timing_excluded": warmup_timing,
        "episode_count": len(specs),
        "boundary_count": len(boundaries),
        "skipped_duplicate_boundary_count": len(skipped_duplicate_boundaries),
        "skipped_duplicate_boundaries": skipped_duplicate_boundaries,
        "initial_correct": initial_correct,
        "initial_accuracy": initial_correct / len(boundaries),
        "initial_accuracy_wilson95": wilson_interval(initial_correct, len(boundaries)),
        "initial_json_valid_rate": initial_json_valid / len(initial_cases),
        "initial_allowed_label_valid_rate": initial_label_valid / len(initial_cases),
        "initial_memory_valid_rate": initial_memory_valid / len(initial_cases),
        "initial_error_count": len(errors),
        "initial_error_recovered_count": len(recovered_errors),
        "correct_before_segment_end": initial_correct + len(recovered_errors),
        "correct_before_segment_end_rate": (initial_correct + len(recovered_errors)) / len(boundaries),
        "correct_before_segment_end_wilson95": wilson_interval(
            initial_correct + len(recovered_errors), len(boundaries)
        ),
        "episode_all_boundaries_initially_correct": episode_initial_success,
        "episode_all_boundaries_initially_correct_rate": episode_initial_success / len(specs),
        "episode_all_boundaries_correct_before_segment_end": episode_eventual_success,
        "episode_all_boundaries_correct_before_segment_end_rate": episode_eventual_success / len(specs),
        "initial_subtask_ready_latency_s": stats(ready_latencies),
        "recovered_error_steps": {
            "mean": statistics.mean(recovery_steps) if recovery_steps else None,
            "median": statistics.median(recovery_steps) if recovery_steps else None,
            "max": max(recovery_steps) if recovery_steps else None,
        },
        "recovered_error_output_ready_after_boundary_s": {
            "mean": statistics.mean(ready_after) if ready_after else None,
            "median": statistics.median(ready_after) if ready_after else None,
            "max": max(ready_after) if ready_after else None,
        },
        "per_label": per_label,
        "initial_confusion": confusion,
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
