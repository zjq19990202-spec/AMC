#!/usr/bin/env python3
"""Probe Qwen3.5 subtask + atomic intervention decisions on stalled Vase training frames."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForMultimodalLM, AutoProcessor


ATOMS = {
    "no_steer": "do not intervene",
    "move_x_pos": "move TCP along base-frame +x",
    "move_x_neg": "move TCP along base-frame -x",
    "move_y_pos": "move TCP along base-frame +y",
    "move_y_neg": "move TCP along base-frame -y",
    "move_z_pos": "move TCP along base-frame +z",
    "move_z_neg": "move TCP along base-frame -z",
    "rotate_z_pos": "rotate TCP positively about base-frame z by the right-hand rule",
    "rotate_z_neg": "rotate TCP negatively about base-frame z by the right-hand rule",
}

SYSTEM = """You monitor one robot performing the Vase task. Use exactly one current base-camera image,
the correct current subtask, and compact semantic memory. The robot has already executed two fresh
action chunks in the same subtask and remained at the visible pose without reducing its task-related
error. Keep the current subtask unchanged unless it is visually complete.

Decide whether one short atomic intervention would help the current subtask resume. Select only from
the supplied closed atomic vocabulary. Directions are in the robot base frame, not image coordinates.
Use no_steer if the correction direction is not visually grounded, the current state is safe and
normally progressing, the action is a gripper-only transition, or an atomic correction could damage
contact. An intervention is one-shot; the original subtask resumes after it.

Return valid one-line JSON only:
{"current_subtask":"EXACT_CURRENT_SUBTASK","progress_status":"stalled|completed|uncertain",
"intervention":{"required":false,"target_arm":"left|right|none",
"atomic_skill":"EXACT_ALLOWED_ATOM","reason":"MAX 12 WORDS","confidence":0.0},
"memory_to_store":{"progress_summary":"MAX 12 WORDS","progress_trend":"unchanged|complete|unknown",
"next_required_phase":"MAX 6 WORDS"}}"""


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--dataset-meta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    return parser.parse_args()


def extract(video: Path, second: float, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-ss", f"{second:.6f}", "-i", str(video),
         "-frames:v", "1", "-y", str(output)], check=True
    )


def parse(text: str) -> dict | None:
    value = text.strip()
    if value.startswith("```json"):
        value = value[7:]
    if value.endswith("```"):
        value = value[:-3]
    try:
        parsed = json.loads(value.strip())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def select_cases(meta: Path, episode: int) -> list[dict]:
    rows = [json.loads(line) for line in (meta / "episode_subtasks.jsonl").open()]
    subtasks = next(row["semantic_segments"] for row in rows if row["episode_index"] == episode)
    arm_segments = {
        arm: json.loads((meta / f"fk_horizon_3hz_gate_top5_stay_v2/{arm}/episode_{episode:06d}.json").read_text())["segments"]
        for arm in ("left", "right")
    }
    allowed = set(ATOMS) - {"no_steer"}
    cases = []
    for index, subtask in enumerate(subtasks):
        start = subtask["start_frame_30hz"] / 30.0
        end = subtask["end_frame_30hz_exclusive"] / 30.0
        candidates = []
        for arm, segments in arm_segments.items():
            for segment in segments:
                labels = [label["skill"] for label in segment.get("gate_labels", [])]
                if (
                    segment["start_s"] >= start + 1.0
                    and segment["end_s"] <= end - 1.0
                    and len(labels) == 1
                    and labels[0] in allowed
                ):
                    candidates.append((arm, labels[0], segment))
        if not candidates:
            continue
        counts = Counter((arm, atom) for arm, atom, _ in candidates)
        dominant_arm, dominant_atom = counts.most_common(1)[0][0]
        matching = [item for item in candidates if item[:2] == (dominant_arm, dominant_atom)]
        midpoint = (start + end) / 2
        arm, atom, segment = min(
            matching, key=lambda item: abs((item[2]["start_s"] + item[2]["end_s"]) / 2 - midpoint)
        )
        cases.append({
            "segment_index": index,
            "subtask": subtask["current_subtask"],
            "segment_start_s": start,
            "segment_end_s": end,
            "decision_s": (segment["start_s"] + segment["end_s"]) / 2,
            "fk_arm": arm,
            "fk_atom": atom,
            "fk_gate_reason": segment.get("gate_reason"),
        })
    return cases


def main() -> None:
    cfg = args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cases = select_cases(cfg.dataset_meta, cfg.episode)
    model = AutoModelForMultimodalLM.from_pretrained(
        cfg.model, dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(cfg.model, local_files_only=True)
    results = []
    for case in cases:
        frame = cfg.output_dir / "frames" / f"segment_{case['segment_index']:02d}_{case['decision_s']:.3f}.png"
        extract(cfg.video, case["decision_s"], frame)
        payload = {
            "current_subtask": case["subtask"],
            "allowed_atomic_skills": ATOMS,
            "semantic_memory": {
                "progress_summary": "Current task-relevant pose persisted for two completed action chunks.",
                "progress_trend": "unchanged",
                "normal_chunks_without_progress": 2,
                "last_atomic_intervention": "no_steer",
                "last_intervention_result": "not_applied",
            },
        }
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [
                {"type": "image", "image": str(frame.resolve())},
                {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
            ]},
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to("cuda:0")
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=cfg.max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        latency = time.perf_counter() - started
        raw = processor.batch_decode(
            [generated[0][inputs.input_ids.shape[1]:]], skip_special_tokens=True
        )[0].strip()
        parsed = parse(raw)
        intervention = {} if parsed is None else parsed.get("intervention", {})
        predicted_atom = intervention.get("atomic_skill") if isinstance(intervention, dict) else None
        predicted_arm = intervention.get("target_arm") if isinstance(intervention, dict) else None
        case.update({
            "frame": str(frame.resolve()),
            "raw_response": raw,
            "parsed_response": parsed,
            "predicted_subtask": None if parsed is None else parsed.get("current_subtask"),
            "subtask_match": parsed is not None and parsed.get("current_subtask") == case["subtask"],
            "intervention_required": intervention.get("required") if isinstance(intervention, dict) else None,
            "predicted_arm": predicted_arm,
            "predicted_atom": predicted_atom,
            "fk_atom_match": predicted_atom == case["fk_atom"] and predicted_arm == case["fk_arm"],
            "latency_s": latency,
        })
        results.append(case)
        print(case["segment_index"], case["fk_arm"], case["fk_atom"], "->", predicted_arm, predicted_atom)
    (cfg.output_dir / "cases.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    summary = {
        "case_count": len(results),
        "subtask_correct": sum(bool(row["subtask_match"]) for row in results),
        "intervention_triggered": sum(row["intervention_required"] is True for row in results),
        "fk_exact_matches": sum(bool(row["fk_atom_match"]) for row in results),
        "mean_latency_s": sum(row["latency_s"] for row in results) / len(results),
        "system_message": SYSTEM,
    }
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
