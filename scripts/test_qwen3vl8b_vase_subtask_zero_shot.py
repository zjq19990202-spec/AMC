#!/usr/bin/env python3
"""Zero-shot closed-vocabulary vase-subtask test for local Qwen3-VL-8B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


LABELS = [
    "Move to the blackboard eraser, tip it bottom-side up, and grasp it",
    "Move the blackboard eraser in front of the vase",
    "Grasp the upper rim of the vase and hold it in place",
    "Wipe the writing in front of the vase with the blackboard eraser using an up-and-down motion",
    "Rotate the vase counterclockwise, then press it down onto the table",
    "Retract and release the blackboard eraser",
    "Move the vase farther away and retract the arm",
]


SYSTEM = """You are the online task-state tracker for a vase-wiping robot. Infer the
single current subtask from recent camera observations and persistent memory.

Closed-vocabulary constraint: `current_subtask` MUST be copied character-for-character from
the supplied `allowed_subtasks`. Never invent, shorten, merge, translate, or paraphrase a label.

Task policy:
- The goal is to erase all writing around the vase, not merely the currently visible front.
- Use the black feature/mark on the vase's upper rim as the rotation reference. A full circuit is
  complete only after that reference has returned to its initial camera-facing orientation.
- If a full circuit is not complete and the surface currently facing the front camera has no
  visible writing, select the counterclockwise-rotation subtask so an uninspected surface can be
  exposed for wiping.
- If writing is visible at the camera-facing surface and the eraser and vase are already held in
  wiping configuration, select the wiping subtask.
- Do not mark a subtask completed from intent alone; require visual evidence of its result.

Return JSON only, with exactly these keys:
{
  "current_subtask": "EXACT_ALLOWED_LABEL",
  "memory_to_store": {
    "completed_progress": ["concise durable facts supported so far"],
    "rotation_reference": "concise state of upper-rim black reference",
    "full_circuit_complete": false,
    "front_writing_visible": false
  },
  "visual_evidence": "one concise sentence",
  "decision_reason": "one concise sentence"
}
The memory is for the next control step. Preserve prior completed facts, update only visually
supported state, and never store the chosen current subtask as completed before it finishes."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/home/admin123/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--infer-circuit",
        action="store_true",
        help="do not tell the model whether the rim reference has completed a circuit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = [path.resolve() for path in args.image]
    prior_memory: dict[str, object] = {
        "completed_progress": [
            "The eraser was grasped bottom-side up and moved in front of the vase.",
            "The vase upper rim was grasped and stabilized.",
            "Writing on the previously camera-facing surface was wiped away.",
            "Earlier counterclockwise rotations exposed additional surfaces, whose visible writing was wiped away.",
        ],
    }
    if args.infer_circuit:
        prior_memory["rotation_reference"] = (
            "The first supplied image is the initial upper-rim reference. Compare it with the "
            "latest image and infer whether the reference has returned."
        )
        prior_memory["full_circuit_complete"] = "unknown; infer from the supplied images"
    else:
        prior_memory["rotation_reference"] = (
            "The upper-rim black reference has advanced through part of the circuit but has not yet "
            "been observed returning to its initial camera-facing orientation."
        )
        prior_memory["full_circuit_complete"] = False
    payload = {
        "task": "wipe the vase",
        "allowed_subtasks": LABELS,
        "prior_memory": prior_memory,
        "observation_order": (
            "first image is the initial upper-rim reference; remaining images are recent "
            "observations in oldest-to-newest order; decide at the final image"
            if args.infer_circuit
            else "oldest_to_newest; decide the subtask at the final image"
        ),
    }
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": str(path)} for path in images],
                {"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)},
            ],
        },
    ]

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to("cuda:0")
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    response = processor.batch_decode(
        [output[len(source) :] for source, output in zip(inputs.input_ids, generated, strict=True)],
        skip_special_tokens=True,
    )[0].strip()
    result = {
        "model": str(args.model.resolve()),
        "images": [str(path) for path in images],
        "system_message": SYSTEM,
        "user_payload": payload,
        "generation": {"do_sample": False, "max_new_tokens": 512},
        "infer_circuit": args.infer_circuit,
        "raw_response": response,
        "expected_current_subtask": LABELS[4],
        "exact_match": False,
    }
    try:
        parsed = json.loads(response.removeprefix("```json").removesuffix("```").strip())
        result["parsed_response"] = parsed
        result["exact_match"] = parsed.get("current_subtask") == LABELS[4]
    except (json.JSONDecodeError, AttributeError) as error:
        result["parse_error"] = str(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(response)
    print(f"\nexact_match={result['exact_match']} output={args.output}")


if __name__ == "__main__":
    main()
