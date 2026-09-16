#!/usr/bin/env python3
"""Run Qwen only on the Cabinet FK intervals that could not reuse old prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from atomic_latent_vla.annotation.client import QwenVLPlusClient


SYSTEM = """You annotate one fixed robot-motion interval. Return one JSON object only:
{"low_level_instruction":"...","visual_evidence":"..."}.
The supplied FK gate is authoritative: preserve every move/rotate family and signed rotation axis,
but use the video for objects, task purpose, arm identity, and contact. Do not invent objects,
contact, forces, or directions that contradict FK. Write one concise 10-24 word English instruction.
For translation use natural words (forward/backward/left/right/up/down) or an explicit target-relative
direction; never use signed xyz for translation. For rotation, retain the signed base-frame axis."""

DIRECTION = {
    "move_x_pos": "forward", "move_x_neg": "backward",
    "move_y_pos": "left", "move_y_neg": "right",
    "move_z_pos": "up", "move_z_neg": "down",
}


def _expected_directions(request: dict) -> list[str]:
    return [DIRECTION[item["skill"]] for item in request["segment"]["gate_labels"] if item["skill"] in DIRECTION]


def _needs_translation_repair(request: dict, text: str) -> bool:
    expected = _expected_directions(request)
    if not expected:
        return False
    lowered = text.lower()
    normalized = lowered.replace("upward", "up").replace("downward", "down").replace("leftward", "left").replace("rightward", "right")
    if any(word not in normalized for word in expected):
        return True
    if any(token in normalized for token in ("+x", "-x", "+y", "-y", "+z", "-z", "negative x", "positive x", "negative y", "positive y", "negative z", "positive z", "along x", "along y", "along z")):
        return True
    return False


def _user_text(request: dict) -> str:
    segment = request["segment"]
    labels = ", ".join(item["skill"] for item in segment["gate_labels"])
    directions = _expected_directions(request)
    direction_rule = ""
    if directions:
        direction_rule = (
            "\nIMMUTABLE translation directions: " + ", ".join(directions) + ". "
            "The instruction MUST explicitly contain every listed direction word. "
            "Never attach x/y/z, signs, positive, or negative to a translation."
        )
    return f"""Task: {request['task']}
Episode context: {request['global_description']}
Coordinate convention: {request['axis_convention']}
Fixed interval: {segment['start_s']:.3f}s to {segment['end_s']:.3f}s.
Fixed FK mode: {segment['gate_mode']}.
Fixed FK atoms: {labels}.
FK weights: {segment['gate_weights']}.
FK audit reason: {segment['gate_reason']}.
Watch only the supplied short montage for this interval and return the two requested fields.{direction_rule}"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=Path("/home/admin123/ckrc/cabinet_latest_qwen_split"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max", type=int, default=None, help="process at most this many new rows")
    parser.add_argument("--repair-from", type=Path, default=None)
    parser.add_argument("--only-translation-repairs", action="store_true")
    parser.add_argument("--model", default="qwen3-vl-plus")
    args = parser.parse_args()
    package = args.package.resolve()
    manifest = package / "qwen_relabel_manifest.jsonl"
    output = args.output.resolve() if args.output else package / "qwen_relabel_results.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    done: set[str] = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line:
                done.add(json.loads(line)["request_id"])
    previous: dict[str, dict] = {}
    if args.repair_from is not None:
        previous = {
            row["request_id"]: row
            for row in (json.loads(line) for line in args.repair_from.read_text(encoding="utf-8").splitlines() if line)
            if row.get("ok")
        }
    client = QwenVLPlusClient(model=args.model)
    new_count = 0
    with output.open("a", encoding="utf-8") as stream:
        for request in rows:
            if request["request_id"] in done:
                continue
            if args.only_translation_repairs:
                old = previous.get(request["request_id"])
                if old is None or not _needs_translation_repair(request, old["response"]["low_level_instruction"]):
                    continue
            clip = package / request["video_clip"]
            messages = [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": str(clip), "fps": request["video_sample_fps"], "max_pixels": 401408},
                        {"type": "text", "text": _user_text(request)},
                    ],
                },
            ]
            try:
                response, usage = client.complete_json(messages, max_tokens=300)
                instruction = response.get("low_level_instruction")
                evidence = response.get("visual_evidence")
                if not isinstance(instruction, str) or not instruction.strip():
                    raise ValueError("response lacks low_level_instruction")
                if not isinstance(evidence, str):
                    raise ValueError("response lacks visual_evidence")
                if args.only_translation_repairs and _needs_translation_repair(request, instruction):
                    raise ValueError("response still violates required natural translation direction")
                record = {"request_id": request["request_id"], "ok": True, "response": response, "usage": usage}
            except Exception as error:
                record = {"request_id": request["request_id"], "ok": False, "error": str(error)}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            new_count += 1
            if new_count % 10 == 0:
                print(f"processed {new_count} new, total_done={len(done) + new_count}", flush=True)
            if args.max is not None and new_count >= args.max:
                break


if __name__ == "__main__":
    main()
