#!/usr/bin/env python3
"""Rebind immutable Vase FK atomic labels to a reviewed subtask sidecar.

Rows contained by one reviewed subtask receive an object-aware relation.
Rows crossing a reviewed boundary retain only their immutable FK prefix.
The input row grouping and FK labels are never changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PREFIX = {
    "move_x_pos": "Move forward",
    "move_x_neg": "Move backward",
    "move_y_pos": "Move leftward",
    "move_y_neg": "Move rightward",
    "move_z_pos": "Move upward",
    "move_z_neg": "Move downward",
    "rotate_x_pos": "Rotate positive about base-frame +x",
    "rotate_x_neg": "Rotate negative about base-frame +x",
    "rotate_y_pos": "Rotate positive about base-frame +y",
    "rotate_y_neg": "Rotate negative about base-frame +y",
    "rotate_z_pos": "Rotate positive about base-frame +z",
    "rotate_z_neg": "Rotate negative about base-frame +z",
    "stay": "Stay stationary",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _prefix(labels: list[str]) -> str:
    return " and ".join(PREFIX[label] for label in labels)


def _relation(text: str, progress: float, stay: bool) -> str:
    text = text.strip().rstrip(".").lower()
    if stay:
        references = {
            "move to the blackboard eraser, tip it bottom-side up, and grasp it": "the blackboard eraser",
            "move the blackboard eraser in front of the vase": "the area in front of the vase",
            "grasp the upper rim of the vase and hold it in place": "the upper rim of the vase",
            "wipe the writing in front of the vase with the blackboard eraser using an up-and-down motion": "the writing area in front of the vase",
            "rotate the vase counterclockwise, then press it down onto the table": "the vase and tabletop",
            "retract and release the blackboard eraser": "the eraser release path",
            "move the vase farther away and retract the arm": "the vase release side of the tabletop",
        }
        return f"relative to {references[text]}"
    if text == "move to the blackboard eraser, tip it bottom-side up, and grasp it":
        return "toward the blackboard eraser" if progress < 0.48 else "around the blackboard eraser to orient its bottom side upward"
    if text == "move the blackboard eraser in front of the vase":
        return "toward the area in front of the vase" if progress < 0.62 else "in front of the vase"
    if text == "grasp the upper rim of the vase and hold it in place":
        return "toward the upper rim of the vase" if progress < 0.62 else "at the upper rim of the vase"
    if text == "wipe the writing in front of the vase with the blackboard eraser using an up-and-down motion":
        return "along the writing area in front of the vase"
    if text == "rotate the vase counterclockwise, then press it down onto the table":
        return "around the vase axis above the tabletop" if progress < 0.56 else "onto the tabletop beneath the vase"
    if text == "retract and release the blackboard eraser":
        return "away from the writing area toward the eraser release position"
    if text == "move the vase farther away and retract the arm":
        return "away from the writing area toward the vase release side"
    raise ValueError(f"unsupported reviewed Vase subtask: {text!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atomic-input", type=Path, required=True)
    parser.add_argument("--subtasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.audit.exists():
        raise FileExistsError("refusing to overwrite an existing output or audit")

    subtasks = {}
    for row in _read_jsonl(args.subtasks):
        subtasks[int(row["episode_index"])] = row["semantic_segments"]

    def subtask_at(episode: int, block: int) -> dict[str, Any]:
        frame = block * 10
        for segment in subtasks[episode]:
            if int(segment["start_frame_30hz"]) <= frame < int(segment["end_frame_30hz_exclusive"]):
                return segment
        return subtasks[episode][-1]

    output = []
    audit = []
    counts = Counter()
    for source in _read_jsonl(args.atomic_input):
        row = dict(source)
        episode = int(row["episode_index"])
        first = int(row["block_start_id"])
        last = int(row["block_end_id"])
        labels = list(row["fk_atomic_labels"])
        prefix = _prefix(labels)
        block_subtasks = [subtask_at(episode, block) for block in range(first, last + 1)]
        identities = {
            (int(item["start_frame_30hz"]), int(item["end_frame_30hz_exclusive"]), item["current_subtask"])
            for item in block_subtasks
        }
        if len(identities) > 1:
            prompt = prefix + "."
            context = {
                "kind": "cross_subtask_object_free",
                "subtasks": [
                    {"text": text, "start_s": start / 30.0, "end_s": end / 30.0}
                    for start, end, text in sorted(identities)
                ],
            }
            counts["cross_boundary_object_free"] += 1
            relation = None
        else:
            start, end, text = next(iter(identities))
            midpoint_s = ((first / 3.0) + ((last + 1) / 3.0)) / 2.0
            progress = min(1.0, max(0.0, (midpoint_s - start / 30.0) / max((end - start) / 30.0, 1e-6)))
            relation = _relation(text, progress, labels == ["stay"])
            prompt = f"{prefix} {relation}."
            context = {
                "kind": "within_subtask",
                "subtask": {"text": text, "source_text": text, "start_s": start / 30.0, "end_s": end / 30.0},
            }
            counts["within_subtask_object_aware"] += 1
        row["fk_prefix"] = prefix
        row["prompt"] = prompt
        row["subtask_context"] = context
        row["prompt_source"] = "fk_final_reviewed_subtask_relation_v3"
        output.append(row)
        audit.append(
            {
                "episode_index": episode,
                "horizon_id": row["horizon_id"],
                "arm": row["arm"],
                "block_start_id": first,
                "block_end_id": last,
                "fk_atomic_labels_unchanged": labels == source["fk_atomic_labels"],
                "source_prompt": source["prompt"],
                "final_prompt": prompt,
                "relation": relation,
                "context_kind": context["kind"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    args.audit.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit), encoding="utf-8")
    summary = {
        "rows": len(output),
        **counts,
        "atomic_input_sha256": hashlib.sha256(args.atomic_input.read_bytes()).hexdigest(),
        "subtasks_sha256": hashlib.sha256(args.subtasks.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
