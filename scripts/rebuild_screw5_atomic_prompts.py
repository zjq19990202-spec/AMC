#!/usr/bin/env python3
"""Rebind immutable Screw5 FK atoms to the corrected five-stage sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PREFIX = {
    "move_x_pos": "Move forward", "move_x_neg": "Move backward",
    "move_y_pos": "Move leftward", "move_y_neg": "Move rightward",
    "move_z_pos": "Move upward", "move_z_neg": "Move downward",
    "rotate_x_pos": "Rotate positive about base-frame +x",
    "rotate_x_neg": "Rotate negative about base-frame +x",
    "rotate_y_pos": "Rotate positive about base-frame +y",
    "rotate_y_neg": "Rotate negative about base-frame +y",
    "rotate_z_pos": "Rotate positive about base-frame +z",
    "rotate_z_neg": "Rotate negative about base-frame +z",
    "stay": "Stay stationary",
}

PICK = "Pick up the screwdriver."
ORIENT = "Orient the screwdriver toward the screw."
POSITION = "Move the screwdriver tip into the screw-fastening position."
FASTEN = "Press and fasten the screw with the screwdriver."
RETRACT = "Release the screwdriver and retract to idle."
VALID_SUBTASKS = {PICK, ORIENT, POSITION, FASTEN, RETRACT}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def relation(text: str, progress: float, stay: bool) -> str:
    if stay:
        return {
            PICK: "relative to the screwdriver",
            ORIENT: "relative to the screwdriver and screw",
            POSITION: "relative to the screw head",
            FASTEN: "relative to the screw while maintaining fastening pressure",
            RETRACT: "relative to the screwdriver release and idle path",
        }[text]
    if text == PICK:
        return "toward the screwdriver" if progress < 0.62 else "at the screwdriver handle"
    if text == ORIENT:
        return "around the screwdriver to orient it toward the screw"
    if text == POSITION:
        return "toward the screw head" if progress < 0.62 else "into the screw-fastening position"
    if text == FASTEN:
        return "into the screw while maintaining fastening pressure"
    if text == RETRACT:
        return "away from the screw toward the idle position"
    raise ValueError(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atomic-input", type=Path, required=True)
    parser.add_argument("--subtasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.audit.exists():
        raise FileExistsError("refusing to overwrite existing output")

    subtasks = {}
    for row in read_jsonl(args.subtasks):
        segments = row["semantic_segments"]
        unknown = {item["current_subtask"] for item in segments} - VALID_SUBTASKS
        if unknown:
            raise ValueError(f"episode {row['episode_index']} has unsupported subtasks: {unknown}")
        subtasks[int(row["episode_index"])] = segments

    def at(episode: int, block: int) -> dict[str, Any]:
        frame = block * 10
        for segment in subtasks[episode]:
            if int(segment["start_frame_30hz"]) <= frame < int(segment["end_frame_30hz_exclusive"]):
                return segment
        return subtasks[episode][-1]

    output = []
    audit = []
    counts = Counter()
    for source in read_jsonl(args.atomic_input):
        row = dict(source)
        episode = int(row["episode_index"])
        first, last = int(row["block_start_id"]), int(row["block_end_id"])
        labels = list(row["fk_atomic_labels"])
        prefix = " and ".join(PREFIX[label] for label in labels)
        identities = {
            (int(item["start_frame_30hz"]), int(item["end_frame_30hz_exclusive"]), item["current_subtask"])
            for item in (at(episode, block) for block in range(first, last + 1))
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
            rel = None
            counts["cross_subtask_object_free"] += 1
        else:
            start, end, text = next(iter(identities))
            midpoint_s = ((first / 3.0) + ((last + 1) / 3.0)) / 2.0
            progress = min(1.0, max(0.0, (midpoint_s - start / 30.0) / max((end - start) / 30.0, 1e-6)))
            rel = relation(text, progress, labels == ["stay"])
            prompt = f"{prefix} {rel}."
            context = {
                "kind": "within_subtask",
                "subtask": {"text": text, "source_text": text, "start_s": start / 30.0, "end_s": end / 30.0},
            }
            counts["within_subtask_object_aware"] += 1
        row.update(
            fk_prefix=prefix,
            prompt=prompt,
            subtask_context=context,
            prompt_source="fk_screw5_corrected_subtask_relation_v3",
        )
        output.append(row)
        audit.append({
            "episode_index": episode, "horizon_id": row["horizon_id"], "arm": row["arm"],
            "block_start_id": first, "block_end_id": last,
            "fk_atomic_labels_unchanged": labels == source["fk_atomic_labels"],
            "source_prompt": source["prompt"], "final_prompt": prompt,
            "context_kind": context["kind"], "relation": rel,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    args.audit.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit), encoding="utf-8")
    print(json.dumps({
        "rows": len(output), **counts,
        "atomic_input_sha256": hashlib.sha256(args.atomic_input.read_bytes()).hexdigest(),
        "subtasks_sha256": hashlib.sha256(args.subtasks.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()
