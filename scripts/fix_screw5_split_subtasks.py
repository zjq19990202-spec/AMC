#!/usr/bin/env python3
"""Restore the reviewed five-stage screw subtasks in a domain-split copy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = (
    "Pick up the screwdriver.",
    "Orient the screwdriver toward the screw.",
    "Move the screwdriver tip into the screw-fastening position.",
    "Press and fasten the screw with the screwdriver.",
    "Release the screwdriver and retract to idle.",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def segment_texts(row: dict) -> tuple[str, ...]:
    return tuple(segment["current_subtask"] for segment in row["semantic_segments"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    args = parser.parse_args()

    source_rows = read_jsonl(args.source / "meta" / "episode_subtasks.jsonl")
    reviewed = {
        int(row["source_episode_index_before_v3"]): row
        for row in source_rows
        if row.get("source_dataset") == "screw_fastening_fixed"
        and segment_texts(row) == EXPECTED
    }
    if len(reviewed) != 213:
        raise ValueError(f"expected 213 reviewed five-stage rows, found {len(reviewed)}")

    split_path = args.split / "meta" / "episode_subtasks.jsonl"
    split_rows = read_jsonl(split_path)
    if len(split_rows) != 213:
        raise ValueError(f"expected 213 split rows, found {len(split_rows)}")

    corrected: list[dict] = []
    for split_row in split_rows:
        source_key = int(split_row["source_episode_index_before_v3"])
        source_row = reviewed[source_key]
        row = dict(split_row)
        row["semantic_segments"] = source_row["semantic_segments"]
        if segment_texts(row) != EXPECTED:
            raise AssertionError(f"episode {row['episode_index']} did not receive reviewed stages")
        corrected.append(row)
    write_jsonl(split_path, corrected)

    global_path = args.split / "meta" / "global_episode_prompts.jsonl"
    global_rows = read_jsonl(global_path)
    for row in global_rows:
        row["subtasks"] = list(EXPECTED)
        row["subtask_authority"] = "reviewed_five_stage_source_sidecar"
    write_jsonl(global_path, global_rows)

    print(f"corrected={len(corrected)} expected_stages={list(EXPECTED)}")


if __name__ == "__main__":
    main()
