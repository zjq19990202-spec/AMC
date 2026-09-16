#!/usr/bin/env python3
"""Merge adjacent identical Vase rotate segments without changing frame labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROTATE = "Rotate the vase counterclockwise, then press it down onto the table"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def merge(episodes: list[dict]) -> list[dict]:
    audit = []
    for episode in episodes:
        output = []
        for segment in episode["semantic_segments"]:
            if (
                output
                and output[-1]["current_subtask"] == ROTATE
                and segment["current_subtask"] == ROTATE
                and output[-1]["end_frame_30hz_exclusive"]
                == segment["start_frame_30hz"]
            ):
                audit.append(
                    {
                        "episode_index": episode["episode_index"],
                        "removed_boundary_frame": segment["start_frame_30hz"],
                        "merged_start_frame": output[-1]["start_frame_30hz"],
                        "merged_end_frame": segment["end_frame_30hz_exclusive"],
                    }
                )
                output[-1]["end_frame_30hz_exclusive"] = segment[
                    "end_frame_30hz_exclusive"
                ]
            else:
                output.append(segment)
        episode["semantic_segments"] = output
    return audit


def write_atomic(path: Path, episodes: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for episode in episodes:
            stream.write(json.dumps(episode, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-sidecar", type=Path, required=True)
    parser.add_argument("--merged-sidecar", type=Path, required=True)
    parser.add_argument("--audit-jsonl", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--backup-suffix", default=".before_merge_consecutive_rotate_20260824"
    )
    args = parser.parse_args()

    split = load(args.split_sidecar)
    merged = load(args.merged_sidecar)
    before_split = sha256(args.split_sidecar)
    before_merged = sha256(args.merged_sidecar)
    split_audit = merge(split)
    merged_audit = merge(merged)

    # The split is a renumbered subset of the merged source.  Every removed
    # boundary must have the same source episode/frame counterpart.
    split_by_source = {
        (item.get("split_from_episode_index"), row["removed_boundary_frame"])
        for item in split
        for row in split_audit
        if row["episode_index"] == item["episode_index"]
    }
    merged_keys = {
        (row["episode_index"], row["removed_boundary_frame"]) for row in merged_audit
    }
    if split_by_source != merged_keys:
        raise ValueError(
            f"split/merged merge mapping differs: split={len(split_by_source)} "
            f"merged={len(merged_keys)}"
        )

    args.audit_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.audit_jsonl.open("w") as stream:
        for row in split_audit:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    backups = [Path(str(args.split_sidecar) + args.backup_suffix), Path(str(args.merged_sidecar) + args.backup_suffix)]
    after_split = after_merged = None
    if args.apply:
        for backup in backups:
            if backup.exists():
                raise FileExistsError(f"refusing to overwrite backup: {backup}")
        shutil.copy2(args.split_sidecar, backups[0])
        shutil.copy2(args.merged_sidecar, backups[1])
        write_atomic(args.split_sidecar, split)
        write_atomic(args.merged_sidecar, merged)
        after_split = sha256(args.split_sidecar)
        after_merged = sha256(args.merged_sidecar)

    print(
        json.dumps(
            {
                "apply": args.apply,
                "removed_split_boundaries": len(split_audit),
                "removed_merged_boundaries": len(merged_audit),
                "before_split_sha256": before_split,
                "after_split_sha256": after_split,
                "before_merged_sha256": before_merged,
                "after_merged_sha256": after_merged,
                "backups": [str(path) for path in backups] if args.apply else [],
                "audit_jsonl": str(args.audit_jsonl),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
