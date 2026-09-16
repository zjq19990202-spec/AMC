#!/usr/bin/env python3
"""Build a clean Vase sidecar solely from the v11 manual-review directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROTATE = "Rotate the vase counterclockwise, then press it down onto the table"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merged-source-offset", type=int, default=150)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")

    source_dirs = sorted(
        args.source.glob("episode_*"), key=lambda path: int(path.name.split("_")[-1])
    )
    original_ids = [int(path.name.split("_")[-1]) for path in source_dirs]
    if len(original_ids) != 167:
        raise ValueError(f"expected 167 accepted episodes, found {len(original_ids)}")
    missing = sorted(set(range(min(original_ids), max(original_ids) + 1)) - set(original_ids))

    result_root = args.output / "v11_final"
    meta_root = args.output / "meta"
    result_root.mkdir(parents=True)
    meta_root.mkdir(parents=True)
    canonical = []
    merge_audit = []
    source_map = []
    for new_index, (original_id, source_dir) in enumerate(zip(original_ids, source_dirs, strict=True)):
        source_file = source_dir / "result.v11_final_memory.json"
        data = json.loads(source_file.read_text(encoding="utf-8"))
        parsed = data.get("parsed") or {}
        segments = parsed.get("subtasks")
        if not isinstance(segments, list) or not segments:
            raise ValueError(f"missing parsed.subtasks: {source_file}")

        merged_segments = []
        for segment in segments:
            segment = dict(segment)
            if (
                merged_segments
                and merged_segments[-1].get("current_subtask") == ROTATE
                and segment.get("current_subtask") == ROTATE
            ):
                if abs(float(merged_segments[-1]["end_s"]) - float(segment["start_s"])) > 0.002:
                    raise ValueError(f"non-contiguous rotate segments: {source_file}")
                merge_audit.append(
                    {
                        "source_episode_index": original_id,
                        "removed_boundary_s": segment["start_s"],
                    }
                )
                merged_segments[-1]["end_s"] = segment["end_s"]
            else:
                merged_segments.append(segment)

        for subtask_id, segment in enumerate(merged_segments):
            segment["subtask_id"] = subtask_id
        parsed["subtasks"] = merged_segments
        data["parsed"] = parsed
        data["source_episode_index"] = original_id
        data["accepted_episode_index"] = new_index
        destination = result_root / f"episode_{new_index:06d}"
        destination.mkdir()
        (destination / "result.v11_final_memory.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        semantic_segments = []
        for segment in merged_segments:
            start_frame = round(float(segment["start_s"]) * 30)
            end_frame = round(float(segment["end_s"]) * 30)
            semantic_segments.append(
                {
                    "start_frame_30hz": start_frame,
                    "end_frame_30hz_exclusive": end_frame,
                    "current_subtask": segment["current_subtask"],
                }
            )
        for previous, current in zip(semantic_segments, semantic_segments[1:], strict=False):
            if previous["end_frame_30hz_exclusive"] != current["start_frame_30hz"]:
                raise ValueError(f"30 Hz continuity mismatch in source episode {original_id}")
        canonical.append(
            {
                "episode_index": new_index,
                "semantic_segments": semantic_segments,
                "split_from_episode_index": args.merged_source_offset + new_index,
                "v11_source_episode_index": original_id,
            }
        )
        source_map.append(
            {
                "accepted_episode_index": new_index,
                "v11_source_episode_index": original_id,
                "merged_source_episode_index": args.merged_source_offset + new_index,
            }
        )

    sidecar = meta_root / "episode_subtasks.jsonl"
    sidecar.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in canonical),
        encoding="utf-8",
    )
    (meta_root / "source_episode_map.json").write_text(
        json.dumps(source_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (meta_root / "merge_consecutive_rotate_audit.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merge_audit),
        encoding="utf-8",
    )
    manifest = {
        "source": str(args.source),
        "accepted_episodes": len(canonical),
        "rejected_episode_indices_absent_from_source": missing,
        "merged_consecutive_rotate_boundaries": len(merge_audit),
        "other_boundaries_modified": 0,
        "episode_subtasks_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
    }
    (args.output / "BUILD_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
