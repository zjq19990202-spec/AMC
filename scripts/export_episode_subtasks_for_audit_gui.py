#!/usr/bin/env python3
"""Export canonical episode_subtasks.jsonl into reviewer_02 GUI sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--task", default="wipe the vase.")
    args = parser.parse_args()

    output_root = args.dataset_root / "sidecars" / "submem" / "v11_final"
    count = 0
    with args.jsonl.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            episode_index = int(record["episode_index"])
            segments = []
            for subtask_id, segment in enumerate(record["semantic_segments"]):
                start_frame = int(segment["start_frame_30hz"])
                end_frame = int(segment["end_frame_30hz_exclusive"])
                segments.append(
                    {
                        "subtask_id": subtask_id,
                        "start_s": round(start_frame / 30.0, 6),
                        "end_s": round(end_frame / 30.0, 6),
                        "current_subtask": segment["current_subtask"],
                        "confidence": "manual",
                        "boundary_reason": "canonical_sidecar_20260824",
                        "evidence": "Imported from training canonical episode_subtasks.jsonl",
                    }
                )

            duration_s = segments[-1]["end_s"] if segments else 0.0
            payload = {
                "dataset": args.dataset_root.name,
                "episode_id": f"episode_{episode_index:06d}",
                "episode_index": episode_index,
                "source_episode_index": record.get("split_from_episode_index"),
                "duration_s": duration_s,
                "task": args.task,
                "episode_task": args.task,
                "status": "needs_review",
                "quality_flags": [],
                "episode_summary": "",
                "semantic_segments": segments,
            }
            episode_dir = output_root / f"episode_{episode_index:06d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            destination = episode_dir / "result.v11_final_memory.json"
            destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            count += 1

    print(f"exported {count} episodes to {output_root}")


if __name__ == "__main__":
    main()
