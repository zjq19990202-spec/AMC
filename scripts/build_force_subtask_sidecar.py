#!/usr/bin/env python3
"""Align a reviewed 30 Hz subtask sidecar to a causal force dataset.

The force reconstructions may omit a small number of unreviewed episodes and,
for the Vase conversion, the first source frame.  This utility makes those
choices explicit and emits strict, gap-free coverage in force-dataset frame
coordinates.  It never invents a global-prompt fallback.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_lengths(root: Path) -> dict[int, int]:
    result: dict[int, int] = {}
    for path in sorted((root / "data").glob("chunk-*/*.parquet")):
        values = np.asarray(pq.read_table(path, columns=["episode_index"])["episode_index"])
        ids, counts = np.unique(values, return_counts=True)
        for episode, count in zip(ids, counts, strict=True):
            key = int(episode)
            result[key] = result.get(key, 0) + int(count)
    if not result:
        raise ValueError(f"no episode rows found under {root / 'data'}")
    return result


def _parse_ints(value: str) -> set[int]:
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-root", type=Path, required=True)
    parser.add_argument("--source-sidecar", type=Path, required=True)
    parser.add_argument("--source-start-row", type=int, default=0)
    parser.add_argument("--source-count", type=int, required=True)
    parser.add_argument("--skip-force-episodes", default="")
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument("--max-length-difference", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lengths = _episode_lengths(args.force_root)
    force_episodes = sorted(lengths)
    skipped = _parse_ints(args.skip_force_episodes)
    unknown_skips = skipped - set(force_episodes)
    if unknown_skips:
        raise ValueError(f"skip list contains unknown force episodes: {sorted(unknown_skips)}")
    retained = [episode for episode in force_episodes if episode not in skipped]

    source_rows = [
        json.loads(line)
        for line in args.source_sidecar.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = source_rows[
        args.source_start_row : args.source_start_row + args.source_count
    ]
    if len(selected) != args.source_count:
        raise ValueError(
            f"requested {args.source_count} source rows at {args.source_start_row}, "
            f"found {len(selected)}"
        )
    if len(retained) != len(selected):
        raise ValueError(
            f"retained force episodes ({len(retained)}) do not match source rows "
            f"({len(selected)})"
        )

    output_rows: list[dict[str, object]] = []
    mappings: list[dict[str, int]] = []
    prompts: set[str] = set()
    for force_episode, source_row in zip(retained, selected, strict=True):
        source_segments = source_row.get("semantic_segments", ())
        if not source_segments:
            raise ValueError(f"source episode {source_row.get('episode_index')} has no segments")
        force_length = lengths[force_episode]
        source_end = int(source_segments[-1]["end_frame_30hz_exclusive"])
        expected_length = source_end + args.frame_offset
        if abs(force_length - expected_length) > args.max_length_difference:
            raise ValueError(
                f"length mismatch force episode {force_episode}: force={force_length}, "
                f"source-adjusted={expected_length}"
            )

        converted: list[dict[str, object]] = []
        previous_end = 0
        for index, source_segment in enumerate(source_segments):
            start = max(0, int(source_segment["start_frame_30hz"]) + args.frame_offset)
            end = max(0, int(source_segment["end_frame_30hz_exclusive"]) + args.frame_offset)
            start = previous_end if index else 0
            end = force_length if index == len(source_segments) - 1 else min(end, force_length)
            if end <= start:
                raise ValueError(
                    f"empty adjusted segment for force episode {force_episode}: [{start}, {end})"
                )
            text = str(source_segment.get("current_subtask", "")).strip()
            if not text:
                raise ValueError(f"empty prompt for force episode {force_episode}")
            prompts.add(text)
            converted.append(
                {
                    "start_frame_30hz": start,
                    "end_frame_30hz_exclusive": end,
                    "current_subtask": text,
                }
            )
            previous_end = end
        output_rows.append(
            {
                "episode_index": force_episode,
                "semantic_segments": converted,
                "aligned_from_episode_index": int(source_row["episode_index"]),
            }
        )
        mappings.append(
            {
                "force_episode_index": force_episode,
                "source_episode_index": int(source_row["episode_index"]),
                "force_frames": force_length,
                "source_end_frame": source_end,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    manifest = {
        "force_root": str(args.force_root.resolve()),
        "source_sidecar": str(args.source_sidecar.resolve()),
        "source_sidecar_sha256": _sha256(args.source_sidecar),
        "output_sidecar": str(args.output.resolve()),
        "output_sidecar_sha256": _sha256(args.output),
        "frame_offset": args.frame_offset,
        "excluded_force_episodes": sorted(skipped),
        "retained_episodes": len(output_rows),
        "unique_subtasks": sorted(prompts),
        "mapping": mappings,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in (
        "output_sidecar", "output_sidecar_sha256", "frame_offset",
        "excluded_force_episodes", "retained_episodes", "unique_subtasks"
    )}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
