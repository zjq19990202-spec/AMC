from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from atomic_latent_vla.data.hdf5 import read_policy_timestamps
from atomic_latent_vla.data.schema import EpisodeAnnotation, load_annotation


def build_window_records(
    hdf5_path: str | Path,
    annotation: EpisodeAnnotation,
    timestamps_s: np.ndarray,
    *,
    horizon: int = 50,
    stride: int = 1,
    min_duration_s: float = 2.0,
    include_unlabeled: bool = True,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    timestamps_s = np.asarray(timestamps_s, dtype=np.float64)
    for segment in annotation.segments:
        if not segment.training_eligible or segment.duration_s < min_duration_s:
            continue
        if len(segment.targets) > 2 or (not segment.targets and not include_unlabeled):
            continue
        start = int(np.searchsorted(timestamps_s, segment.start_s, side="left"))
        # Segments are half-open [start, end): an action exactly at end_s belongs
        # to the following segment and must never leak into this window.
        end = int(np.searchsorted(timestamps_s, segment.end_s, side="left"))
        last_start = end - horizon
        # Intentionally keep only complete action chunks for now.  A future
        # partial-window path must carry an `action_valid_mask` through the
        # Dataset/Collator/PolicyBatch and apply it to Flow Matching loss; merely
        # padding the tail would incorrectly supervise the artificial values.
        if last_start < start:
            continue
        labels = [target.label for target in segment.targets]
        weights = [target.confidence for target in segment.targets]
        while len(labels) < 2:
            labels.append(-1)
            weights.append(0.0)
        for window_start in range(start, last_start + 1, stride):
            records.append(
                {
                    "episode_id": annotation.episode_id,
                    "hdf5_path": str(Path(hdf5_path).expanduser().resolve()),
                    "start_idx": window_start,
                    "end_idx": window_start + horizon,
                    "start_s": float(timestamps_s[window_start]),
                    "segment_id": segment.segment_id,
                    "segment_start_s": segment.start_s,
                    "segment_end_s": segment.end_s,
                    "task": annotation.task,
                    "global_description": annotation.global_description,
                    "instruction": segment.instruction,
                    "atomic_mode": segment.atomic_mode,
                    "atomic_labels": labels,
                    "atomic_weights": weights,
                    "atomic_supervision_mask": bool(segment.targets),
                    "quantity_target": (
                        float(annotation.raw["quantity_value"])
                        / float(annotation.raw.get("quantity_scale") or 1.0)
                        if annotation.raw.get("quantity_value") is not None
                        else 0.0
                    ),
                    "quantity_valid": annotation.raw.get("quantity_value") is not None,
                    "quantity_unit": annotation.raw.get("quantity_unit"),
                }
            )
    return records


def write_jsonl(records: Iterable[dict[str, Any]], output: str | Path) -> int:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an atomic-latent CR1 window index"
    )
    parser.add_argument("--hdf5", required=True, type=Path)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min-duration-s", type=float, default=2.0)
    parser.add_argument(
        "--exclude-unlabeled",
        action="store_true",
        help="drop otherwise eligible trajectory windows that have no atomic label",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    annotation = load_annotation(args.annotation)
    timestamps = read_policy_timestamps(args.hdf5)
    records = build_window_records(
        args.hdf5,
        annotation,
        timestamps,
        horizon=args.horizon,
        stride=args.stride,
        min_duration_s=args.min_duration_s,
        include_unlabeled=not args.exclude_unlabeled,
    )
    count = write_jsonl(records, args.output)
    print(f"wrote {count} windows to {args.output}")


if __name__ == "__main__":
    main()
