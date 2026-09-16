#!/usr/bin/env python3
"""Build a conservative, atomic-block-aligned episode-start exclusion manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads


def annotations(path: Path) -> dict[int, dict[int, dict[str, tuple[str, ...]]]]:
    result: dict[int, dict[int, dict[str, tuple[str, ...]]]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            arm = str(row.get("arm"))
            if arm not in {"left", "right"}:
                continue
            episode = int(row["episode_index"])
            labels = tuple(str(x) for x in row.get("fk_atomic_labels", []))
            for block in range(int(row["block_start_id"]), int(row["block_end_id"]) + 1):
                result.setdefault(episode, {}).setdefault(block, {})[arm] = labels
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--max-start-blocks", type=int, default=15)
    args = parser.parse_args()

    reports = []
    for root in args.dataset_root:
        atom = annotations(root / "meta/atomic_horizon_prompts_3hz.jsonl")
        table = pads.dataset(str(root / "data"), format="parquet").to_table(
            columns=["episode_index", "frame_index", "action"]
        )
        episode = np.asarray(table["episode_index"])
        frame = np.asarray(table["frame_index"])
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        changes = np.flatnonzero(np.diff(episode)) + 1
        starts, ends = np.r_[0, changes], np.r_[changes, len(episode)]
        records = []
        for lo, hi in zip(starts, ends, strict=True):
            ep = int(episode[lo])
            block_rows = []
            scan_started = False
            for block in range(args.max_start_blocks):
                labels = atom.get(ep, {}).get(block, {})
                left, right = labels.get("left"), labels.get("right")
                left_active = left is not None and left != ("stay",)
                right_active = right is not None and right != ("stay",)
                # Only a fully annotated, unequivocally single-active block is eligible.
                if left is None or right is None or left_active == right_active:
                    # Leading both-stay/unannotated blocks precede the actual task motion.
                    # Once single-arm motion has started, ambiguity terminates the prefix scan.
                    if scan_started:
                        break
                    continue
                scan_started = True
                inactive = "right" if left_active else "left"
                local = block * 10
                if local + args.horizon > hi - lo:
                    break
                if int(frame[lo + local + args.horizon - 1]) != int(frame[lo + local]) + args.horizon - 1:
                    break
                sl = slice(8, 15) if inactive == "right" else slice(0, 7)
                delta = float(
                    np.linalg.norm(
                        action[lo + local + args.horizon - 1, sl] - action[lo + local, sl]
                    )
                )
                block_rows.append((block, inactive, delta))
                # The first stable block terminates the removable start prefix.
                if delta <= args.threshold:
                    break
            bad = [row for row in block_rows if row[2] > args.threshold]
            if not bad:
                continue
            last_block = bad[-1][0]
            records.append(
                {
                    "episode_index": ep,
                    "mask_start_frame": 0,
                    "mask_end_frame_exclusive": (last_block + 1) * 10,
                    "mask_start_block": 0,
                    "mask_end_block_exclusive": last_block + 1,
                    "mask_seconds": (last_block + 1) / 3,
                    "inactive_arm": bad[0][1],
                    "block_deltas_rad": [
                        {"block": b, "inactive_arm": arm, "delta_rad": delta}
                        for b, arm, delta in block_rows
                    ],
                }
            )
        reports.append(
            {
                "dataset_root": str(root),
                "episodes_total": len(starts),
                "episodes_masked": len(records),
                "masked_seconds": sum(row["mask_seconds"] for row in records),
                "records": records,
            }
        )
    payload = {
        "contract": {
            "scope": "episode-start consecutive prefix only",
            "fps": 30,
            "atomic_block_frames": 10,
            "active_arm": "exactly one arm has a non-stay atomic label",
            "ambiguous_or_dual_block": "stop without masking it",
            "stable_block": "first inactive-arm 50-step joint delta <= threshold stops scan",
            "application": "exclude training horizon starts before mask_end_frame_exclusive",
            "source_data_modified": False,
            "threshold_rad": args.threshold,
            "horizon_steps": args.horizon,
        },
        "datasets": reports,
        "summary": {
            "episodes_total": sum(x["episodes_total"] for x in reports),
            "episodes_masked": sum(x["episodes_masked"] for x in reports),
            "masked_seconds": sum(x["masked_seconds"] for x in reports),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload["summary"], indent=2))
    for report in reports:
        print(Path(report["dataset_root"]).name, report["episodes_masked"], report["masked_seconds"])


if __name__ == "__main__":
    main()
